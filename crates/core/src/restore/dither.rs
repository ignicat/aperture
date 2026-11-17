use crate::color::WorkingSpace;
use crate::error::EngineResult;
use crate::pixel::PlanarF32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitherMode {
    None,
    Bayer,
    FloydSteinberg,
    BlueNoise,
}

impl DitherMode {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "none" | "off" => Some(Self::None),
            "bayer" | "ordered" => Some(Self::Bayer),
            "floyd" | "floyd-steinberg" | "fs" => Some(Self::FloydSteinberg),
            "blue" | "blue-noise" | "bluenoise" => Some(Self::BlueNoise),
            _ => None,
        }
    }
}

const BAYER_4X4: [f32; 16] = [
    -0.5, 0.0, -0.375, 0.125,
    0.25, -0.25, 0.375, -0.125,
    -0.3125, 0.1875, -0.4375, 0.0625,
    0.4375, -0.0625, 0.3125, -0.1875,
];

pub fn apply_dither_and_grain(
    src: &PlanarF32,
    mode: DitherMode,
    grain_amount: f32,
    levels: f32,
) -> EngineResult<PlanarF32> {
    if mode == DitherMode::None && grain_amount <= 0.001 {
        return Ok(src.clone());
    }
    let orig_space = src.working_space;
    let mut work = src.clone();
    work.convert_working_space(WorkingSpace::PerceptualSrgb);

    let w = work.width as usize;
    let h = work.height as usize;
    let q_step = 1.0 / levels.max(2.0);
    let grain_amp = (grain_amount.clamp(0.0, 1.0)) * 0.045;

    match mode {
        DitherMode::None => {}
        DitherMode::Bayer => {
            for y in 0..h {
                for x in 0..w {
                    let t = BAYER_4X4[(y & 3) * 4 + (x & 3)] * q_step;
                    let i = y * w + x;
                    work.r[i] = (work.r[i] + t).clamp(0.0, 1.0);
                    work.g[i] = (work.g[i] + t).clamp(0.0, 1.0);
                    work.b[i] = (work.b[i] + t).clamp(0.0, 1.0);
                }
            }
        }
        DitherMode::BlueNoise => {
            let bn = build_blue_noise_tile_16x16();
            for y in 0..h {
                for x in 0..w {
                    let t = (bn[(y & 15) * 16 + (x & 15)] - 0.5) * q_step;
                    let i = y * w + x;
                    work.r[i] = (work.r[i] + t).clamp(0.0, 1.0);
                    work.g[i] = (work.g[i] + t).clamp(0.0, 1.0);
                    work.b[i] = (work.b[i] + t).clamp(0.0, 1.0);
                }
            }
        }
        DitherMode::FloydSteinberg => {
            floyd_steinberg_channel(&mut work.r, w, h, levels);
            floyd_steinberg_channel(&mut work.g, w, h, levels);
            floyd_steinberg_channel(&mut work.b, w, h, levels);
        }
    }

    if grain_amp > 1e-4 {
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let lum = 0.2126 * work.r[i] + 0.7152 * work.g[i] + 0.0722 * work.b[i];
                let midtone_mask = (4.0 * lum * (1.0 - lum)).clamp(0.15, 1.0);
                let n = triangular_noise(x as u32, y as u32) * grain_amp * midtone_mask;
                work.r[i] = (work.r[i] + n).clamp(0.0, 1.0);
                work.g[i] = (work.g[i] + n).clamp(0.0, 1.0);
                work.b[i] = (work.b[i] + n).clamp(0.0, 1.0);
            }
        }
    }

    work.convert_working_space(orig_space);
    Ok(work)
}

fn floyd_steinberg_channel(plane: &mut [f32], w: usize, h: usize, levels: f32) {
    let scale = levels.max(2.0);
    let inv = 1.0 / scale;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let old = plane[i];
            let new_val = (old * scale).round() * inv;
            let err = old - new_val;
            plane[i] = new_val.clamp(0.0, 1.0);
            if x + 1 < w {
                plane[i + 1] += err * (7.0 / 16.0);
            }
            if y + 1 < h {
                if x > 0 {
                    plane[i + w - 1] += err * (3.0 / 16.0);
                }
                plane[i + w] += err * (5.0 / 16.0);
                if x + 1 < w {
                    plane[i + w + 1] += err * (1.0 / 16.0);
                }
            }
        }
    }
}

fn build_blue_noise_tile_16x16() -> [f32; 256] {
    let mut tile = [0.0f32; 256];
    let mut order = [0usize; 256];
    let mut placed = [false; 256];
    let mut energy = [0.0f32; 256];
    placed[0] = true;
    order[0] = 0;
    add_toroidal_gaussian(&mut energy, 0, 16);

    for rank in 1..256usize {
        let mut best_idx = 0usize;
        let mut min_e = f32::MAX;
        for (idx, &e) in energy.iter().enumerate() {
            if !placed[idx] && e < min_e {
                min_e = e;
                best_idx = idx;
            }
        }
        placed[best_idx] = true;
        order[best_idx] = rank;
        add_toroidal_gaussian(&mut energy, best_idx, 16);
    }

    for (i, &r) in order.iter().enumerate() {
        tile[i] = (r as f32 + 0.5) / 256.0;
    }
    tile
}

fn add_toroidal_gaussian(energy: &mut [f32; 256], seed_idx: usize, side: usize) {
    let sx = (seed_idx % side) as isize;
    let sy = (seed_idx / side) as isize;
    let half = (side / 2) as isize;
    for y in 0..side {
        let mut dy = (y as isize - sy).abs();
        if dy > half {
            dy = side as isize - dy;
        }
        for x in 0..side {
            let mut dx = (x as isize - sx).abs();
            if dx > half {
                dx = side as isize - dx;
            }
            let d2 = (dx * dx + dy * dy) as f32;
            energy[y * side + x] += (-d2 / (2.0 * 1.5 * 1.5)).exp();
        }
    }
}

#[inline]
fn triangular_noise(x: u32, y: u32) -> f32 {
    let mut h1 = x.wrapping_mul(0x27d4_eb2d) ^ y.wrapping_mul(0x1656_67b1);
    h1 ^= h1 >> 15;
    h1 = h1.wrapping_mul(0x85eb_ca6b);
    let u1 = (h1 & 0xFFFF) as f32 / 65535.0;
    let u2 = ((h1 >> 16) & 0xFFFF) as f32 / 65535.0;
    u1 - u2
}
