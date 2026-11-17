use crate::pixel::PlanarF32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentKind {
    Photo,
    LineArt,
    PixelArt,
    UiText,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageProfile {
    pub kind: ContentKind,
    pub unique_color_ratio: f32,
    pub flat_ratio: f32,
    pub strong_edge_ratio: f32,
    pub axis_aligned_ratio: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileMetrics {
    pub edge_strength: f32,
    pub variance: f32,
    pub pixel_blockiness: f32,
}

pub fn classify_image(img: &PlanarF32) -> ImageProfile {
    let w = img.width as usize;
    let h = img.height as usize;
    if w < 4 || h < 4 {
        return ImageProfile {
            kind: ContentKind::Photo,
            unique_color_ratio: 1.0,
            flat_ratio: 0.0,
            strong_edge_ratio: 0.0,
            axis_aligned_ratio: 0.0,
        };
    }

    let step_x = (w / 128).max(1);
    let step_y = (h / 128).max(1);
    let mut colors: Vec<u32> = Vec::with_capacity(2048);
    let mut flat_count = 0usize;
    let mut strong_edge_count = 0usize;
    let mut axis_edge_count = 0usize;
    let mut total_samples = 0usize;
    let mut exact_dup_steps = 0usize;

    let luma = |idx: usize| -> f32 {
        0.2126 * img.r[idx] + 0.7152 * img.g[idx] + 0.0722 * img.b[idx]
    };

    for y in (1..h - 1).step_by(step_y) {
        for x in (1..w - 1).step_by(step_x) {
            let idx = y * w + x;
            total_samples += 1;

            let q = ((img.r[idx].clamp(0.0, 1.0) * 63.0).round() as u32) << 12
                | ((img.g[idx].clamp(0.0, 1.0) * 63.0).round() as u32) << 6
                | (img.b[idx].clamp(0.0, 1.0) * 63.0).round() as u32;
            if colors.len() < 512 && !colors.contains(&q) {
                colors.push(q);
            }

            let lc = luma(idx);
            let lx = luma(idx + 1);
            let ly = luma(idx + w);
            let dx = (lx - lc).abs();
            let dy = (ly - lc).abs();

            if dx == 0.0 || dy == 0.0 {
                exact_dup_steps += 1;
            }
            let grad = dx.max(dy);
            if grad < 0.01 {
                flat_count += 1;
            } else if grad > 0.12 {
                strong_edge_count += 1;
                if (dx < 0.005 && dy > 0.1) || (dy < 0.005 && dx > 0.1) {
                    axis_edge_count += 1;
                }
            }
        }
    }

    let denom = total_samples.max(1) as f32;
    let unique_color_ratio = (colors.len() as f32) / denom;
    let flat_ratio = flat_count as f32 / denom;
    let strong_edge_ratio = strong_edge_count as f32 / denom;
    let axis_aligned_ratio = axis_edge_count as f32 / strong_edge_count.max(1) as f32;
    let dup_ratio = exact_dup_steps as f32 / denom;

    let kind = if w <= 512 && h <= 512 && colors.len() <= 96 && dup_ratio > 0.45 && flat_ratio > 0.55 {
        ContentKind::PixelArt
    } else if axis_aligned_ratio > 0.58 && flat_ratio > 0.50 {
        ContentKind::UiText
    } else if flat_ratio > 0.58 && strong_edge_ratio > 0.03 && colors.len() < 420 {
        ContentKind::LineArt
    } else {
        ContentKind::Photo
    };

    ImageProfile {
        kind,
        unique_color_ratio,
        flat_ratio,
        strong_edge_ratio,
        axis_aligned_ratio,
    }
}

pub fn classify_tiles_8x8(img: &PlanarF32) -> (usize, usize, Vec<TileMetrics>) {
    let w = img.width as usize;
    let h = img.height as usize;
    let tx_count = w.div_ceil(8);
    let ty_count = h.div_ceil(8);
    let mut tiles = Vec::with_capacity(tx_count * ty_count);

    for ty in 0..ty_count {
        let y0 = ty * 8;
        let y1 = (y0 + 8).min(h);
        for tx in 0..tx_count {
            let x0 = tx * 8;
            let x1 = (x0 + 8).min(w);
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            let mut edge_sum = 0.0f32;
            let mut zero_steps = 0usize;
            let mut count = 0usize;

            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = y * w + x;
                    let lum = 0.2126 * img.r[idx] + 0.7152 * img.g[idx] + 0.0722 * img.b[idx];
                    sum += lum;
                    sum_sq += lum * lum;
                    count += 1;
                    if x + 1 < x1 {
                        let r_lum = 0.2126 * img.r[idx + 1]
                            + 0.7152 * img.g[idx + 1]
                            + 0.0722 * img.b[idx + 1];
                        let d = (lum - r_lum).abs();
                        edge_sum += d;
                        if d < 1e-4 {
                            zero_steps += 1;
                        }
                    }
                    if y + 1 < y1 {
                        let b_lum = 0.2126 * img.r[idx + w]
                            + 0.7152 * img.g[idx + w]
                            + 0.0722 * img.b[idx + w];
                        edge_sum += (lum - b_lum).abs();
                    }
                }
            }
            let n = count.max(1) as f32;
            let mean = sum / n;
            let variance = (sum_sq / n - mean * mean).max(0.0);
            tiles.push(TileMetrics {
                edge_strength: edge_sum / n,
                variance,
                pixel_blockiness: zero_steps as f32 / n,
            });
        }
    }
    (tx_count, ty_count, tiles)
}

pub fn compute_dhash_64(img: &PlanarF32) -> u64 {
    let w = img.width as usize;
    let h = img.height as usize;
    if w == 0 || h == 0 {
        return 0;
    }
    let mut grid = [0.0f32; 72];
    for gy in 0..8usize {
        let sy = ((gy as f32 + 0.5) * h as f32 / 8.0) as usize;
        let y = sy.min(h - 1);
        for gx in 0..9usize {
            let sx = ((gx as f32 + 0.5) * w as f32 / 9.0) as usize;
            let x = sx.min(w - 1);
            let idx = y * w + x;
            grid[gy * 9 + gx] = 0.2126 * img.r[idx] + 0.7152 * img.g[idx] + 0.0722 * img.b[idx];
        }
    }
    let mut hash = 0u64;
    let mut bit = 0u32;
    for gy in 0..8usize {
        for gx in 0..8usize {
            if grid[gy * 9 + gx] > grid[gy * 9 + gx + 1] {
                hash |= 1u64 << bit;
            }
            bit += 1;
        }
    }
    hash
}

pub fn dhash_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

pub fn fnv1a_128(data: &[u8]) -> u128 {
    let mut hash: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
    for &b in data {
        hash ^= b as u128;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}
