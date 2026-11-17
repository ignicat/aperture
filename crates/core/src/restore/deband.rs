use crate::color::WorkingSpace;
use crate::error::EngineResult;
use crate::pixel::PlanarF32;
use crate::pool::ThreadPool;

pub fn deband_stochastic(
    src: &PlanarF32,
    strength: f32,
    iterations: u8,
    range: f32,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    if strength <= 0.001 || src.width < 4 || src.height < 4 {
        return Ok(src.clone());
    }
    let orig_space = src.working_space;
    let mut cur = src.clone();
    cur.convert_working_space(WorkingSpace::PerceptualSrgb);

    let w = cur.width as usize;
    let h = cur.height as usize;
    let base_thresh = (strength.clamp(0.0, 1.0)) * 0.055;
    let iters = iterations.clamp(1, 4);
    let max_range = range.clamp(4.0, 48.0);

    for iter in 0..iters {
        let snap_r = cur.r.clone();
        let snap_g = cur.g.clone();
        let snap_b = cur.b.clone();
        let iter_radius = max_range * ((iter as f32 + 1.0) / iters as f32);
        let iter_thresh = base_thresh / (iter as f32 + 1.0);

        pool.parallel_row_stripes_4_mut(
            &mut cur.r,
            &mut cur.g,
            &mut cur.b,
            &mut Vec::new(),
            w,
            |y0, sr, sg, sb, _| {
                let rows = sr.len() / w;
                for ly in 0..rows {
                    let y = y0 + ly;
                    for x in 0..w {
                        let seed = hash_pixel(x as u32, y as u32, iter as u32);
                        let u1 = (seed & 0xFFFF) as f32 / 65535.0;
                        let u2 = ((seed >> 16) & 0xFFFF) as f32 / 65535.0;
                        let dist = (u1 * iter_radius).max(1.0);
                        let angle = u2 * std::f32::consts::TAU;
                        let dx = (dist * angle.cos()).round() as isize;
                        let dy = (dist * angle.sin()).round() as isize;

                        let p_offsets = [
                            (dx, dy),
                            (-dx, -dy),
                            (-dy, dx),
                            (dy, -dx),
                        ];
                        let idx = y * w + x;
                        let cr = snap_r[idx];
                        let cg = snap_g[idx];
                        let cb = snap_b[idx];

                        let mut sum_r = 0.0f32;
                        let mut sum_g = 0.0f32;
                        let mut sum_b = 0.0f32;
                        let mut max_diff = 0.0f32;
                        for (ox, oy) in p_offsets {
                            let sx = (x as isize + ox).clamp(0, w as isize - 1) as usize;
                            let sy = (y as isize + oy).clamp(0, h as isize - 1) as usize;
                            let si = sy * w + sx;
                            let pr = snap_r[si];
                            let pg = snap_g[si];
                            let pb = snap_b[si];
                            sum_r += pr;
                            sum_g += pg;
                            sum_b += pb;
                            max_diff = max_diff
                                .max((pr - cr).abs())
                                .max((pg - cg).abs())
                                .max((pb - cb).abs());
                        }
                        let avg_r = sum_r * 0.25;
                        let avg_g = sum_g * 0.25;
                        let avg_b = sum_b * 0.25;
                        let avg_diff = (avg_r - cr)
                            .abs()
                            .max((avg_g - cg).abs())
                            .max((avg_b - cb).abs());

                        let oi = ly * w + x;
                        if avg_diff < iter_thresh && max_diff < iter_thresh * 2.2 {
                            let factor = 1.0 - (avg_diff / iter_thresh.max(1e-6)) * 0.35;
                            sr[oi] = cr + (avg_r - cr) * factor;
                            sg[oi] = cg + (avg_g - cg) * factor;
                            sb[oi] = cb + (avg_b - cb) * factor;
                        }
                    }
                }
            },
        );
    }

    cur.convert_working_space(orig_space);
    Ok(cur)
}

#[inline]
fn hash_pixel(x: u32, y: u32, iter: u32) -> u32 {
    let mut h = x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x85eb_ca6b) ^ iter.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}
