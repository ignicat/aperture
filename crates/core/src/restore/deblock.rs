use crate::color::WorkingSpace;
use crate::error::EngineResult;
use crate::pixel::PlanarF32;
use rustdct::DctPlanner;

pub fn deblock_shifted_dct(src: &PlanarF32, strength: f32, high_quality: bool) -> EngineResult<PlanarF32> {
    if strength <= 0.001 || src.width < 8 || src.height < 8 {
        return Ok(src.clone());
    }
    let orig_space = src.working_space;
    let mut work = src.clone();
    work.convert_working_space(WorkingSpace::PerceptualSrgb);

    let w = work.width as usize;
    let h = work.height as usize;
    let len = w * h;

    let mut y_plane = vec![0.0f32; len];
    let mut cb_plane = vec![0.0f32; len];
    let mut cr_plane = vec![0.0f32; len];
    for i in 0..len {
        let r = work.r[i];
        let g = work.g[i];
        let b = work.b[i];
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        y_plane[i] = y;
        cb_plane[i] = (b - y) * 0.564_334;
        cr_plane[i] = (r - y) * 0.713_267;
    }

    let thresh_y = (strength.clamp(0.0, 1.0)) * 0.085;
    let thresh_c = thresh_y * 1.35;
    let shifts: &[(usize, usize)] = if high_quality {
        &[
            (0, 0),
            (4, 0),
            (0, 4),
            (4, 4),
            (2, 2),
            (6, 2),
            (2, 6),
            (6, 6),
        ]
    } else {
        &[(0, 0), (4, 0), (0, 4), (4, 4)]
    };

    let y_clean = spp_channel(&y_plane, w, h, thresh_y, shifts);
    let cb_clean = spp_channel(&cb_plane, w, h, thresh_c, shifts);
    let cr_clean = spp_channel(&cr_plane, w, h, thresh_c, shifts);

    for i in 0..len {
        let y = y_clean[i];
        let cb = cb_clean[i] / 0.564_334;
        let cr = cr_clean[i] / 0.713_267;
        let r = y + cr;
        let b = y + cb;
        let g = (y - 0.299 * r - 0.114 * b) / 0.587;
        work.r[i] = r.clamp(0.0, 1.0);
        work.g[i] = g.clamp(0.0, 1.0);
        work.b[i] = b.clamp(0.0, 1.0);
    }

    work.convert_working_space(orig_space);
    Ok(work)
}

fn spp_channel(
    plane: &[f32],
    w: usize,
    h: usize,
    threshold: f32,
    shifts: &[(usize, usize)],
) -> Vec<f32> {
    let mut planner = DctPlanner::new();
    let dct2 = planner.plan_dct2(8);
    let dct3 = planner.plan_dct3(8);

    let len = w * h;
    let mut acc = vec![0.0f32; len];
    let mut weight_sum = vec![0.0f32; len];

    let mut block = [0.0f32; 64];
    let mut tmp = [0.0f32; 64];
    let scale = 1.0 / 16.0;

    for &(ox, oy) in shifts {
        let mut by = oy;
        while by + 8 <= h {
            let mut bx = ox;
            while bx + 8 <= w {
                for r in 0..8 {
                    let s_off = (by + r) * w + bx;
                    block[r * 8..r * 8 + 8].copy_from_slice(&plane[s_off..s_off + 8]);
                    dct2.process_dct2(&mut block[r * 8..r * 8 + 8]);
                }
                for c in 0..8 {
                    let mut col = [0.0f32; 8];
                    for r in 0..8 {
                        col[r] = block[r * 8 + c];
                    }
                    dct2.process_dct2(&mut col);
                    for r in 0..8 {
                        tmp[r * 8 + c] = col[r] * scale;
                    }
                }

                for v in tmp.iter_mut().skip(1) {
                    let val = *v;
                    let mag = val.abs();
                    if mag <= threshold {
                        *v = 0.0;
                    } else {
                        *v = val.signum() * (mag - threshold * 0.6);
                    }
                }

                for c in 0..8 {
                    let mut col = [0.0f32; 8];
                    for r in 0..8 {
                        col[r] = tmp[r * 8 + c];
                    }
                    dct3.process_dct3(&mut col);
                    for r in 0..8 {
                        block[r * 8 + c] = col[r];
                    }
                }
                for r in 0..8 {
                    dct3.process_dct3(&mut block[r * 8..r * 8 + 8]);
                    let d_off = (by + r) * w + bx;
                    for c in 0..8 {
                        acc[d_off + c] += block[r * 8 + c] * scale;
                        weight_sum[d_off + c] += 1.0;
                    }
                }
                bx += 8;
            }
            by += 8;
        }
    }

    let mut out = plane.to_vec();
    for i in 0..len {
        if weight_sum[i] > 0.0 {
            out[i] = (acc[i] / weight_sum[i]).clamp(-1.0, 1.0);
        }
    }
    out
}
