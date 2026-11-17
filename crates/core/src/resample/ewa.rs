use crate::error::{EngineError, EngineResult};
use crate::pixel::PlanarF32;
use crate::pool::ThreadPool;
use crate::resample::kernels::{ResampleFilter, eval_ewa_kernel};

const EWA_LUT_LEN: usize = 4096;

pub fn resample_ewa_planar(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    filter: ResampleFilter,
    antiring: f32,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    if src.width == dst_w && src.height == dst_h {
        return Ok(src.clone());
    }
    if dst_w == 0 || dst_h == 0 {
        return Err(EngineError::InvalidDimensions {
            width: dst_w,
            height: dst_h,
        });
    }

    let base_r = match filter {
        ResampleFilter::EwaLanczos => 3.238_315_484,
        ResampleFilter::EwaLanczosSharp => 3.238_315_484 * 0.981_250_583_722_370_7,
        ResampleFilter::EwaGinseng => 3.238_315_484,
        ResampleFilter::EwaRobidoux | ResampleFilter::EwaRobidouxSharp => 2.0,
        _ => filter.base_radius(),
    };

    let mut w_lut = vec![0.0f32; EWA_LUT_LEN + 2];
    for (i, slot) in w_lut.iter_mut().enumerate().take(EWA_LUT_LEN + 1) {
        let r2 = i as f64 / EWA_LUT_LEN as f64;
        let r = r2.sqrt() * base_r;
        *slot = eval_ewa_kernel(filter, r) as f32;
    }

    let scale_x = src.width as f64 / dst_w as f64;
    let scale_y = src.height as f64 / dst_h as f64;
    let rx = base_r * scale_x.max(1.0);
    let ry = base_r * scale_y.max(1.0);
    let inv_rx = 1.0 / rx;
    let inv_ry = 1.0 / ry;
    let max_dx = rx.ceil() as i32;
    let max_dy = ry.ceil() as i32;

    let sw = src.width as usize;
    let sh = src.height as usize;
    let dw = dst_w as usize;
    let ar = antiring.clamp(0.0, 1.0);

    let mut dst = PlanarF32::new(dst_w, dst_h, src.has_alpha, src.working_space)?;
    dst.premultiplied = src.premultiplied;

    pool.parallel_row_stripes_4_mut(
        &mut dst.r,
        &mut dst.g,
        &mut dst.b,
        &mut dst.a,
        dw,
        |y0, sr, sg, sb, mut sa| {
            let rows = sr.len() / dw;
            for local_y in 0..rows {
                let dy = y0 + local_y;
                let src_cy = (dy as f64 + 0.5) * scale_y - 0.5;
                let cy_i = src_cy.floor() as i32;
                let y_min = (src_cy - ry).ceil() as i32;
                let y_max = (src_cy + ry).floor() as i32;
                let mut dy_offsets = Vec::with_capacity((max_dy * 2 + 2) as usize);
                for sy_i in y_min..=y_max {
                    let sy_clamped = sy_i.clamp(0, sh as i32 - 1) as usize;
                    let norm_dy = (sy_i as f64 - src_cy) * inv_ry;
                    dy_offsets.push((sy_clamped, (norm_dy * norm_dy) as f32));
                }

                for dx in 0..dw {
                    let src_cx = (dx as f64 + 0.5) * scale_x - 0.5;
                    let cx_i = src_cx.floor() as i32;
                    let x_min = (src_cx - rx).ceil() as i32;
                    let x_max = (src_cx + rx).floor() as i32;

                    let mut acc_r = 0.0f32;
                    let mut acc_g = 0.0f32;
                    let mut acc_b = 0.0f32;
                    let mut acc_a = 0.0f32;
                    let mut acc_w = 0.0f32;

                    let mut min_r = f32::MAX;
                    let mut max_r = f32::MIN;
                    let mut min_g = f32::MAX;
                    let mut max_g = f32::MIN;
                    let mut min_b = f32::MAX;
                    let mut max_b = f32::MIN;

                    if ar > 0.0 {
                        for oy in 0..=1 {
                            let sy = (cy_i + oy).clamp(0, sh as i32 - 1) as usize;
                            for ox in 0..=1 {
                                let sx = (cx_i + ox).clamp(0, sw as i32 - 1) as usize;
                                let sidx = sy * sw + sx;
                                min_r = min_r.min(src.r[sidx]);
                                max_r = max_r.max(src.r[sidx]);
                                min_g = min_g.min(src.g[sidx]);
                                max_g = max_g.max(src.g[sidx]);
                                min_b = min_b.min(src.b[sidx]);
                                max_b = max_b.max(src.b[sidx]);
                            }
                        }
                    }

                    let mut dx_table = Vec::with_capacity((max_dx * 2 + 2) as usize);
                    for sx_i in x_min..=x_max {
                        let sx_clamped = sx_i.clamp(0, sw as i32 - 1) as usize;
                        let norm_dx = (sx_i as f64 - src_cx) * inv_rx;
                        dx_table.push((sx_clamped, (norm_dx * norm_dx) as f32));
                    }

                    for &(sy_clamped, ndy2) in &dy_offsets {
                        let row_base = sy_clamped * sw;
                        for &(sx_clamped, ndx2) in &dx_table {
                            let r2 = ndx2 + ndy2;
                            if r2 >= 1.0 {
                                continue;
                            }
                            let lut_pos = r2 * EWA_LUT_LEN as f32;
                            let li = lut_pos as usize;
                            let frac = lut_pos - li as f32;
                            let w = w_lut[li] + (w_lut[li + 1] - w_lut[li]) * frac;
                            let sidx = row_base + sx_clamped;
                            acc_r += src.r[sidx] * w;
                            acc_g += src.g[sidx] * w;
                            acc_b += src.b[sidx] * w;
                            if src.has_alpha {
                                acc_a += src.a[sidx] * w;
                            }
                            acc_w += w;
                        }
                    }

                    let out_idx = local_y * dw + dx;
                    if acc_w.abs() > 1e-8 {
                        let inv_w = 1.0 / acc_w;
                        let mut vr = acc_r * inv_w;
                        let mut vg = acc_g * inv_w;
                        let mut vb = acc_b * inv_w;
                        if ar > 0.0 {
                            vr += (vr.clamp(min_r, max_r) - vr) * ar;
                            vg += (vg.clamp(min_g, max_g) - vg) * ar;
                            vb += (vb.clamp(min_b, max_b) - vb) * ar;
                        }
                        sr[out_idx] = vr;
                        sg[out_idx] = vg;
                        sb[out_idx] = vb;
                        if let Some(ref mut sa_buf) = sa {
                            sa_buf[out_idx] = (acc_a * inv_w).clamp(0.0, 1.0);
                        }
                    } else {
                        let sidx = cy_i.clamp(0, sh as i32 - 1) as usize * sw
                            + cx_i.clamp(0, sw as i32 - 1) as usize;
                        sr[out_idx] = src.r[sidx];
                        sg[out_idx] = src.g[sidx];
                        sb[out_idx] = src.b[sidx];
                        if let Some(ref mut sa_buf) = sa {
                            sa_buf[out_idx] = src.a[sidx];
                        }
                    }
                }
            }
        },
    );

    Ok(dst)
}
