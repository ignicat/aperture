use crate::color::WorkingSpace;
use crate::error::{EngineError, EngineResult};
use crate::pixel::PlanarF32;
use crate::pool::ThreadPool;
use crate::resample::ewa::resample_ewa_planar;
use crate::resample::kernels::ResampleFilter;

pub fn upscale_fsr_easu_rcas(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    rcas_sharpness: f32,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    if dst_w == 0 || dst_h == 0 {
        return Err(EngineError::InvalidDimensions {
            width: dst_w,
            height: dst_h,
        });
    }
    if dst_w <= src.width && dst_h <= src.height {
        return resample_ewa_planar(src, dst_w, dst_h, ResampleFilter::EwaLanczosSharp, 0.5, pool);
    }

    let orig_space = src.working_space;
    let was_premul = src.premultiplied;
    let mut work = src.clone();
    if work.premultiplied {
        work.unpremultiply();
    }
    work.convert_working_space(WorkingSpace::PerceptualSrgb);

    let mut cur = work;
    let mut passes = 0u8;
    while passes < 3 && (dst_w as f32 / cur.width as f32 > 2.05 || dst_h as f32 / cur.height as f32 > 2.05) {
        let next_w = (cur.width * 2).min(dst_w);
        let next_h = (cur.height * 2).min(dst_h);
        cur = easu_single_pass(&cur, next_w, next_h, pool)?;
        passes += 1;
    }

    if cur.width != dst_w || cur.height != dst_h {
        if dst_w >= cur.width && dst_h >= cur.height {
            cur = easu_single_pass(&cur, dst_w, dst_h, pool)?;
        } else {
            cur = resample_ewa_planar(&cur, dst_w, dst_h, ResampleFilter::EwaLanczosSharp, 0.5, pool)?;
        }
    }

    if rcas_sharpness > 0.001 {
        cur = rcas_pass(&cur, rcas_sharpness.clamp(0.0, 1.0), pool)?;
    }

    cur.convert_working_space(orig_space);
    if was_premul {
        cur.premultiply();
    }
    Ok(cur)
}

fn easu_single_pass(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    let sw = src.width as usize;
    let sh = src.height as usize;
    let dw = dst_w as usize;
    let scale_x = src.width as f32 / dst_w as f32;
    let scale_y = src.height as f32 / dst_h as f32;

    let mut dst = PlanarF32::new(dst_w, dst_h, src.has_alpha, src.working_space)?;
    dst.premultiplied = src.premultiplied;

    let sample_at = |x: i32, y: i32| -> (f32, f32, f32, f32) {
        let cx = x.clamp(0, sw as i32 - 1) as usize;
        let cy = y.clamp(0, sh as i32 - 1) as usize;
        let idx = cy * sw + cx;
        let a = if src.has_alpha { src.a[idx] } else { 1.0 };
        (src.r[idx], src.g[idx], src.b[idx], a)
    };

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
                let pp_y = (dy as f32 + 0.5) * scale_y - 0.5;
                let fp_y = pp_y.floor();
                let off_y = pp_y - fp_y;
                let iy = fp_y as i32;

                for dx in 0..dw {
                    let pp_x = (dx as f32 + 0.5) * scale_x - 0.5;
                    let fp_x = pp_x.floor();
                    let off_x = pp_x - fp_x;
                    let ix = fp_x as i32;

                    let b = sample_at(ix, iy - 1);
                    let c = sample_at(ix + 1, iy - 1);
                    let e = sample_at(ix - 1, iy);
                    let f = sample_at(ix, iy);
                    let g = sample_at(ix + 1, iy);
                    let h = sample_at(ix + 2, iy);
                    let i = sample_at(ix - 1, iy + 1);
                    let j = sample_at(ix, iy + 1);
                    let k = sample_at(ix + 1, iy + 1);
                    let l = sample_at(ix + 2, iy + 1);
                    let n = sample_at(ix, iy + 2);
                    let o = sample_at(ix + 1, iy + 2);

                    let lum = |px: (f32, f32, f32, f32)| 0.5 * px.1 + 0.25 * (px.0 + px.2);
                    let lb = lum(b);
                    let lc = lum(c);
                    let le = lum(e);
                    let lf = lum(f);
                    let lg = lum(g);
                    let lh = lum(h);
                    let li = lum(i);
                    let lj = lum(j);
                    let lk = lum(k);
                    let ll = lum(l);
                    let ln = lum(n);
                    let lo = lum(o);

                    let mut dir_x = 0.0f32;
                    let mut dir_y = 0.0f32;
                    let mut len_sum = 0.0f32;

                    let mut accum_set = |w: f32, top: f32, left: f32, center: f32, right: f32, bot: f32| {
                        let gx = right - left;
                        let gy = bot - top;
                        dir_x += gx * w;
                        dir_y += gy * w;
                        let max_x = (center - left).abs().max((right - center).abs());
                        let len_x = if max_x > 1e-5 {
                            ((right - left).abs() / max_x).min(1.0)
                        } else {
                            0.0
                        };
                        let max_y = (center - top).abs().max((bot - center).abs());
                        let len_y = if max_y > 1e-5 {
                            ((bot - top).abs() / max_y).min(1.0)
                        } else {
                            0.0
                        };
                        let l = (len_x * len_x + len_y * len_y) * 0.5;
                        len_sum += l * l * w;
                    };

                    accum_set((1.0 - off_x) * (1.0 - off_y), lb, le, lf, lg, lj);
                    accum_set(off_x * (1.0 - off_y), lc, lf, lg, lh, lk);
                    accum_set((1.0 - off_x) * off_y, lf, li, lj, lk, ln);
                    accum_set(off_x * off_y, lg, lj, lk, ll, lo);

                    let d2 = dir_x * dir_x + dir_y * dir_y;
                    let (nx, ny) = if d2 > 1e-8 {
                        let inv = 1.0 / d2.sqrt();
                        (dir_x * inv, dir_y * inv)
                    } else {
                        (1.0, 0.0)
                    };

                    let len = len_sum * 0.5;
                    let len2 = len * len;
                    let stretch = if nx.abs().max(ny.abs()) > 1e-5 {
                        (nx * nx + ny * ny) / nx.abs().max(ny.abs())
                    } else {
                        1.0
                    };
                    let sx = 1.0 + (stretch - 1.0) * len2;
                    let sy = 1.0 - 0.5 * len2;
                    let lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len2;
                    let clp = 1.0 / lob;

                    let mut a_r = 0.0f32;
                    let mut a_g = 0.0f32;
                    let mut a_b = 0.0f32;
                    let mut a_a = 0.0f32;
                    let mut a_w = 0.0f32;

                    let mut tap = |px: (f32, f32, f32, f32), tx: f32, ty: f32| {
                        let vx = tx - off_x;
                        let vy = ty - off_y;
                        let rx = (vx * nx + vy * ny) * sx;
                        let ry = (vx * (-ny) + vy * nx) * sy;
                        let dist2 = (rx * rx + ry * ry).min(clp);
                        let wb = (2.0 / 5.0) * dist2 - 1.0;
                        let wa = lob * dist2 - 1.0;
                        let w = ((25.0 / 16.0) * wb * wb - (25.0 / 16.0 - 1.0)) * (wa * wa);
                        a_r += px.0 * w;
                        a_g += px.1 * w;
                        a_b += px.2 * w;
                        a_a += px.3 * w;
                        a_w += w;
                    };

                    tap(b, 0.0, -1.0);
                    tap(c, 1.0, -1.0);
                    tap(e, -1.0, 0.0);
                    tap(f, 0.0, 0.0);
                    tap(g, 1.0, 0.0);
                    tap(h, 2.0, 0.0);
                    tap(i, -1.0, 1.0);
                    tap(j, 0.0, 1.0);
                    tap(k, 1.0, 1.0);
                    tap(l, 2.0, 1.0);
                    tap(n, 0.0, 2.0);
                    tap(o, 1.0, 2.0);

                    let inv_w = if a_w.abs() > 1e-8 { 1.0 / a_w } else { 1.0 };
                    let min_r = f.0.min(g.0).min(j.0).min(k.0);
                    let max_r = f.0.max(g.0).max(j.0).max(k.0);
                    let min_g = f.1.min(g.1).min(j.1).min(k.1);
                    let max_g = f.1.max(g.1).max(j.1).max(k.1);
                    let min_b = f.2.min(g.2).min(j.2).min(k.2);
                    let max_b = f.2.max(g.2).max(j.2).max(k.2);
                    let min_a = f.3.min(g.3).min(j.3).min(k.3);
                    let max_a = f.3.max(g.3).max(j.3).max(k.3);

                    let out_idx = local_y * dw + dx;
                    sr[out_idx] = (a_r * inv_w).clamp(min_r, max_r);
                    sg[out_idx] = (a_g * inv_w).clamp(min_g, max_g);
                    sb[out_idx] = (a_b * inv_w).clamp(min_b, max_b);
                    if let Some(ref mut sa_buf) = sa {
                        sa_buf[out_idx] = (a_a * inv_w).clamp(min_a, max_a);
                    }
                }
            }
        },
    );

    Ok(dst)
}

pub fn rcas_pass(src: &PlanarF32, sharpness: f32, pool: &ThreadPool) -> EngineResult<PlanarF32> {
    let w = src.width as usize;
    let h = src.height as usize;
    let mut dst = PlanarF32::new(src.width, src.height, src.has_alpha, src.working_space)?;
    dst.premultiplied = src.premultiplied;
    if src.has_alpha {
        dst.a.copy_from_slice(&src.a);
    }

    const RCAS_LIMIT: f32 = 0.25 - (1.0 / 16.0);
    let sharp = sharpness.clamp(0.0, 1.0);

    pool.parallel_row_stripes_4_mut(
        &mut dst.r,
        &mut dst.g,
        &mut dst.b,
        &mut Vec::new(),
        w,
        |y0, sr, sg, sb, _| {
            let rows = sr.len() / w;
            for local_y in 0..rows {
                let y = y0 + local_y;
                let ym = y.saturating_sub(1);
                let yp = (y + 1).min(h - 1);
                for x in 0..w {
                    let xm = x.saturating_sub(1);
                    let xp = (x + 1).min(w - 1);

                    let ic = y * w + x;
                    let ib = ym * w + x;
                    let id = y * w + xm;
                    let iff = y * w + xp;
                    let ih = yp * w + x;

                    let lum = |idx: usize| 0.5 * src.g[idx] + 0.25 * (src.r[idx] + src.b[idx]);
                    let lb = lum(ib);
                    let ld = lum(id);
                    let lc = lum(ic);
                    let lf = lum(iff);
                    let lh = lum(ih);

                    let min4 = lb.min(ld).min(lf).min(lh);
                    let max4 = lb.max(ld).max(lf).max(lh);
                    let hit_min = min4 / (4.0 * max4 + 1e-6);
                    let hit_max = (1.0 - max4) / (4.0 * min4 - 4.0 - 1e-6);
                    let lobe = hit_min.max(hit_max).clamp(-RCAS_LIMIT, 0.0);
                    let mut weight = (lobe * sharp).clamp(-RCAS_LIMIT, 0.0);

                    let nz = 0.25 * (lb + ld + lf + lh) - lc;
                    let max_n = max4.max(lc);
                    let min_n = min4.min(lc);
                    let range = (max_n - min_n).max(1e-6);
                    let damp = (1.0 - (nz.abs() / range)).clamp(0.0, 1.0);
                    weight *= damp * 0.75 + 0.25;

                    let rcp = 1.0 / (4.0 * weight + 1.0);
                    let out_i = local_y * w + x;
                    sr[out_i] = ((weight * (src.r[ib] + src.r[id] + src.r[iff] + src.r[ih]) + src.r[ic]) * rcp)
                        .clamp(0.0, 1.0);
                    sg[out_i] = ((weight * (src.g[ib] + src.g[id] + src.g[iff] + src.g[ih]) + src.g[ic]) * rcp)
                        .clamp(0.0, 1.0);
                    sb[out_i] = ((weight * (src.b[ib] + src.b[id] + src.b[iff] + src.b[ih]) + src.b[ic]) * rcp)
                        .clamp(0.0, 1.0);
                }
            }
        },
    );

    Ok(dst)
}
