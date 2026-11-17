use crate::error::{EngineError, EngineResult};
use crate::pixel::PlanarF32;
use crate::pool::ThreadPool;
use crate::resample::ewa::resample_ewa_planar;
use crate::resample::kernels::ResampleFilter;

pub fn upscale_icbi(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    iterations: u8,
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

    let mut cur = src.clone();
    let mut passes = 0u8;
    while passes < 3 && (cur.width < dst_w || cur.height < dst_h) {
        cur = icbi_2x(&cur, iterations.clamp(1, 8), pool)?;
        passes += 1;
        if (cur.width as f32 / dst_w as f32) >= 0.95 && (cur.height as f32 / dst_h as f32) >= 0.95 {
            break;
        }
    }

    if cur.width != dst_w || cur.height != dst_h {
        cur = resample_ewa_planar(&cur, dst_w, dst_h, ResampleFilter::EwaLanczosSharp, 0.65, pool)?;
    }
    Ok(cur)
}

pub fn icbi_2x(src: &PlanarF32, iterations: u8, pool: &ThreadPool) -> EngineResult<PlanarF32> {
    let sw = src.width as usize;
    let sh = src.height as usize;
    let dw = sw * 2;
    let dh = sh * 2;
    let mut dst = PlanarF32::new(dw as u32, dh as u32, src.has_alpha, src.working_space)?;
    dst.premultiplied = src.premultiplied;

    for sy in 0..sh {
        for sx in 0..sw {
            let si = sy * sw + sx;
            let di = (sy * 2) * dw + (sx * 2);
            dst.r[di] = src.r[si];
            dst.g[di] = src.g[si];
            dst.b[di] = src.b[si];
            if src.has_alpha {
                dst.a[di] = src.a[si];
            }
        }
    }

    let lum_at = |d: &PlanarF32, x: isize, y: isize| -> f32 {
        let cx = x.clamp(0, dw as isize - 1) as usize;
        let cy = y.clamp(0, dh as isize - 1) as usize;
        let idx = cy * dw + cx;
        0.2126 * d.r[idx] + 0.7152 * d.g[idx] + 0.0722 * d.b[idx]
    };

    for sy in 0..sh {
        let y = (sy * 2 + 1) as isize;
        for sx in 0..sw {
            let x = (sx * 2 + 1) as isize;
            let p00 = (y - 1).clamp(0, dh as isize - 1) as usize * dw + (x - 1).clamp(0, dw as isize - 1) as usize;
            let p20 = (y - 1).clamp(0, dh as isize - 1) as usize * dw + (x + 1).clamp(0, dw as isize - 1) as usize;
            let p02 = (y + 1).clamp(0, dh as isize - 1) as usize * dw + (x - 1).clamp(0, dw as isize - 1) as usize;
            let p22 = (y + 1).clamp(0, dh as isize - 1) as usize * dw + (x + 1).clamp(0, dw as isize - 1) as usize;

            let d_main = (lum_at(&dst, x - 3, y - 3) + lum_at(&dst, x + 3, y + 3)
                - (lum_at(&dst, x - 1, y - 1) + lum_at(&dst, x + 1, y + 1)))
            .abs()
                + (lum_at(&dst, x - 1, y - 1) - lum_at(&dst, x + 1, y + 1)).abs();
            let d_anti = (lum_at(&dst, x + 3, y - 3) + lum_at(&dst, x - 3, y + 3)
                - (lum_at(&dst, x + 1, y - 1) + lum_at(&dst, x - 1, y + 1)))
            .abs()
                + (lum_at(&dst, x + 1, y - 1) - lum_at(&dst, x - 1, y + 1)).abs();

            let di = y as usize * dw + x as usize;
            if (d_main - d_anti).abs() < 0.02 {
                dst.r[di] = 0.25 * (dst.r[p00] + dst.r[p20] + dst.r[p02] + dst.r[p22]);
                dst.g[di] = 0.25 * (dst.g[p00] + dst.g[p20] + dst.g[p02] + dst.g[p22]);
                dst.b[di] = 0.25 * (dst.b[p00] + dst.b[p20] + dst.b[p02] + dst.b[p22]);
                if src.has_alpha {
                    dst.a[di] = 0.25 * (dst.a[p00] + dst.a[p20] + dst.a[p02] + dst.a[p22]);
                }
            } else if d_main < d_anti {
                dst.r[di] = 0.5 * (dst.r[p00] + dst.r[p22]);
                dst.g[di] = 0.5 * (dst.g[p00] + dst.g[p22]);
                dst.b[di] = 0.5 * (dst.b[p00] + dst.b[p22]);
                if src.has_alpha {
                    dst.a[di] = 0.5 * (dst.a[p00] + dst.a[p22]);
                }
            } else {
                dst.r[di] = 0.5 * (dst.r[p20] + dst.r[p02]);
                dst.g[di] = 0.5 * (dst.g[p20] + dst.g[p02]);
                dst.b[di] = 0.5 * (dst.b[p20] + dst.b[p02]);
                if src.has_alpha {
                    dst.a[di] = 0.5 * (dst.a[p20] + dst.a[p02]);
                }
            }
        }
    }

    for _ in 0..iterations {
        refine_icbi_phase(&mut dst, true, pool);
    }

    for y in 0..dh {
        for x in 0..dw {
            if (x & 1) == (y & 1) {
                continue;
            }
            let xi = x as isize;
            let yi = y as isize;
            let xm = (xi - 1).clamp(0, dw as isize - 1) as usize;
            let xp = (xi + 1).clamp(0, dw as isize - 1) as usize;
            let ym = (yi - 1).clamp(0, dh as isize - 1) as usize;
            let yp = (yi + 1).clamp(0, dh as isize - 1) as usize;

            let il = y * dw + xm;
            let ir = y * dw + xp;
            let it = ym * dw + x;
            let ib = yp * dw + x;

            let dh_curv = (lum_at(&dst, xi - 1, yi) - lum_at(&dst, xi + 1, yi)).abs();
            let dv_curv = (lum_at(&dst, xi, yi - 1) - lum_at(&dst, xi, yi + 1)).abs();

            let di = y * dw + x;
            if (dh_curv - dv_curv).abs() < 0.02 {
                dst.r[di] = 0.25 * (dst.r[il] + dst.r[ir] + dst.r[it] + dst.r[ib]);
                dst.g[di] = 0.25 * (dst.g[il] + dst.g[ir] + dst.g[it] + dst.g[ib]);
                dst.b[di] = 0.25 * (dst.b[il] + dst.b[ir] + dst.b[it] + dst.b[ib]);
                if src.has_alpha {
                    dst.a[di] = 0.25 * (dst.a[il] + dst.a[ir] + dst.a[it] + dst.a[ib]);
                }
            } else if dh_curv < dv_curv {
                dst.r[di] = 0.5 * (dst.r[il] + dst.r[ir]);
                dst.g[di] = 0.5 * (dst.g[il] + dst.g[ir]);
                dst.b[di] = 0.5 * (dst.b[il] + dst.b[ir]);
                if src.has_alpha {
                    dst.a[di] = 0.5 * (dst.a[il] + dst.a[ir]);
                }
            } else {
                dst.r[di] = 0.5 * (dst.r[it] + dst.r[ib]);
                dst.g[di] = 0.5 * (dst.g[it] + dst.g[ib]);
                dst.b[di] = 0.5 * (dst.b[it] + dst.b[ib]);
                if src.has_alpha {
                    dst.a[di] = 0.5 * (dst.a[it] + dst.a[ib]);
                }
            }
        }
    }

    for _ in 0..iterations {
        refine_icbi_phase(&mut dst, false, pool);
    }

    Ok(dst)
}

fn refine_icbi_phase(img: &mut PlanarF32, diag_phase: bool, pool: &ThreadPool) {
    let w = img.width as usize;
    let h = img.height as usize;
    let snap_r = img.r.clone();
    let snap_g = img.g.clone();
    let snap_b = img.b.clone();

    let lum = |x: isize, y: isize| -> f32 {
        let cx = x.clamp(0, w as isize - 1) as usize;
        let cy = y.clamp(0, h as isize - 1) as usize;
        let idx = cy * w + cx;
        0.2126 * snap_r[idx] + 0.7152 * snap_g[idx] + 0.0722 * snap_b[idx]
    };

    pool.parallel_row_stripes_4_mut(
        &mut img.r,
        &mut img.g,
        &mut img.b,
        &mut Vec::new(),
        w,
        |y0, sr, sg, sb, _| {
            let rows = sr.len() / w;
            for local_y in 0..rows {
                let y = y0 + local_y;
                for x in 0..w {
                    let is_target = if diag_phase {
                        (x & 1) == 1 && (y & 1) == 1
                    } else {
                        (x & 1) != (y & 1)
                    };
                    if !is_target {
                        continue;
                    }
                    let xi = x as isize;
                    let yi = y as isize;

                    let (n1, n2, n3, n4) = if diag_phase {
                        (
                            (xi - 1, yi - 1),
                            (xi + 1, yi - 1),
                            (xi - 1, yi + 1),
                            (xi + 1, yi + 1),
                        )
                    } else {
                        ((xi - 1, yi), (xi + 1, yi), (xi, yi - 1), (xi, yi + 1))
                    };

                    let sample = |(ax, ay): (isize, isize)| -> (f32, f32, f32) {
                        let cx = ax.clamp(0, w as isize - 1) as usize;
                        let cy = ay.clamp(0, h as isize - 1) as usize;
                        let i = cy * w + cx;
                        (snap_r[i], snap_g[i], snap_b[i])
                    };
                    let s1 = sample(n1);
                    let s2 = sample(n2);
                    let s3 = sample(n3);
                    let s4 = sample(n4);

                    let d1 = (lum(xi - 2, yi) + lum(xi + 2, yi) - 2.0 * lum(xi, yi)).abs();
                    let d2 = (lum(xi, yi - 2) + lum(xi, yi + 2) - 2.0 * lum(xi, yi)).abs();
                    let grad_h = (lum(xi + 2, yi) - lum(xi - 2, yi)).abs();
                    let grad_v = (lum(xi, yi + 2) - lum(xi, yi - 2)).abs();
                    let iso_bias = if grad_h > grad_v {
                        0.5 * (lum(xi, yi - 1) + lum(xi, yi + 1)) - lum(xi, yi)
                    } else {
                        0.5 * (lum(xi - 1, yi) + lum(xi + 1, yi)) - lum(xi, yi)
                    };
                    let step = (iso_bias * 0.22 - 0.05 * (d1 - d2)).clamp(-0.08, 0.08);

                    let oi = local_y * w + x;
                    let min_r = s1.0.min(s2.0).min(s3.0).min(s4.0);
                    let max_r = s1.0.max(s2.0).max(s3.0).max(s4.0);
                    let min_g = s1.1.min(s2.1).min(s3.1).min(s4.1);
                    let max_g = s1.1.max(s2.1).max(s3.1).max(s4.1);
                    let min_b = s1.2.min(s2.2).min(s3.2).min(s4.2);
                    let max_b = s1.2.max(s2.2).max(s3.2).max(s4.2);

                    sr[oi] = (sr[oi] + step).clamp(min_r, max_r);
                    sg[oi] = (sg[oi] + step).clamp(min_g, max_g);
                    sb[oi] = (sb[oi] + step).clamp(min_b, max_b);
                }
            }
        },
    );
}

pub fn upscale_nedi(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
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

    let mut cur = src.clone();
    let mut passes = 0u8;
    while passes < 3 && (cur.width < dst_w || cur.height < dst_h) {
        cur = nedi_2x(&cur, pool)?;
        passes += 1;
        if (cur.width as f32 / dst_w as f32) >= 0.95 && (cur.height as f32 / dst_h as f32) >= 0.95 {
            break;
        }
    }

    if cur.width != dst_w || cur.height != dst_h {
        cur = resample_ewa_planar(&cur, dst_w, dst_h, ResampleFilter::EwaLanczosSharp, 0.6, pool)?;
    }
    Ok(cur)
}

pub fn nedi_2x(src: &PlanarF32, pool: &ThreadPool) -> EngineResult<PlanarF32> {
    let sw = src.width as usize;
    let sh = src.height as usize;
    let dw = sw * 2;
    let dh = sh * 2;
    let mut dst = PlanarF32::new(dw as u32, dh as u32, src.has_alpha, src.working_space)?;
    dst.premultiplied = src.premultiplied;

    for sy in 0..sh {
        for sx in 0..sw {
            let si = sy * sw + sx;
            let di = (sy * 2) * dw + (sx * 2);
            dst.r[di] = src.r[si];
            dst.g[di] = src.g[si];
            dst.b[di] = src.b[si];
            if src.has_alpha {
                dst.a[di] = src.a[si];
            }
        }
    }

    let src_lum: Vec<f32> = src
        .r
        .iter()
        .zip(&src.g)
        .zip(&src.b)
        .map(|((&r, &g), &b)| 0.2126 * r + 0.7152 * g + 0.0722 * b)
        .collect();

    let lum_s = |sx: isize, sy: isize| -> f32 {
        let x = sx.clamp(0, sw as isize - 1) as usize;
        let y = sy.clamp(0, sh as isize - 1) as usize;
        src_lum[y * sw + x]
    };

    let get_s = |sx: isize, sy: isize| -> (f32, f32, f32, f32) {
        let x = sx.clamp(0, sw as isize - 1) as usize;
        let y = sy.clamp(0, sh as isize - 1) as usize;
        let idx = y * sw + x;
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
                if (dy & 1) == 0 {
                    continue;
                }
                let sy = (dy / 2) as isize;
                for sx in 0..sw {
                    let sxi = sx as isize;
                    let dx = sx * 2 + 1;
                    let oi = local_y * dw + dx;

                    let p0 = get_s(sxi, sy);
                    let p1 = get_s(sxi + 1, sy);
                    let p2 = get_s(sxi, sy + 1);
                    let p3 = get_s(sxi + 1, sy + 1);

                    let weights = solve_nedi_weights_diag(sxi, sy, &lum_s);
                    let (vr, vg, vb, va) = if let Some(w) = weights {
                        let min_r = p0.0.min(p1.0).min(p2.0).min(p3.0);
                        let max_r = p0.0.max(p1.0).max(p2.0).max(p3.0);
                        let min_g = p0.1.min(p1.1).min(p2.1).min(p3.1);
                        let max_g = p0.1.max(p1.1).max(p2.1).max(p3.1);
                        let min_b = p0.2.min(p1.2).min(p2.2).min(p3.2);
                        let max_b = p0.2.max(p1.2).max(p2.2).max(p3.2);
                        (
                            (w[0] * p0.0 + w[1] * p1.0 + w[2] * p2.0 + w[3] * p3.0).clamp(min_r, max_r),
                            (w[0] * p0.1 + w[1] * p1.1 + w[2] * p2.1 + w[3] * p3.1).clamp(min_g, max_g),
                            (w[0] * p0.2 + w[1] * p1.2 + w[2] * p2.2 + w[3] * p3.2).clamp(min_b, max_b),
                            (w[0] * p0.3 + w[1] * p1.3 + w[2] * p2.3 + w[3] * p3.3).clamp(0.0, 1.0),
                        )
                    } else {
                        (
                            0.25 * (p0.0 + p1.0 + p2.0 + p3.0),
                            0.25 * (p0.1 + p1.1 + p2.1 + p3.1),
                            0.25 * (p0.2 + p1.2 + p2.2 + p3.2),
                            0.25 * (p0.3 + p1.3 + p2.3 + p3.3),
                        )
                    };
                    sr[oi] = vr;
                    sg[oi] = vg;
                    sb[oi] = vb;
                    if let Some(ref mut sa_buf) = sa {
                        sa_buf[oi] = va;
                    }
                }
            }
        },
    );

    let snap_r = dst.r.clone();
    let snap_g = dst.g.clone();
    let snap_b = dst.b.clone();
    let snap_a = dst.a.clone();
    let lum_d = |x: isize, y: isize| -> f32 {
        let cx = x.clamp(0, dw as isize - 1) as usize;
        let cy = y.clamp(0, dh as isize - 1) as usize;
        let idx = cy * dw + cx;
        0.2126 * snap_r[idx] + 0.7152 * snap_g[idx] + 0.0722 * snap_b[idx]
    };
    let get_d = |x: isize, y: isize| -> (f32, f32, f32, f32) {
        let cx = x.clamp(0, dw as isize - 1) as usize;
        let cy = y.clamp(0, dh as isize - 1) as usize;
        let idx = cy * dw + cx;
        let a = if src.has_alpha { snap_a[idx] } else { 1.0 };
        (snap_r[idx], snap_g[idx], snap_b[idx], a)
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
                let y = y0 + local_y;
                let yi = y as isize;
                for x in 0..dw {
                    if (x & 1) == (y & 1) {
                        continue;
                    }
                    let xi = x as isize;
                    let p0 = get_d(xi - 1, yi);
                    let p1 = get_d(xi + 1, yi);
                    let p2 = get_d(xi, yi - 1);
                    let p3 = get_d(xi, yi + 1);

                    let weights = solve_nedi_weights_axial(xi, yi, &lum_d);
                    let oi = local_y * dw + x;
                    let (vr, vg, vb, va) = if let Some(w) = weights {
                        let min_r = p0.0.min(p1.0).min(p2.0).min(p3.0);
                        let max_r = p0.0.max(p1.0).max(p2.0).max(p3.0);
                        let min_g = p0.1.min(p1.1).min(p2.1).min(p3.1);
                        let max_g = p0.1.max(p1.1).max(p2.1).max(p3.1);
                        let min_b = p0.2.min(p1.2).min(p2.2).min(p3.2);
                        let max_b = p0.2.max(p1.2).max(p2.2).max(p3.2);
                        (
                            (w[0] * p0.0 + w[1] * p1.0 + w[2] * p2.0 + w[3] * p3.0).clamp(min_r, max_r),
                            (w[0] * p0.1 + w[1] * p1.1 + w[2] * p2.1 + w[3] * p3.1).clamp(min_g, max_g),
                            (w[0] * p0.2 + w[1] * p1.2 + w[2] * p2.2 + w[3] * p3.2).clamp(min_b, max_b),
                            (w[0] * p0.3 + w[1] * p1.3 + w[2] * p2.3 + w[3] * p3.3).clamp(0.0, 1.0),
                        )
                    } else {
                        (
                            0.25 * (p0.0 + p1.0 + p2.0 + p3.0),
                            0.25 * (p0.1 + p1.1 + p2.1 + p3.1),
                            0.25 * (p0.2 + p1.2 + p2.2 + p3.2),
                            0.25 * (p0.3 + p1.3 + p2.3 + p3.3),
                        )
                    };
                    sr[oi] = vr;
                    sg[oi] = vg;
                    sb[oi] = vb;
                    if let Some(ref mut sa_buf) = sa {
                        sa_buf[oi] = va;
                    }
                }
            }
        },
    );

    Ok(dst)
}

fn solve_nedi_weights_diag<F: Fn(isize, isize) -> f32>(
    sx: isize,
    sy: isize,
    lum: &F,
) -> Option<[f32; 4]> {
    let mut ata = [[0.0f64; 4]; 4];
    let mut aty = [0.0f64; 4];
    let mut sum_v = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut n = 0usize;

    for wy in -2..=2isize {
        for wx in -2..=2isize {
            let cx = sx + wx;
            let cy = sy + wy;
            let y = lum(cx, cy) as f64;
            let c = [
                lum(cx - 1, cy - 1) as f64,
                lum(cx + 1, cy - 1) as f64,
                lum(cx - 1, cy + 1) as f64,
                lum(cx + 1, cy + 1) as f64,
            ];
            for i in 0..4 {
                aty[i] += c[i] * y;
                for j in 0..4 {
                    ata[i][j] += c[i] * c[j];
                }
            }
            sum_v += y;
            sum_sq += y * y;
            n += 1;
        }
    }

    let mean = sum_v / n as f64;
    let var = sum_sq / n as f64 - mean * mean;
    if var < 2e-4 {
        return None;
    }
    solve_regularized_4x4(ata, aty)
}

fn solve_nedi_weights_axial<F: Fn(isize, isize) -> f32>(
    x: isize,
    y: isize,
    lum: &F,
) -> Option<[f32; 4]> {
    let mut ata = [[0.0f64; 4]; 4];
    let mut aty = [0.0f64; 4];
    let mut sum_v = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut n = 0usize;

    for wy in -2..=2isize {
        for wx in -2..=2isize {
            let cx = x + wx * 2;
            let cy = y + wy * 2;
            let target = lum(cx, cy) as f64;
            let c = [
                lum(cx - 2, cy) as f64,
                lum(cx + 2, cy) as f64,
                lum(cx, cy - 2) as f64,
                lum(cx, cy + 2) as f64,
            ];
            for i in 0..4 {
                aty[i] += c[i] * target;
                for j in 0..4 {
                    ata[i][j] += c[i] * c[j];
                }
            }
            sum_v += target;
            sum_sq += target * target;
            n += 1;
        }
    }

    let mean = sum_v / n as f64;
    let var = sum_sq / n as f64 - mean * mean;
    if var < 2e-4 {
        return None;
    }
    solve_regularized_4x4(ata, aty)
}

fn solve_regularized_4x4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f32; 4]> {
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += 1e-3;
    }
    for col in 0..4 {
        let mut pivot_row = col;
        let mut max_val = a[col][col].abs();
        for (r, row) in a.iter().enumerate().skip(col + 1) {
            if row[col].abs() > max_val {
                max_val = row[col].abs();
                pivot_row = r;
            }
        }
        if max_val < 1e-9 {
            return None;
        }
        if pivot_row != col {
            a.swap(col, pivot_row);
            b.swap(col, pivot_row);
        }
        let inv_diag = 1.0 / a[col][col];
        let (head, tail) = a.split_at_mut(col + 1);
        let pivot_slice = &head[col][col..4];
        for (r_idx, row_slice) in tail.iter_mut().enumerate() {
            let r = col + 1 + r_idx;
            let factor = row_slice[col] * inv_diag;
            for (dst, &src_v) in row_slice[col..4].iter_mut().zip(pivot_slice) {
                *dst -= factor * src_v;
            }
            b[r] -= factor * b[col];
        }
    }

    let mut x = [0.0f64; 4];
    for r in (0..4).rev() {
        let mut sum = b[r];
        for c in (r + 1)..4 {
            sum -= a[r][c] * x[c];
        }
        x[r] = sum / a[r][r];
    }

    let sum_w: f64 = x.iter().sum();
    let l1: f64 = x.iter().map(|v| v.abs()).sum();
    if !sum_w.is_finite() || sum_w.abs() < 0.5 || l1 > 2.5 {
        return None;
    }
    let inv = 1.0 / sum_w;
    Some([
        (x[0] * inv) as f32,
        (x[1] * inv) as f32,
        (x[2] * inv) as f32,
        (x[3] * inv) as f32,
    ])
}
