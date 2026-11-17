use crate::arena::PlanCache;
use crate::color::WorkingSpace;
use crate::error::{EngineError, EngineResult};
use crate::pixel::PlanarF32;
use crate::pool::ThreadPool;
use crate::resample::easu::upscale_fsr_easu_rcas;
use crate::resample::ewa::resample_ewa_planar;
use crate::resample::kernels::ResampleFilter;
use crate::resample::separable::resample_separable;

pub fn ssim_downscale(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    oversharp: f32,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    if dst_w == 0 || dst_h == 0 {
        return Err(EngineError::InvalidDimensions {
            width: dst_w,
            height: dst_h,
        });
    }
    if dst_w >= src.width && dst_h >= src.height {
        return resample_ewa_planar(src, dst_w, dst_h, ResampleFilter::EwaLanczosSharp, 0.5, pool);
    }

    let base_l = resample_separable(src, dst_w, dst_h, ResampleFilter::Mitchell, 0.25, plan_cache, pool)?;

    let sw = src.width as usize;
    let sh = src.height as usize;
    let src_sq = PlanarF32 {
        width: src.width,
        height: src.height,
        has_alpha: false,
        premultiplied: false,
        working_space: src.working_space,
        r: src.r.iter().map(|&v| v * v).collect(),
        g: src.g.iter().map(|&v| v * v).collect(),
        b: src.b.iter().map(|&v| v * v).collect(),
        a: Vec::new(),
    };
    let down_sq = resample_separable(&src_sq, dst_w, dst_h, ResampleFilter::Box, 0.0, plan_cache, pool)?;

    let dw = dst_w as usize;
    let dh = dst_h as usize;
    let mu_l = box_blur_3x3(&base_l, pool);
    let sq_l = PlanarF32 {
        width: dst_w,
        height: dst_h,
        has_alpha: false,
        premultiplied: false,
        working_space: base_l.working_space,
        r: base_l.r.iter().map(|&v| v * v).collect(),
        g: base_l.g.iter().map(|&v| v * v).collect(),
        b: base_l.b.iter().map(|&v| v * v).collect(),
        a: Vec::new(),
    };
    let mu_sq_l = box_blur_3x3(&sq_l, pool);

    let mut out = base_l.clone();
    let max_gain = (1.0 + oversharp.clamp(0.0, 2.0)).max(1.0);
    let _ = (sw, sh);

    for i in 0..dw * dh {
        let restore_ch = |l_val: f32, mu_val: f32, l_sq_avg: f32, h_sq_down: f32| -> f32 {
            let var_h = (h_sq_down - l_val * l_val).max(0.0);
            let var_l = (l_sq_avg - mu_val * mu_val).max(1e-5);
            let r = (var_h / var_l).sqrt().clamp(1.0, max_gain);
            let detail = l_val - mu_val;
            (l_val + (r - 1.0) * detail).clamp(0.0, 1.0)
        };
        out.r[i] = restore_ch(base_l.r[i], mu_l.r[i], mu_sq_l.r[i], down_sq.r[i]);
        out.g[i] = restore_ch(base_l.g[i], mu_l.g[i], mu_sq_l.g[i], down_sq.g[i]);
        out.b[i] = restore_ch(base_l.b[i], mu_l.b[i], mu_sq_l.b[i], down_sq.b[i]);
    }
    Ok(out)
}

pub fn ssim_super_res(
    src: &PlanarF32,
    upscaled: &PlanarF32,
    strength: f32,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    let dw = upscaled.width as usize;
    let dh = upscaled.height as usize;
    let down = resample_separable(
        upscaled,
        src.width,
        src.height,
        ResampleFilter::Mitchell,
        0.0,
        plan_cache,
        pool,
    )?;

    let mut err_src = PlanarF32::new(src.width, src.height, false, src.working_space)?;
    for i in 0..(src.width as usize * src.height as usize) {
        err_src.r[i] = src.r[i] - down.r[i];
        err_src.g[i] = src.g[i] - down.g[i];
        err_src.b[i] = src.b[i] - down.b[i];
    }
    let err_up = resample_separable(
        &err_src,
        upscaled.width,
        upscaled.height,
        ResampleFilter::CatmullRom,
        0.0,
        plan_cache,
        pool,
    )?;

    let s = strength.clamp(0.0, 2.0);
    let mut out = upscaled.clone();
    pool.parallel_row_stripes_4_mut(
        &mut out.r,
        &mut out.g,
        &mut out.b,
        &mut Vec::new(),
        dw,
        |y0, sr, sg, sb, _| {
            let rows = sr.len() / dw;
            for local_y in 0..rows {
                let y = y0 + local_y;
                let ym = y.saturating_sub(1);
                let yp = (y + 1).min(dh - 1);
                for x in 0..dw {
                    let xm = x.saturating_sub(1);
                    let xp = (x + 1).min(dw - 1);
                    let idx = y * dw + x;
                    let oi = local_y * dw + x;

                    let clamp_chan = |plane: &[f32], delta: f32| -> f32 {
                        let c = plane[idx];
                        let n1 = plane[ym * dw + x];
                        let n2 = plane[yp * dw + x];
                        let n3 = plane[y * dw + xm];
                        let n4 = plane[y * dw + xp];
                        let lo = c.min(n1).min(n2).min(n3).min(n4);
                        let hi = c.max(n1).max(n2).max(n3).max(n4);
                        let bounded_delta = (delta * s).tanh() * 0.25;
                        (c + bounded_delta).clamp(lo, hi)
                    };

                    sr[oi] = clamp_chan(&upscaled.r, err_up.r[idx]);
                    sg[oi] = clamp_chan(&upscaled.g, err_up.g[idx]);
                    sb[oi] = clamp_chan(&upscaled.b, err_up.b[idx]);
                }
            }
        },
    );

    Ok(out)
}

pub fn irani_peleg_ibp(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    iters: u8,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    if dst_w <= src.width && dst_h <= src.height {
        return ssim_downscale(src, dst_w, dst_h, 0.5, plan_cache, pool);
    }

    let mut cur = upscale_fsr_easu_rcas(src, dst_w, dst_h, 0.35, pool)?;
    let dw = dst_w as usize;
    let dh = dst_h as usize;
    let sw = src.width as usize;
    let sh = src.height as usize;

    let mut activity = vec![1.0f32; sw * sh];
    for y in 0..sh {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(sh - 1);
        for x in 0..sw {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(sw - 1);
            let gx = (src.g[y * sw + xp] - src.g[y * sw + xm]).abs();
            let gy = (src.g[yp * sw + x] - src.g[ym * sw + x]).abs();
            let edge = (gx + gy) * 4.0;
            activity[y * sw + x] = (edge / (edge + 0.08)).clamp(0.15, 1.0);
        }
    }

    let count = iters.clamp(1, 6);
    for iter in 0..count {
        let sim = resample_separable(
            &cur,
            src.width,
            src.height,
            ResampleFilter::Lanczos3,
            0.0,
            plan_cache,
            pool,
        )?;
        let mut diff = PlanarF32::new(src.width, src.height, false, cur.working_space)?;
        for (i, &w) in activity.iter().enumerate().take(sw * sh) {
            diff.r[i] = (src.r[i] - sim.r[i]) * w;
            diff.g[i] = (src.g[i] - sim.g[i]) * w;
            diff.b[i] = (src.b[i] - sim.b[i]) * w;
        }
        let back = resample_separable(
            &diff,
            dst_w,
            dst_h,
            ResampleFilter::CatmullRom,
            0.0,
            plan_cache,
            pool,
        )?;

        let step = 0.65 / (1.0 + 0.25 * iter as f32);
        let snap_r = cur.r.clone();
        let snap_g = cur.g.clone();
        let snap_b = cur.b.clone();
        pool.parallel_row_stripes_4_mut(
            &mut cur.r,
            &mut cur.g,
            &mut cur.b,
            &mut Vec::new(),
            dw,
            |y0, sr, sg, sb, _| {
                let rows = sr.len() / dw;
                for local_y in 0..rows {
                    let y = y0 + local_y;
                    let ym = y.saturating_sub(1);
                    let yp = (y + 1).min(dh - 1);
                    for x in 0..dw {
                        let xm = x.saturating_sub(1);
                        let xp = (x + 1).min(dw - 1);
                        let i = y * dw + x;
                        let oi = local_y * dw + x;

                        let bound = |p: &[f32], delta: f32| -> f32 {
                            let c = p[i];
                            let lo = c.min(p[ym * dw + x]).min(p[yp * dw + x]).min(p[y * dw + xm]).min(p[y * dw + xp]);
                            let hi = c.max(p[ym * dw + x]).max(p[yp * dw + x]).max(p[y * dw + xm]).max(p[y * dw + xp]);
                            (c + delta * step).clamp(lo, hi)
                        };
                        sr[oi] = bound(&snap_r, back.r[i]);
                        sg[oi] = bound(&snap_g, back.g[i]);
                        sb[oi] = bound(&snap_b, back.b[i]);
                    }
                }
            },
        );
    }

    Ok(cur)
}

pub fn krig_bilateral_chroma(
    img: &PlanarF32,
    sigma_spatial: f32,
    sigma_luma: f32,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    let orig_space = img.working_space;
    let mut work = img.clone();
    work.convert_working_space(WorkingSpace::PerceptualSrgb);

    let w = work.width as usize;
    let h = work.height as usize;
    let mut y_plane = vec![0.0f32; w * h];
    let mut cb_plane = vec![0.0f32; w * h];
    let mut cr_plane = vec![0.0f32; w * h];

    for i in 0..w * h {
        let r = work.r[i];
        let g = work.g[i];
        let b = work.b[i];
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        y_plane[i] = y;
        cb_plane[i] = (b - y) / 1.8556;
        cr_plane[i] = (r - y) / 1.5748;
    }

    let mut out_cb = cb_plane.clone();
    let mut out_cr = cr_plane.clone();
    let inv_2ss = 1.0 / (2.0 * sigma_spatial.max(0.2) * sigma_spatial.max(0.2));
    let inv_2sl = 1.0 / (2.0 * sigma_luma.max(0.01) * sigma_luma.max(0.01));

    pool.parallel_row_stripes_mut(&mut out_cb, w, |y0, stripe| {
        let rows = stripe.len() / w;
        for ly in 0..rows {
            let y = y0 + ly;
            for x in 0..w {
                let yc = y_plane[y * w + x];
                let mut acc_cb = 0.0f32;
                let mut acc_w = 0.0f32;
                for ky in -1..=1isize {
                    let sy = (y as isize + ky).clamp(0, h as isize - 1) as usize;
                    for kx in -1..=1isize {
                        let sx = (x as isize + kx).clamp(0, w as isize - 1) as usize;
                        let sidx = sy * w + sx;
                        let ds = (kx * kx + ky * ky) as f32;
                        let dl = y_plane[sidx] - yc;
                        let wgt = (-ds * inv_2ss - dl * dl * inv_2sl).exp();
                        acc_cb += cb_plane[sidx] * wgt;
                        acc_w += wgt;
                    }
                }
                stripe[ly * w + x] = if acc_w > 1e-8 { acc_cb / acc_w } else { cb_plane[y * w + x] };
            }
        }
    });

    pool.parallel_row_stripes_mut(&mut out_cr, w, |y0, stripe| {
        let rows = stripe.len() / w;
        for ly in 0..rows {
            let y = y0 + ly;
            for x in 0..w {
                let yc = y_plane[y * w + x];
                let mut acc_cr = 0.0f32;
                let mut acc_w = 0.0f32;
                for ky in -1..=1isize {
                    let sy = (y as isize + ky).clamp(0, h as isize - 1) as usize;
                    for kx in -1..=1isize {
                        let sx = (x as isize + kx).clamp(0, w as isize - 1) as usize;
                        let sidx = sy * w + sx;
                        let ds = (kx * kx + ky * ky) as f32;
                        let dl = y_plane[sidx] - yc;
                        let wgt = (-ds * inv_2ss - dl * dl * inv_2sl).exp();
                        acc_cr += cr_plane[sidx] * wgt;
                        acc_w += wgt;
                    }
                }
                stripe[ly * w + x] = if acc_w > 1e-8 { acc_cr / acc_w } else { cr_plane[y * w + x] };
            }
        }
    });

    for i in 0..w * h {
        let y = y_plane[i];
        let cb = out_cb[i];
        let cr = out_cr[i];
        let r = y + 1.5748 * cr;
        let b = y + 1.8556 * cb;
        let g = (y - 0.2126 * r - 0.0722 * b) / 0.7152;
        work.r[i] = r.clamp(0.0, 1.0);
        work.g[i] = g.clamp(0.0, 1.0);
        work.b[i] = b.clamp(0.0, 1.0);
    }

    work.convert_working_space(orig_space);
    Ok(work)
}

fn box_blur_3x3(src: &PlanarF32, pool: &ThreadPool) -> PlanarF32 {
    let w = src.width as usize;
    let h = src.height as usize;
    let mut dst = src.clone();
    pool.parallel_row_stripes_4_mut(
        &mut dst.r,
        &mut dst.g,
        &mut dst.b,
        &mut Vec::new(),
        w,
        |y0, sr, sg, sb, _| {
            let rows = sr.len() / w;
            for ly in 0..rows {
                let y = y0 + ly;
                for x in 0..w {
                    let mut ar = 0.0f32;
                    let mut ag = 0.0f32;
                    let mut ab = 0.0f32;
                    for ky in -1..=1isize {
                        let sy = (y as isize + ky).clamp(0, h as isize - 1) as usize;
                        for kx in -1..=1isize {
                            let sx = (x as isize + kx).clamp(0, w as isize - 1) as usize;
                            let idx = sy * w + sx;
                            ar += src.r[idx];
                            ag += src.g[idx];
                            ab += src.b[idx];
                        }
                    }
                    let oi = ly * w + x;
                    sr[oi] = ar / 9.0;
                    sg[oi] = ag / 9.0;
                    sb[oi] = ab / 9.0;
                }
            }
        },
    );
    dst
}
