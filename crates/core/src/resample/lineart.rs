use crate::color::WorkingSpace;
use crate::error::EngineResult;
use crate::pixel::PlanarF32;
use crate::pool::ThreadPool;
use crate::resample::easu::upscale_fsr_easu_rcas;

pub fn upscale_lineart(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    strength: f32,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    let orig_space = src.working_space;
    let mut work = src.clone();
    work.convert_working_space(WorkingSpace::PerceptualSrgb);

    let mut cur = upscale_fsr_easu_rcas(&work, dst_w, dst_h, 0.25, pool)?;
    let orig_highlights = capture_local_max(&cur);

    let s = strength.clamp(0.1, 2.0);
    cur = darken_lines(&cur, 0.14 * s, pool)?;
    cur = thin_lines(&cur, 0.16 * s, 2, pool)?;
    cur = deblur_dog(&cur, 0.45 * s, pool)?;
    clamp_highlights(&mut cur, &orig_highlights);

    cur.convert_working_space(orig_space);
    Ok(cur)
}

fn capture_local_max(img: &PlanarF32) -> Vec<[f32; 3]> {
    let w = img.width as usize;
    let h = img.height as usize;
    let mut out = vec![[0.0f32; 3]; w * h];
    for y in 0..h {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(h - 1);
        for x in 0..w {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(w - 1);
            let mut mr = 0.0f32;
            let mut mg = 0.0f32;
            let mut mb = 0.0f32;
            for ny in [ym, y, yp] {
                for nx in [xm, x, xp] {
                    let i = ny * w + nx;
                    mr = mr.max(img.r[i]);
                    mg = mg.max(img.g[i]);
                    mb = mb.max(img.b[i]);
                }
            }
            out[y * w + x] = [mr, mg, mb];
        }
    }
    out
}

fn clamp_highlights(img: &mut PlanarF32, orig_max: &[[f32; 3]]) {
    for (i, &[mr, mg, mb]) in orig_max.iter().enumerate() {
        img.r[i] = img.r[i].clamp(0.0, mr);
        img.g[i] = img.g[i].clamp(0.0, mg);
        img.b[i] = img.b[i].clamp(0.0, mb);
    }
}

fn darken_lines(src: &PlanarF32, strength: f32, pool: &ThreadPool) -> EngineResult<PlanarF32> {
    let w = src.width as usize;
    let h = src.height as usize;
    let mut dst = src.clone();
    let lum: Vec<f32> = src
        .r
        .iter()
        .zip(&src.g)
        .zip(&src.b)
        .map(|((&r, &g), &b)| 0.2126 * r + 0.7152 * g + 0.0722 * b)
        .collect();

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
                let ym = y.saturating_sub(1);
                let yp = (y + 1).min(h - 1);
                for x in 0..w {
                    let xm = x.saturating_sub(1);
                    let xp = (x + 1).min(w - 1);
                    let i = y * w + x;
                    let avg = 0.25 * (lum[ym * w + x] + lum[yp * w + x] + lum[y * w + xm] + lum[y * w + xp]);
                    let dip = (avg - lum[i]).max(0.0) * strength;
                    let oi = ly * w + x;
                    sr[oi] = (src.r[i] - dip).clamp(0.0, 1.0);
                    sg[oi] = (src.g[i] - dip).clamp(0.0, 1.0);
                    sb[oi] = (src.b[i] - dip).clamp(0.0, 1.0);
                }
            }
        },
    );
    Ok(dst)
}

fn thin_lines(src: &PlanarF32, strength: f32, iterations: u8, pool: &ThreadPool) -> EngineResult<PlanarF32> {
    let w = src.width as usize;
    let h = src.height as usize;
    let mut cur = src.clone();

    for _ in 0..iterations {
        let snap = cur.clone();
        let lum: Vec<f32> = snap
            .r
            .iter()
            .zip(&snap.g)
            .zip(&snap.b)
            .map(|((&r, &g), &b)| 0.2126 * r + 0.7152 * g + 0.0722 * b)
            .collect();

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
                    let ym = y.saturating_sub(1);
                    let yp = (y + 1).min(h - 1);
                    for x in 0..w {
                        let xm = x.saturating_sub(1);
                        let xp = (x + 1).min(w - 1);
                        let gx = (lum[ym * w + xp] + 2.0 * lum[y * w + xp] + lum[yp * w + xp])
                            - (lum[ym * w + xm] + 2.0 * lum[y * w + xm] + lum[yp * w + xm]);
                        let gy = (lum[yp * w + xm] + 2.0 * lum[yp * w + x] + lum[yp * w + xp])
                            - (lum[ym * w + xm] + 2.0 * lum[ym * w + x] + lum[ym * w + xp]);
                        let mag = (gx * gx + gy * gy).sqrt();
                        let i = y * w + x;
                        let oi = ly * w + x;
                        if mag > 0.05 {
                            let sx = if gx > 0.0 { xp } else if gx < 0.0 { xm } else { x };
                            let sy = if gy > 0.0 { yp } else if gy < 0.0 { ym } else { y };
                            let target = sy * w + sx;
                            if lum[target] > lum[i] {
                                sr[oi] = snap.r[i] + (snap.r[target] - snap.r[i]) * strength;
                                sg[oi] = snap.g[i] + (snap.g[target] - snap.g[i]) * strength;
                                sb[oi] = snap.b[i] + (snap.b[target] - snap.b[i]) * strength;
                            }
                        }
                    }
                }
            },
        );
    }

    Ok(cur)
}

fn deblur_dog(src: &PlanarF32, strength: f32, pool: &ThreadPool) -> EngineResult<PlanarF32> {
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
                let ym = y.saturating_sub(1);
                let yp = (y + 1).min(h - 1);
                for x in 0..w {
                    let xm = x.saturating_sub(1);
                    let xp = (x + 1).min(w - 1);
                    let i = y * w + x;
                    let oi = ly * w + x;

                    let apply_ch = |plane: &[f32]| -> f32 {
                        let c = plane[i];
                        let blur = (plane[ym * w + xm]
                            + 2.0 * plane[ym * w + x]
                            + plane[ym * w + xp]
                            + 2.0 * plane[y * w + xm]
                            + 4.0 * c
                            + 2.0 * plane[y * w + xp]
                            + plane[yp * w + xm]
                            + 2.0 * plane[yp * w + x]
                            + plane[yp * w + xp])
                            / 16.0;
                        let lo = c
                            .min(plane[ym * w + x])
                            .min(plane[yp * w + x])
                            .min(plane[y * w + xm])
                            .min(plane[y * w + xp]);
                        let hi = c
                            .max(plane[ym * w + x])
                            .max(plane[yp * w + x])
                            .max(plane[y * w + xm])
                            .max(plane[y * w + xp]);
                        let res = c - blur;
                        let gated = res.signum() * res.abs().powf(0.85);
                        (c + gated * strength).clamp(lo, hi)
                    };

                    sr[oi] = apply_ch(&src.r);
                    sg[oi] = apply_ch(&src.g);
                    sb[oi] = apply_ch(&src.b);
                }
            }
        },
    );
    Ok(dst)
}
