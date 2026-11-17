use crate::error::{EngineError, EngineResult};
use crate::pixel::PlanarF32;
use crate::pool::ThreadPool;
use crate::resample::ewa::resample_ewa_planar;
use crate::resample::kernels::ResampleFilter;

#[derive(Clone, Copy, PartialEq)]
struct RgbaPx {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl RgbaPx {
    #[inline]
    fn from_planar(p: &PlanarF32, idx: usize) -> Self {
        Self {
            r: p.r[idx],
            g: p.g[idx],
            b: p.b[idx],
            a: if p.has_alpha { p.a[idx] } else { 1.0 },
        }
    }

    #[inline]
    fn blend(self, other: Self, weight_other: f32) -> Self {
        let t = weight_other.clamp(0.0, 1.0);
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }

    #[inline]
    fn luma(self) -> f32 {
        (0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b) * self.a
    }

    #[inline]
    fn to_perceptual(self) -> Self {
        Self {
            r: self.r.max(0.0).powf(1.0 / 2.2),
            g: self.g.max(0.0).powf(1.0 / 2.2),
            b: self.b.max(0.0).powf(1.0 / 2.2),
            a: self.a.clamp(0.0, 1.0),
        }
    }

    #[inline]
    fn dist_compuphase(self, other: Self) -> f32 {
        let s = self.to_perceptual();
        let o = other.to_perceptual();
        let rmean = 0.5 * (s.r + o.r);
        let dr = s.r - o.r;
        let dg = s.g - o.g;
        let db = s.b - o.b;
        let da = (s.a - o.a) * 1.5;
        (((2.0 + rmean) * dr * dr + 4.0 * dg * dg + (3.0 - rmean) * db * db) + da * da).sqrt()
    }

    #[inline]
    fn eq_exact(self, other: Self) -> bool {
        (self.r - other.r).abs() < 1e-4
            && (self.g - other.g).abs() < 1e-4
            && (self.b - other.b).abs() < 1e-4
            && (self.a - other.a).abs() < 1e-4
    }
}

pub fn nearest_resize(src: &PlanarF32, dst_w: u32, dst_h: u32, pool: &ThreadPool) -> EngineResult<PlanarF32> {
    if dst_w == 0 || dst_h == 0 {
        return Err(EngineError::InvalidDimensions {
            width: dst_w,
            height: dst_h,
        });
    }
    let sw = src.width as usize;
    let sh = src.height as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;
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
            for ly in 0..rows {
                let dy = y0 + ly;
                let sy = ((dy * sh) / dh).min(sh - 1);
                for dx in 0..dw {
                    let sx = ((dx * sw) / dw).min(sw - 1);
                    let si = sy * sw + sx;
                    let oi = ly * dw + dx;
                    sr[oi] = src.r[si];
                    sg[oi] = src.g[si];
                    sb[oi] = src.b[si];
                    if let Some(ref mut sa_buf) = sa {
                        sa_buf[oi] = src.a[si];
                    }
                }
            }
        },
    );
    Ok(dst)
}

pub fn upscale_mmpx(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    if dst_w <= src.width && dst_h <= src.height {
        return nearest_resize(src, dst_w, dst_h, pool);
    }
    let mut cur = src.clone();
    let mut passes = 0u8;
    while passes < 3 && (cur.width < dst_w || cur.height < dst_h) {
        cur = mmpx_2x(&cur, pool)?;
        passes += 1;
    }
    if cur.width != dst_w || cur.height != dst_h {
        if dst_w.is_multiple_of(cur.width) && dst_h.is_multiple_of(cur.height) {
            return nearest_resize(&cur, dst_w, dst_h, pool);
        }
        return resample_ewa_planar(&cur, dst_w, dst_h, ResampleFilter::EwaRobidouxSharp, 0.7, pool);
    }
    Ok(cur)
}

pub fn mmpx_2x(src: &PlanarF32, pool: &ThreadPool) -> EngineResult<PlanarF32> {
    let sw = src.width as usize;
    let sh = src.height as usize;
    let dw = sw * 2;
    let mut dst = PlanarF32::new(dw as u32, (sh * 2) as u32, src.has_alpha, src.working_space)?;
    dst.premultiplied = src.premultiplied;

    let sample = |x: isize, y: isize| -> RgbaPx {
        let cx = x.clamp(0, sw as isize - 1) as usize;
        let cy = y.clamp(0, sh as isize - 1) as usize;
        RgbaPx::from_planar(src, cy * sw + cx)
    };

    pool.parallel_row_stripes_4_mut(
        &mut dst.r,
        &mut dst.g,
        &mut dst.b,
        &mut dst.a,
        dw,
        |y0, sr, sg, sb, mut sa| {
            let rows = sr.len() / dw;
            for ly in 0..rows {
                let dy = y0 + ly;
                let sy = (dy / 2) as isize;
                let sub_y = dy & 1;

                for sx in 0..sw {
                    let sxi = sx as isize;
                    let e = sample(sxi, sy);
                    let a = sample(sxi - 1, sy - 1);
                    let b = sample(sxi, sy - 1);
                    let c = sample(sxi + 1, sy - 1);
                    let d = sample(sxi - 1, sy);
                    let f = sample(sxi + 1, sy);
                    let g = sample(sxi - 1, sy + 1);
                    let h = sample(sxi, sy + 1);
                    let i = sample(sxi + 1, sy + 1);

                    let mut j = e;
                    let mut k = e;
                    let mut l = e;
                    let mut m = e;

                    if !a.eq_exact(e)
                        || !b.eq_exact(e)
                        || !c.eq_exact(e)
                        || !d.eq_exact(e)
                        || !f.eq_exact(e)
                        || !g.eq_exact(e)
                        || !h.eq_exact(e)
                        || !i.eq_exact(e)
                    {
                        let p = sample(sxi, sy - 2);
                        let q = sample(sxi - 2, sy);
                        let r = sample(sxi + 2, sy);
                        let s = sample(sxi, sy + 2);

                        let bl = b.luma();
                        let dl = d.luma();
                        let el = e.luma();
                        let fl = f.luma();
                        let hl = h.luma();

                        if d.eq_exact(b) && !d.eq_exact(h) && !b.eq_exact(f) {
                            let same = (!a.eq_exact(e) || !c.eq_exact(e) || !g.eq_exact(e))
                                && (el >= dl || e.eq_exact(a) || e.eq_exact(c) || e.eq_exact(g));
                            if (!a.eq_exact(e) && !d.eq_exact(q)) || (!e.eq_exact(c) && !b.eq_exact(p)) || same {
                                j = b.blend(e, 0.15);
                            }
                            if el > bl && (c.eq_exact(e) || b.eq_exact(f)) {
                                j = b;
                            }
                        }
                        if b.eq_exact(f) && !b.eq_exact(d) && !f.eq_exact(h) {
                            let same = (!c.eq_exact(e) || !a.eq_exact(e) || !i.eq_exact(e))
                                && (el >= fl || e.eq_exact(c) || e.eq_exact(a) || e.eq_exact(i));
                            if (!c.eq_exact(e) && !f.eq_exact(r)) || (!e.eq_exact(a) && !b.eq_exact(p)) || same {
                                k = f.blend(e, 0.15);
                            }
                            if el > fl && (i.eq_exact(e) || f.eq_exact(h)) {
                                k = f;
                            }
                        }
                        if h.eq_exact(d) && !h.eq_exact(f) && !d.eq_exact(b) {
                            let same = (!g.eq_exact(e) || !a.eq_exact(e) || !i.eq_exact(e))
                                && (el >= dl || e.eq_exact(g) || e.eq_exact(a) || e.eq_exact(i));
                            if (!g.eq_exact(e) && !d.eq_exact(q)) || (!e.eq_exact(i) && !h.eq_exact(s)) || same {
                                l = d.blend(e, 0.15);
                            }
                            if el > hl && (a.eq_exact(e) || d.eq_exact(b)) {
                                l = d;
                            }
                        }
                        if f.eq_exact(h) && !f.eq_exact(b) && !h.eq_exact(d) {
                            let same = (!i.eq_exact(e) || !c.eq_exact(e) || !g.eq_exact(e))
                                && (el >= hl || e.eq_exact(i) || e.eq_exact(c) || e.eq_exact(g));
                            if (!i.eq_exact(e) && !f.eq_exact(r)) || (!e.eq_exact(g) && !h.eq_exact(s)) || same {
                                m = h.blend(e, 0.15);
                            }
                            if el > fl && (c.eq_exact(e) || h.eq_exact(d)) {
                                m = h;
                            }
                        }
                    }

                    let (left_px, right_px) = if sub_y == 0 { (j, k) } else { (l, m) };
                    let oi = ly * dw + sx * 2;
                    sr[oi] = left_px.r;
                    sg[oi] = left_px.g;
                    sb[oi] = left_px.b;
                    sr[oi + 1] = right_px.r;
                    sg[oi + 1] = right_px.g;
                    sb[oi + 1] = right_px.b;
                    if let Some(ref mut sa_buf) = sa {
                        sa_buf[oi] = left_px.a;
                        sa_buf[oi + 1] = right_px.a;
                    }
                }
            }
        },
    );

    Ok(dst)
}

pub fn upscale_xbrz_or_scalefx(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    scalefx_mode: bool,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    if dst_w <= src.width && dst_h <= src.height {
        return nearest_resize(src, dst_w, dst_h, pool);
    }
    let ratio = (dst_w as f32 / src.width as f32)
        .max(dst_h as f32 / src.height as f32)
        .ceil()
        .clamp(2.0, 6.0) as usize;
    let scaled = xbr_style_nx(src, ratio, scalefx_mode, pool)?;
    if scaled.width != dst_w || scaled.height != dst_h {
        if dst_w.is_multiple_of(scaled.width) && dst_h.is_multiple_of(scaled.height) {
            return nearest_resize(&scaled, dst_w, dst_h, pool);
        }
        return resample_ewa_planar(&scaled, dst_w, dst_h, ResampleFilter::EwaRobidouxSharp, 0.6, pool);
    }
    Ok(scaled)
}

fn xbr_style_nx(
    src: &PlanarF32,
    factor: usize,
    preserve_palette: bool,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    let sw = src.width as usize;
    let sh = src.height as usize;
    let dw = sw * factor;
    let dh = sh * factor;
    let mut dst = PlanarF32::new(dw as u32, dh as u32, src.has_alpha, src.working_space)?;
    dst.premultiplied = src.premultiplied;

    let sample = |x: isize, y: isize| -> RgbaPx {
        let cx = x.clamp(0, sw as isize - 1) as usize;
        let cy = y.clamp(0, sh as isize - 1) as usize;
        RgbaPx::from_planar(src, cy * sw + cx)
    };

    pool.parallel_row_stripes_4_mut(
        &mut dst.r,
        &mut dst.g,
        &mut dst.b,
        &mut dst.a,
        dw,
        |y0, sr, sg, sb, mut sa| {
            let rows = sr.len() / dw;
            for ly in 0..rows {
                let dy = y0 + ly;
                let sy = (dy / factor) as isize;
                let fy = ((dy % factor) as f32 + 0.5) / factor as f32 - 0.5;

                for sx in 0..sw {
                    let sxi = sx as isize;
                    let pe = sample(sxi, sy);
                    let pb = sample(sxi, sy - 1);
                    let pd = sample(sxi - 1, sy);
                    let pf = sample(sxi + 1, sy);
                    let ph = sample(sxi, sy + 1);
                    let pa = sample(sxi - 1, sy - 1);
                    let pc = sample(sxi + 1, sy - 1);
                    let pg = sample(sxi - 1, sy + 1);
                    let pi = sample(sxi + 1, sy + 1);

                    let threshold = 0.22f32;
                    let min_diff = 0.08f32;
                    let w_nw = pb.dist_compuphase(pd) < threshold
                        && pe.dist_compuphase(pb) > min_diff
                        && pe.dist_compuphase(pd) > min_diff
                        && pb.dist_compuphase(pi) > min_diff
                        && (pa.dist_compuphase(pe) > min_diff
                            || (pb.dist_compuphase(pc) > min_diff && pd.dist_compuphase(pg) > min_diff));
                    let w_ne = pb.dist_compuphase(pf) < threshold
                        && pe.dist_compuphase(pb) > min_diff
                        && pe.dist_compuphase(pf) > min_diff
                        && pb.dist_compuphase(pg) > min_diff
                        && (pc.dist_compuphase(pe) > min_diff
                            || (pb.dist_compuphase(pa) > min_diff && pf.dist_compuphase(pi) > min_diff));
                    let w_sw = ph.dist_compuphase(pd) < threshold
                        && pe.dist_compuphase(ph) > min_diff
                        && pe.dist_compuphase(pd) > min_diff
                        && ph.dist_compuphase(pc) > min_diff
                        && (pg.dist_compuphase(pe) > min_diff
                            || (ph.dist_compuphase(pi) > min_diff && pd.dist_compuphase(pa) > min_diff));
                    let w_se = ph.dist_compuphase(pf) < threshold
                        && pe.dist_compuphase(ph) > min_diff
                        && pe.dist_compuphase(pf) > min_diff
                        && ph.dist_compuphase(pa) > min_diff
                        && (pi.dist_compuphase(pe) > min_diff
                            || (ph.dist_compuphase(pg) > min_diff && pf.dist_compuphase(pc) > min_diff));

                    for sub_x in 0..factor {
                        let fx = ((sub_x as f32 + 0.5) / factor as f32) - 0.5;
                        let mut px = pe;

                        let diag_tl = fx + fy;
                        let diag_tr = (-fx) + fy;
                        let diag_bl = fx - fy;
                        let diag_br = (-fx) - fy;
                        let cutoff = -0.50f32;
                        let slope = (factor as f32).max(2.0);

                        if w_nw && diag_tl < cutoff {
                            let blend_c = pb.blend(pd, 0.5);
                            px = if preserve_palette {
                                if pb.dist_compuphase(pe) <= pd.dist_compuphase(pe) { pb } else { pd }
                            } else {
                                pe.blend(blend_c, ((cutoff - diag_tl) * slope).clamp(0.0, 1.0))
                            };
                        } else if w_ne && diag_tr < cutoff {
                            let blend_c = pb.blend(pf, 0.5);
                            px = if preserve_palette {
                                if pb.dist_compuphase(pe) <= pf.dist_compuphase(pe) { pb } else { pf }
                            } else {
                                pe.blend(blend_c, ((cutoff - diag_tr) * slope).clamp(0.0, 1.0))
                            };
                        } else if w_sw && diag_bl < cutoff {
                            let blend_c = ph.blend(pd, 0.5);
                            px = if preserve_palette {
                                if ph.dist_compuphase(pe) <= pd.dist_compuphase(pe) { ph } else { pd }
                            } else {
                                pe.blend(blend_c, ((cutoff - diag_bl) * slope).clamp(0.0, 1.0))
                            };
                        } else if w_se && diag_br < cutoff {
                            let blend_c = ph.blend(pf, 0.5);
                            px = if preserve_palette {
                                if ph.dist_compuphase(pe) <= pf.dist_compuphase(pe) { ph } else { pf }
                            } else {
                                pe.blend(blend_c, ((cutoff - diag_br) * slope).clamp(0.0, 1.0))
                            };
                        }

                        let oi = ly * dw + sx * factor + sub_x;
                        sr[oi] = px.r;
                        sg[oi] = px.g;
                        sb[oi] = px.b;
                        if let Some(ref mut sa_buf) = sa {
                            sa_buf[oi] = px.a;
                        }
                    }
                }
            }
        },
    );

    Ok(dst)
}
