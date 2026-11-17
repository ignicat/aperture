use crate::error::{EngineError, EngineResult};
use crate::geometry::saliency::{SaliencyMap, compute_saliency_map};
use crate::pixel::PlanarF32;

#[derive(Debug, Clone)]
pub struct VisibilityMap {
    pub width: usize,
    pub height: usize,
    pub seam_order: Vec<u16>,
}

pub fn seam_carve_resize(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
) -> EngineResult<PlanarF32> {
    if dst_w == 0 || dst_h == 0 {
        return Err(EngineError::InvalidDimensions {
            width: dst_w,
            height: dst_h,
        });
    }
    if src.width == dst_w && src.height == dst_h {
        return Ok(src.clone());
    }

    let sal = compute_saliency_map(src);
    let mut cur = src.clone();

    while cur.width != dst_w {
        if dst_w < cur.width {
            let remove_count = (cur.width - dst_w) as usize;
            cur = carve_vertical_seams(&cur, remove_count, &sal)?;
        } else {
            let max_step = ((cur.width as f32 * 0.45).round() as u32).max(1);
            let step = (dst_w - cur.width).min(max_step) as usize;
            cur = insert_vertical_seams(&cur, step, &sal)?;
        }
    }

    if cur.height != dst_h {
        cur = transpose_planar(&cur)?;
        let sal_t = compute_saliency_map(&cur);
        while cur.width != dst_h {
            if dst_h < cur.width {
                let remove_count = (cur.width - dst_h) as usize;
                cur = carve_vertical_seams(&cur, remove_count, &sal_t)?;
            } else {
                let max_step = ((cur.width as f32 * 0.45).round() as u32).max(1);
                let step = (dst_h - cur.width).min(max_step) as usize;
                cur = insert_vertical_seams(&cur, step, &sal_t)?;
            }
        }
        cur = transpose_planar(&cur)?;
    }

    Ok(cur)
}

pub fn build_vertical_visibility_map(src: &PlanarF32, max_seams: usize) -> VisibilityMap {
    let w = src.width as usize;
    let h = src.height as usize;
    let mut seam_order = vec![u16::MAX; w * h];
    if w <= 1 || h == 0 || max_seams == 0 {
        return VisibilityMap {
            width: w,
            height: h,
            seam_order,
        };
    }

    let mut lum: Vec<f32> = src
        .r
        .iter()
        .zip(&src.g)
        .zip(&src.b)
        .map(|((&r, &g), &b)| 0.2126 * r + 0.7152 * g + 0.0722 * b)
        .collect();
    let mut orig_x: Vec<usize> = (0..h).flat_map(|_| 0..w).collect();
    let mut cur_w = w;
    let limit = max_seams.min(w - 1).min(u16::MAX as usize - 1);

    for step in 0..limit {
        let seam = find_forward_energy_seam(&lum, cur_w, h, None);
        for (y, &sx) in seam.iter().enumerate() {
            let ox = orig_x[y * cur_w + sx];
            seam_order[y * w + ox] = step as u16;
        }
        let next_w = cur_w - 1;
        let mut next_lum = Vec::with_capacity(next_w * h);
        let mut next_ox = Vec::with_capacity(next_w * h);
        for (y, &sx) in seam.iter().enumerate() {
            let off = y * cur_w;
            next_lum.extend_from_slice(&lum[off..off + sx]);
            next_lum.extend_from_slice(&lum[off + sx + 1..off + cur_w]);
            next_ox.extend_from_slice(&orig_x[off..off + sx]);
            next_ox.extend_from_slice(&orig_x[off + sx + 1..off + cur_w]);
        }
        lum = next_lum;
        orig_x = next_ox;
        cur_w = next_w;
    }

    VisibilityMap {
        width: w,
        height: h,
        seam_order,
    }
}

fn carve_vertical_seams(
    src: &PlanarF32,
    remove_count: usize,
    sal: &SaliencyMap,
) -> EngineResult<PlanarF32> {
    let w = src.width as usize;
    let h = src.height as usize;
    if remove_count >= w {
        return Err(EngineError::InvalidDimensions {
            width: 0,
            height: src.height,
        });
    }

    let proxy_limit = 420usize;
    if w > proxy_limit && h > proxy_limit {
        let scale = w.max(h) as f32 / proxy_limit as f32;
        let pw = ((w as f32 / scale).round() as usize).max(32);
        let ph = ((h as f32 / scale).round() as usize).max(32);
        let mut p_lum = downsample_luma(src, pw, ph);
        let mut p_cur_w = pw;
        let proxy_remove = ((remove_count as f32 / scale).round() as usize)
            .clamp(1, p_cur_w - 1);

        let mut work = src.clone();
        let ratio_per_seam = w as f32 / pw as f32;
        let mut removed_full = 0usize;

        for _ in 0..proxy_remove {
            if removed_full >= remove_count || work.width <= 1 {
                break;
            }
            let p_seam = find_forward_energy_seam(&p_lum, p_cur_w, ph, Some(sal));
            let batch_cols = ((ratio_per_seam.round() as usize).max(1)).min(remove_count - removed_full);
            for _ in 0..batch_cols {
                if work.width <= 1 {
                    break;
                }
                let full_seam = project_and_snap_seam(&p_seam, p_cur_w, ph, &work);
                work = remove_single_seam(&work, &full_seam)?;
                removed_full += 1;
            }
            let next_pw = p_cur_w - 1;
            let mut next_p = Vec::with_capacity(next_pw * ph);
            for (y, &sx) in p_seam.iter().enumerate() {
                let off = y * p_cur_w;
                next_p.extend_from_slice(&p_lum[off..off + sx]);
                next_p.extend_from_slice(&p_lum[off + sx + 1..off + p_cur_w]);
            }
            p_lum = next_p;
            p_cur_w = next_pw;
        }

        while removed_full < remove_count && work.width > 1 {
            let lum = extract_luma(&work);
            let seam = find_forward_energy_seam(&lum, work.width as usize, h, Some(sal));
            work = remove_single_seam(&work, &seam)?;
            removed_full += 1;
        }
        return Ok(work);
    }

    let mut cur = src.clone();
    for _ in 0..remove_count {
        let cw = cur.width as usize;
        if cw <= 1 {
            break;
        }
        let lum = extract_luma(&cur);
        let seam = find_forward_energy_seam(&lum, cw, h, Some(sal));
        cur = remove_single_seam(&cur, &seam)?;
    }
    Ok(cur)
}

fn insert_vertical_seams(
    src: &PlanarF32,
    insert_count: usize,
    sal: &SaliencyMap,
) -> EngineResult<PlanarF32> {
    let w = src.width as usize;
    let h = src.height as usize;
    let count = insert_count.min(w.saturating_sub(1)).max(1);
    let mut lum = extract_luma(src);
    let mut orig_x: Vec<usize> = (0..h).flat_map(|_| 0..w).collect();
    let mut cur_w = w;
    let mut seams_in_orig: Vec<Vec<usize>> = Vec::with_capacity(count);

    for _ in 0..count {
        if cur_w <= 1 {
            break;
        }
        let seam = find_forward_energy_seam(&lum, cur_w, h, Some(sal));
        let mut orig_seam = Vec::with_capacity(h);
        for (y, &sx) in seam.iter().enumerate() {
            orig_seam.push(orig_x[y * cur_w + sx]);
        }
        seams_in_orig.push(orig_seam);

        let next_w = cur_w - 1;
        let mut next_lum = Vec::with_capacity(next_w * h);
        let mut next_ox = Vec::with_capacity(next_w * h);
        for (y, &sx) in seam.iter().enumerate() {
            let off = y * cur_w;
            next_lum.extend_from_slice(&lum[off..off + sx]);
            next_lum.extend_from_slice(&lum[off + sx + 1..off + cur_w]);
            next_ox.extend_from_slice(&orig_x[off..off + sx]);
            next_ox.extend_from_slice(&orig_x[off + sx + 1..off + cur_w]);
        }
        lum = next_lum;
        orig_x = next_ox;
        cur_w = next_w;
    }

    let inserted = seams_in_orig.len();
    let new_w = w + inserted;
    let mut dst = PlanarF32::new(new_w as u32, src.height, src.has_alpha, src.working_space)?;
    dst.premultiplied = src.premultiplied;

    for y in 0..h {
        let mut row_cols: Vec<usize> = seams_in_orig.iter().map(|s| s[y]).collect();
        row_cols.sort_unstable();
        let s_off = y * w;
        let d_off = y * new_w;
        let mut seam_ptr = 0usize;
        let mut dx = 0usize;
        for sx in 0..w {
            let si = s_off + sx;
            dst.r[d_off + dx] = src.r[si];
            dst.g[d_off + dx] = src.g[si];
            dst.b[d_off + dx] = src.b[si];
            if src.has_alpha {
                dst.a[d_off + dx] = src.a[si];
            }
            dx += 1;
            while seam_ptr < row_cols.len() && row_cols[seam_ptr] == sx {
                let nr_i = s_off + (sx + 1).min(w - 1);
                dst.r[d_off + dx] = 0.5 * (src.r[si] + src.r[nr_i]);
                dst.g[d_off + dx] = 0.5 * (src.g[si] + src.g[nr_i]);
                dst.b[d_off + dx] = 0.5 * (src.b[si] + src.b[nr_i]);
                if src.has_alpha {
                    dst.a[d_off + dx] = 0.5 * (src.a[si] + src.a[nr_i]);
                }
                dx += 1;
                seam_ptr += 1;
            }
        }
    }

    Ok(dst)
}

fn find_forward_energy_seam(
    lum: &[f32],
    w: usize,
    h: usize,
    sal: Option<&SaliencyMap>,
) -> Vec<usize> {
    let mut prev_dp = vec![0.0f32; w];
    let mut curr_dp = vec![0.0f32; w];
    let mut backtrack = vec![0i8; w * h];

    let at = |x: usize, y: usize| -> f32 { lum[y * w + x] };
    let sal_bias = |x: usize, y: usize| -> f32 {
        match sal {
            Some(s) => s.sample_norm(x as f32 / w.max(1) as f32, y as f32 / h.max(1) as f32) * 0.45,
            None => 0.0,
        }
    };

    for (x, slot) in prev_dp.iter_mut().enumerate().take(w) {
        let xm = x.saturating_sub(1);
        let xp = (x + 1).min(w - 1);
        *slot = (at(xp, 0) - at(xm, 0)).abs() + sal_bias(x, 0);
    }

    for y in 1..h {
        for x in 0..w {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(w - 1);
            let c_u = (at(xp, y) - at(xm, y)).abs();
            let bias = sal_bias(x, y);

            let mut best_cost = prev_dp[x] + c_u + bias;
            let mut best_dir = 0i8;

            if x > 0 {
                let c_l = c_u + (at(x, y - 1) - at(xm, y)).abs();
                let cost_l = prev_dp[x - 1] + c_l + bias;
                if cost_l < best_cost {
                    best_cost = cost_l;
                    best_dir = -1;
                }
            }
            if x + 1 < w {
                let c_r = c_u + (at(x, y - 1) - at(xp, y)).abs();
                let cost_r = prev_dp[x + 1] + c_r + bias;
                if cost_r < best_cost {
                    best_cost = cost_r;
                    best_dir = 1;
                }
            }

            curr_dp[x] = best_cost;
            backtrack[y * w + x] = best_dir;
        }
        std::mem::swap(&mut prev_dp, &mut curr_dp);
    }

    let mut seam = vec![0usize; h];
    let mut best_x = 0usize;
    let mut best_v = f32::MAX;
    for (x, &v) in prev_dp.iter().enumerate() {
        if v < best_v {
            best_v = v;
            best_x = x;
        }
    }
    seam[h - 1] = best_x;
    for y in (1..h).rev() {
        let dir = backtrack[y * w + best_x];
        best_x = (best_x as isize + dir as isize).clamp(0, w as isize - 1) as usize;
        seam[y - 1] = best_x;
    }
    seam
}

fn project_and_snap_seam(
    proxy_seam: &[usize],
    pw: usize,
    ph: usize,
    full: &PlanarF32,
) -> Vec<usize> {
    let w = full.width as usize;
    let h = full.height as usize;
    let mut full_seam = vec![0usize; h];
    let mut prev_x = 0usize;

    for (y, slot) in full_seam.iter_mut().enumerate() {
        let py_f = (y as f32 / h.max(1) as f32) * (ph - 1) as f32;
        let py0 = (py_f.floor() as usize).min(ph - 1);
        let py1 = (py0 + 1).min(ph - 1);
        let frac = py_f - py0 as f32;
        let px_interp = proxy_seam[py0] as f32 * (1.0 - frac) + proxy_seam[py1] as f32 * frac;
        let base_x = ((px_interp / pw.max(1) as f32) * w as f32)
            .round()
            .clamp(0.0, (w - 1) as f32) as usize;

        let search_lo = if y == 0 {
            base_x.saturating_sub(4)
        } else {
            prev_x.saturating_sub(1).max(base_x.saturating_sub(4))
        };
        let search_hi = if y == 0 {
            (base_x + 4).min(w - 1)
        } else {
            (prev_x + 1).min(base_x + 4).min(w - 1)
        };

        let mut best_x = base_x.clamp(search_lo, search_hi);
        let mut best_e = f32::MAX;
        for cx in search_lo..=search_hi {
            let xm = cx.saturating_sub(1);
            let xp = (cx + 1).min(w - 1);
            let l0 = full.g[y * w + xm];
            let l1 = full.g[y * w + xp];
            let e = (l1 - l0).abs();
            if e < best_e {
                best_e = e;
                best_x = cx;
            }
        }
        *slot = best_x;
        prev_x = best_x;
    }
    full_seam
}

fn remove_single_seam(src: &PlanarF32, seam: &[usize]) -> EngineResult<PlanarF32> {
    let w = src.width as usize;
    let h = src.height as usize;
    let nw = w - 1;
    let mut dst = PlanarF32::new(nw as u32, src.height, src.has_alpha, src.working_space)?;
    dst.premultiplied = src.premultiplied;

    for (y, &sx) in seam.iter().enumerate().take(h) {
        let s_off = y * w;
        let d_off = y * nw;
        dst.r[d_off..d_off + sx].copy_from_slice(&src.r[s_off..s_off + sx]);
        dst.r[d_off + sx..d_off + nw].copy_from_slice(&src.r[s_off + sx + 1..s_off + w]);
        dst.g[d_off..d_off + sx].copy_from_slice(&src.g[s_off..s_off + sx]);
        dst.g[d_off + sx..d_off + nw].copy_from_slice(&src.g[s_off + sx + 1..s_off + w]);
        dst.b[d_off..d_off + sx].copy_from_slice(&src.b[s_off..s_off + sx]);
        dst.b[d_off + sx..d_off + nw].copy_from_slice(&src.b[s_off + sx + 1..s_off + w]);
        if src.has_alpha {
            dst.a[d_off..d_off + sx].copy_from_slice(&src.a[s_off..s_off + sx]);
            dst.a[d_off + sx..d_off + nw].copy_from_slice(&src.a[s_off + sx + 1..s_off + w]);
        }
    }
    Ok(dst)
}

fn extract_luma(img: &PlanarF32) -> Vec<f32> {
    img.r
        .iter()
        .zip(&img.g)
        .zip(&img.b)
        .map(|((&r, &g), &b)| 0.2126 * r + 0.7152 * g + 0.0722 * b)
        .collect()
}

fn downsample_luma(src: &PlanarF32, pw: usize, ph: usize) -> Vec<f32> {
    let sw = src.width as usize;
    let sh = src.height as usize;
    let mut out = vec![0.0f32; pw * ph];
    for py in 0..ph {
        let y0 = (py * sh) / ph;
        let y1 = (((py + 1) * sh) / ph).max(y0 + 1).min(sh);
        for px in 0..pw {
            let x0 = (px * sw) / pw;
            let x1 = (((px + 1) * sw) / pw).max(x0 + 1).min(sw);
            let mut acc = 0.0f32;
            let mut cnt = 0usize;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let i = sy * sw + sx;
                    acc += 0.2126 * src.r[i] + 0.7152 * src.g[i] + 0.0722 * src.b[i];
                    cnt += 1;
                }
            }
            out[py * pw + px] = acc / cnt.max(1) as f32;
        }
    }
    out
}

pub fn transpose_planar(src: &PlanarF32) -> EngineResult<PlanarF32> {
    let sw = src.width as usize;
    let sh = src.height as usize;
    let mut dst = PlanarF32::new(src.height, src.width, src.has_alpha, src.working_space)?;
    dst.premultiplied = src.premultiplied;
    for y in 0..sh {
        for x in 0..sw {
            let si = y * sw + x;
            let di = x * sh + y;
            dst.r[di] = src.r[si];
            dst.g[di] = src.g[si];
            dst.b[di] = src.b[si];
            if src.has_alpha {
                dst.a[di] = src.a[si];
            }
        }
    }
    Ok(dst)
}
