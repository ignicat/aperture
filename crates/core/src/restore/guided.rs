use crate::error::EngineResult;
use crate::pixel::PlanarF32;

pub fn guided_filter_denoise(
    src: &PlanarF32,
    radius: usize,
    eps: f32,
    detail_boost: f32,
) -> EngineResult<PlanarF32> {
    let w = src.width as usize;
    let h = src.height as usize;
    if w == 0 || h == 0 || radius == 0 {
        return Ok(src.clone());
    }

    let r_filtered = guided_channel(&src.r, w, h, radius, eps, detail_boost);
    let g_filtered = guided_channel(&src.g, w, h, radius, eps, detail_boost);
    let b_filtered = guided_channel(&src.b, w, h, radius, eps, detail_boost);

    let mut dst = src.clone();
    dst.r = r_filtered;
    dst.g = g_filtered;
    dst.b = b_filtered;
    Ok(dst)
}

pub fn box_mean_plane(src: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    if w == 0 || h == 0 || radius == 0 {
        return src.to_vec();
    }
    let mut horiz = vec![0.0f32; w * h];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let out_row = &mut horiz[y * w..(y + 1) * w];
        let mut acc = 0.0f64;
        let mut count = 0usize;
        for &val in row.iter().take(radius.min(w - 1) + 1) {
            acc += val as f64;
            count += 1;
        }
        for (x, slot) in out_row.iter_mut().enumerate().take(w) {
            *slot = (acc / count as f64) as f32;
            let add_x = x + radius + 1;
            if add_x < w {
                acc += row[add_x] as f64;
                count += 1;
            }
            if let Some(sub_x) = x.checked_sub(radius) {
                acc -= row[sub_x] as f64;
                count -= 1;
            }
        }
    }

    let mut out = vec![0.0f32; w * h];
    for x in 0..w {
        let mut acc = 0.0f64;
        let mut count = 0usize;
        for y in 0..=radius.min(h - 1) {
            acc += horiz[y * w + x] as f64;
            count += 1;
        }
        for y in 0..h {
            out[y * w + x] = (acc / count as f64) as f32;
            let add_y = y + radius + 1;
            if add_y < h {
                acc += horiz[add_y * w + x] as f64;
                count += 1;
            }
            if let Some(sub_y) = y.checked_sub(radius) {
                acc -= horiz[sub_y * w + x] as f64;
                count -= 1;
            }
        }
    }
    out
}

fn guided_channel(
    plane: &[f32],
    w: usize,
    h: usize,
    radius: usize,
    eps: f32,
    detail_boost: f32,
) -> Vec<f32> {
    let mean_i = box_mean_plane(plane, w, h, radius);
    let sq: Vec<f32> = plane.iter().map(|&v| v * v).collect();
    let mean_ii = box_mean_plane(&sq, w, h, radius);

    let len = w * h;
    let mut a = vec![0.0f32; len];
    let mut b = vec![0.0f32; len];
    let safe_eps = eps.max(1e-6);

    for i in 0..len {
        let var = (mean_ii[i] - mean_i[i] * mean_i[i]).max(0.0);
        let ai = var / (var + safe_eps);
        a[i] = ai;
        b[i] = (1.0 - ai) * mean_i[i];
    }

    let mean_a = box_mean_plane(&a, w, h, radius);
    let mean_b = box_mean_plane(&b, w, h, radius);
    let mut out = vec![0.0f32; len];

    for i in 0..len {
        let q = mean_a[i] * plane[i] + mean_b[i];
        let val = if detail_boost > 0.0 {
            q + detail_boost * (plane[i] - q)
        } else {
            q
        };
        out[i] = val.clamp(0.0, 1.0);
    }
    out
}
