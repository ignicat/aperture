use crate::arena::CompiledAxisPlan;
use crate::error::EngineResult;
use crate::pixel::PlanarF32;
use crate::pool::ThreadPool;

pub fn resample_separable_planar_f32(
    src: &PlanarF32,
    horiz: &CompiledAxisPlan,
    vert: &CompiledAxisPlan,
    antiring: f32,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    let dst_w = horiz.dst_len;
    let dst_h = vert.dst_len;
    let mut dst = PlanarF32::new(dst_w, dst_h, src.has_alpha, src.working_space)?;
    dst.premultiplied = src.premultiplied;

    let src_w = src.width as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;

    let mut tmp_r = vec![0.0f32; src_w * dh];
    let mut tmp_g = vec![0.0f32; src_w * dh];
    let mut tmp_b = vec![0.0f32; src_w * dh];
    let mut tmp_a = if src.has_alpha {
        vec![0.0f32; src_w * dh]
    } else {
        Vec::new()
    };

    vertical_pass_blocked(
        &src.r,
        &mut tmp_r,
        src_w,
        vert,
        antiring,
        pool,
    );
    vertical_pass_blocked(
        &src.g,
        &mut tmp_g,
        src_w,
        vert,
        antiring,
        pool,
    );
    vertical_pass_blocked(
        &src.b,
        &mut tmp_b,
        src_w,
        vert,
        antiring,
        pool,
    );
    if src.has_alpha {
        vertical_pass_blocked(
            &src.a,
            &mut tmp_a,
            src_w,
            vert,
            antiring,
            pool,
        );
    }

    horizontal_pass_rows(
        &tmp_r,
        &mut dst.r,
        src_w,
        dw,
        dh,
        horiz,
        antiring,
        pool,
    );
    horizontal_pass_rows(
        &tmp_g,
        &mut dst.g,
        src_w,
        dw,
        dh,
        horiz,
        antiring,
        pool,
    );
    horizontal_pass_rows(
        &tmp_b,
        &mut dst.b,
        src_w,
        dw,
        dh,
        horiz,
        antiring,
        pool,
    );
    if src.has_alpha {
        horizontal_pass_rows(
            &tmp_a,
            &mut dst.a,
            src_w,
            dw,
            dh,
            horiz,
            antiring,
            pool,
        );
    }

    Ok(dst)
}

fn vertical_pass_blocked(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    vert: &CompiledAxisPlan,
    antiring: f32,
    pool: &ThreadPool,
) {
    let ar = antiring.clamp(0.0, 1.0);
    pool.parallel_row_stripes_mut(dst, width, |y_start, stripe| {
        for (local_y, out_row) in stripe.chunks_exact_mut(width).enumerate() {
            let dy = y_start + local_y;
            let sy_start = vert.src_starts[dy] as usize;
            let taps = vert.tap_counts[dy] as usize;
            let w_off = vert.weight_offsets[dy] as usize;
            let weights = &vert.weights[w_off..w_off + taps];
            vertical_row_dispatch(src, out_row, width, sy_start, weights, ar);
        }
    });
}

fn vertical_row_dispatch(
    src: &[f32],
    out_row: &mut [f32],
    width: usize,
    sy_start: usize,
    weights: &[f32],
    antiring: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        let required_src = sy_start
            .saturating_add(weights.len())
            .saturating_mul(width);
        if antiring <= 0.0
            && out_row.len() >= width
            && required_src <= src.len()
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            unsafe {
                vertical_row_avx2_fma(src, out_row, width, sy_start, weights);
            }
            return;
        }
    }
    vertical_row_scalar(src, out_row, width, sy_start, weights, antiring);
}

pub fn vertical_row_scalar(
    src: &[f32],
    out_row: &mut [f32],
    width: usize,
    sy_start: usize,
    weights: &[f32],
    antiring: f32,
) {
    for slot in out_row.iter_mut() {
        *slot = 0.0;
    }
    const BLOCK: usize = 64;
    let mut x0 = 0usize;
    while x0 < width {
        let x1 = (x0 + BLOCK).min(width);
        for (tap_i, &w) in weights.iter().enumerate() {
            let src_row = &src[(sy_start + tap_i) * width + x0..(sy_start + tap_i) * width + x1];
            let dst_slice = &mut out_row[x0..x1];
            for (d, &s) in dst_slice.iter_mut().zip(src_row.iter()) {
                *d += s * w;
            }
        }
        if antiring > 0.0 && weights.len() >= 2 {
            let mid = weights.len() / 2;
            let r0 = &src[(sy_start + mid - 1) * width + x0..(sy_start + mid - 1) * width + x1];
            let r1 = &src[(sy_start + mid) * width + x0..(sy_start + mid) * width + x1];
            let dst_slice = &mut out_row[x0..x1];
            for i in 0..dst_slice.len() {
                let lo = r0[i].min(r1[i]);
                let hi = r0[i].max(r1[i]);
                let clamped = dst_slice[i].clamp(lo, hi);
                dst_slice[i] = dst_slice[i] + (clamped - dst_slice[i]) * antiring;
            }
        }
        x0 = x1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vertical_row_avx2_fma(
    src: &[f32],
    out_row: &mut [f32],
    width: usize,
    sy_start: usize,
    weights: &[f32],
) {
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let mut x = 0usize;
    while x + 8 <= width {
        let mut acc = _mm256_setzero_ps();
        for (tap_i, &w) in weights.iter().enumerate() {
            let wv = _mm256_set1_ps(w);
            let idx = (sy_start + tap_i) * width + x;
            let sv = unsafe { _mm256_loadu_ps(src.as_ptr().add(idx)) };
            acc = _mm256_fmadd_ps(sv, wv, acc);
        }
        unsafe { _mm256_storeu_ps(out_row.as_mut_ptr().add(x), acc) };
        x += 8;
    }
    while x < width {
        let mut sum = 0.0f32;
        for (tap_i, &w) in weights.iter().enumerate() {
            sum += src[(sy_start + tap_i) * width + x] * w;
        }
        out_row[x] = sum;
        x += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn horizontal_pass_rows(
    src: &[f32],
    dst: &mut [f32],
    src_w: usize,
    dst_w: usize,
    _dst_h: usize,
    horiz: &CompiledAxisPlan,
    antiring: f32,
    pool: &ThreadPool,
) {
    let ar = antiring.clamp(0.0, 1.0);
    pool.parallel_row_stripes_mut(dst, dst_w, |y_start, stripe| {
        for (local_y, dst_row) in stripe.chunks_exact_mut(dst_w).enumerate() {
            let y = y_start + local_y;
            let src_row = &src[y * src_w..(y + 1) * src_w];
            for (dx, slot) in dst_row.iter_mut().enumerate() {
                let sx_start = horiz.src_starts[dx] as usize;
                let taps = horiz.tap_counts[dx] as usize;
                let w_off = horiz.weight_offsets[dx] as usize;
                let weights = &horiz.weights[w_off..w_off + taps];
                let mut sum = 0.0f32;
                for (t, &w) in weights.iter().enumerate() {
                    sum += src_row[sx_start + t] * w;
                }
                if ar > 0.0 && taps >= 2 {
                    let mid = taps / 2;
                    let a = src_row[sx_start + mid - 1];
                    let b = src_row[sx_start + mid];
                    let clamped = sum.clamp(a.min(b), a.max(b));
                    sum += (clamped - sum) * ar;
                }
                *slot = sum;
            }
        }
    });
}
