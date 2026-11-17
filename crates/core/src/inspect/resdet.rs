use crate::arena::PlanCache;
use crate::error::EngineResult;
use crate::pixel::PlanarF32;
use crate::pool::ThreadPool;
use crate::resample::kernels::ResampleFilter;
use crate::resample::separable::resample_separable;
use rustdct::DctPlanner;

#[derive(Debug, Clone, PartialEq)]
pub struct NativeResolutionReport {
    pub width: u32,
    pub height: u32,
    pub detected_native_width: Option<u32>,
    pub detected_native_height: Option<u32>,
    pub confidence_x: f32,
    pub confidence_y: f32,
    pub is_fake_upscale: bool,
    pub x_cutoff_ratio: f32,
    pub y_cutoff_ratio: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct AxisCutoff {
    pub native_len: u32,
    pub confidence: f32,
    pub ratio: f32,
}

pub fn detect_native_resolution(
    src: &PlanarF32,
    cache: &PlanCache,
    pool: &ThreadPool,
) -> NativeResolutionReport {
    let w = src.width as usize;
    let h = src.height as usize;
    if w < 32 || h < 32 {
        return NativeResolutionReport {
            width: src.width,
            height: src.height,
            detected_native_width: None,
            detected_native_height: None,
            confidence_x: 0.0,
            confidence_y: 0.0,
            is_fake_upscale: false,
            x_cutoff_ratio: 1.0,
            y_cutoff_ratio: 1.0,
        };
    }

    let lum: Vec<f32> = src
        .r
        .iter()
        .zip(&src.g)
        .zip(&src.b)
        .map(|((&r, &g), &b)| 0.2126 * r + 0.7152 * g + 0.0722 * b)
        .collect();

    let x_cutoff = estimate_axis_cutoff(&lum, w, h, true);
    let y_cutoff = estimate_axis_cutoff(&lum, w, h, false);

    let (conf_x, cand_w) = refine_axis_with_roundtrip(src, x_cutoff, true, cache, pool);
    let (conf_y, cand_h) = refine_axis_with_roundtrip(src, y_cutoff, false, cache, pool);

    let valid_x = conf_x >= 0.42 && cand_w < src.width;
    let valid_y = conf_y >= 0.42 && cand_h < src.height;
    let is_fake_upscale = (valid_x && valid_y) || (valid_x && conf_x >= 0.65) || (valid_y && conf_y >= 0.65);

    NativeResolutionReport {
        width: src.width,
        height: src.height,
        detected_native_width: if valid_x { Some(cand_w) } else { None },
        detected_native_height: if valid_y { Some(cand_h) } else { None },
        confidence_x: conf_x,
        confidence_y: conf_y,
        is_fake_upscale,
        x_cutoff_ratio: if valid_x { cand_w as f32 / src.width as f32 } else { 1.0 },
        y_cutoff_ratio: if valid_y { cand_h as f32 / src.height as f32 } else { 1.0 },
    }
}

pub fn descale_if_suspected(
    src: &PlanarF32,
    min_confidence: f32,
    cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<(PlanarF32, NativeResolutionReport, bool)> {
    let report = detect_native_resolution(src, cache, pool);
    let max_conf = report.confidence_x.max(report.confidence_y);
    if !report.is_fake_upscale || max_conf < min_confidence {
        return Ok((src.clone(), report, false));
    }

    let nw = report.detected_native_width.unwrap_or_else(|| {
        let ratio = report.y_cutoff_ratio;
        ((src.width as f32 * ratio).round() as u32).clamp(16, src.width)
    });
    let nh = report.detected_native_height.unwrap_or_else(|| {
        let ratio = report.x_cutoff_ratio;
        ((src.height as f32 * ratio).round() as u32).clamp(16, src.height)
    });

    if nw >= src.width && nh >= src.height {
        return Ok((src.clone(), report, false));
    }

    let descaled = resample_separable(src, nw, nh, ResampleFilter::Lanczos3, 0.0, cache, pool)?;
    Ok((descaled, report, true))
}

fn estimate_axis_cutoff(lum: &[f32], w: usize, h: usize, horizontal: bool) -> AxisCutoff {
    let axis_len = if horizontal { w } else { h };
    let cross_len = if horizontal { h } else { w };
    let stride = (cross_len / 96).max(1);

    let mut planner = DctPlanner::new();
    let dct = planner.plan_dct2(axis_len);
    let mut mag_acc = vec![0.0f64; axis_len];
    let mut line = vec![0.0f32; axis_len];
    let mut lines_counted = 0usize;

    for s in (0..cross_len).step_by(stride) {
        let mut mean = 0.0f64;
        if horizontal {
            let row = &lum[s * w..(s + 1) * w];
            line.copy_from_slice(row);
            for &v in row {
                mean += v as f64;
            }
        } else {
            for (y, dst) in line.iter_mut().enumerate() {
                let v = lum[y * w + s];
                *dst = v;
                mean += v as f64;
            }
        }
        let m = (mean / axis_len as f64) as f32;
        for (i, v) in line.iter_mut().enumerate() {
            let hann = 0.5
                * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (axis_len - 1).max(1) as f32).cos());
            *v = (*v - m) * (0.2 + 0.8 * hann);
        }
        dct.process_dct2(&mut line);
        for (acc, &c) in mag_acc.iter_mut().zip(&line) {
            *acc += (c as f64).abs();
        }
        lines_counted += 1;
    }

    let norm = (lines_counted.max(1) as f64) * (axis_len as f64).sqrt();
    for v in &mut mag_acc {
        *v /= norm;
    }

    let win = (axis_len / 48).clamp(2, 8);
    let mut smooth = vec![0.0f64; axis_len];
    for (k, slot) in smooth.iter_mut().enumerate() {
        let lo = k.saturating_sub(win);
        let hi = (k + win + 1).min(axis_len);
        let sum: f64 = mag_acc[lo..hi].iter().sum();
        *slot = sum / (hi - lo) as f64;
    }

    let min_k = (axis_len / 5).max(8);
    let max_k = ((axis_len * 11) / 12).min(axis_len.saturating_sub(4));
    if max_k <= min_k + 4 {
        return AxisCutoff {
            native_len: axis_len as u32,
            confidence: 0.0,
            ratio: 1.0,
        };
    }

    let low_band: f64 = smooth[2..min_k].iter().sum::<f64>() / (min_k - 2).max(1) as f64;
    let tail_band: f64 =
        smooth[axis_len * 7 / 8..].iter().sum::<f64>() / (axis_len - axis_len * 7 / 8).max(1) as f64;
    if low_band <= 1e-6 || tail_band / low_band > 0.22 {
        return AxisCutoff {
            native_len: axis_len as u32,
            confidence: 0.0,
            ratio: 1.0,
        };
    }

    let mut best_k = axis_len;
    let mut best_score = 0.0f64;

    let span = (axis_len / 20).clamp(3, 12);
    for k in min_k..max_k {
        let left_start = k.saturating_sub(span);
        let right_end = (k + span * 2).min(axis_len);
        let left_mean = smooth[left_start..k].iter().sum::<f64>() / (k - left_start).max(1) as f64;
        let right_mean = smooth[k..right_end].iter().sum::<f64>() / (right_end - k).max(1) as f64;
        let drop = (left_mean - right_mean).max(0.0) / (left_mean + tail_band + 1e-6);
        let residual_suppr = 1.0 - (right_mean / low_band).clamp(0.0, 1.0);
        let score = drop * residual_suppr;
        if score > best_score {
            best_score = score;
            best_k = k;
        }
    }

    let tail_ratio = smooth[best_k.min(axis_len - 1)..]
        .iter()
        .sum::<f64>()
        / ((axis_len - best_k.min(axis_len - 1)).max(1) as f64 * low_band + 1e-6);

    let conf = (best_score * 1.45 * (1.0 - tail_ratio.min(1.0))).clamp(0.0, 1.0) as f32;
    let snapped = snap_common_dimension(best_k as u32, axis_len as u32);
    AxisCutoff {
        native_len: snapped,
        confidence: conf,
        ratio: snapped as f32 / axis_len as f32,
    }
}

fn snap_common_dimension(detected: u32, full: u32) -> u32 {
    let common = [
        360u32, 480, 540, 576, 720, 768, 800, 900, 960, 1080, 1200, 1280, 1440, 1600, 1920,
        2160, 2560, 3840,
    ];
    for &c in &common {
        if c < full && detected.abs_diff(c) <= (full / 40).max(4) {
            return c;
        }
    }
    if full.is_multiple_of(2) && detected.abs_diff(full / 2) <= (full / 32).max(4) {
        return full / 2;
    }
    if full.is_multiple_of(3) && detected.abs_diff(full / 3) <= (full / 32).max(4) {
        return full / 3;
    }
    detected.max(16).min(full)
}

fn refine_axis_with_roundtrip(
    src: &PlanarF32,
    est: AxisCutoff,
    horizontal: bool,
    cache: &PlanCache,
    pool: &ThreadPool,
) -> (f32, u32) {
    if est.confidence < 0.25 || est.native_len >= (if horizontal { src.width } else { src.height }) {
        return (est.confidence * 0.5, if horizontal { src.width } else { src.height });
    }

    let axis_full = if horizontal { src.width } else { src.height };
    let cand = est.native_len.clamp(16, axis_full - 1);
    let mid_up = ((cand + axis_full) / 2).min(axis_full - 1);
    let test_half = (cand * 3 / 4).max(8);

    let err_at = |len: u32| -> f64 {
        let (dw, dh) = if horizontal {
            (len, (src.height / 2).max(16))
        } else {
            ((src.width / 2).max(16), len)
        };
        let Ok(down) = resample_separable(src, dw, dh, ResampleFilter::Mitchell, 0.0, cache, pool) else {
            return 1.0;
        };
        let Ok(back) = resample_separable(
            &down,
            src.width,
            src.height,
            ResampleFilter::Mitchell,
            0.0,
            cache,
            pool,
        ) else {
            return 1.0;
        };
        let mut mse = 0.0f64;
        for (i, ((&r0, &g0), &b0)) in src.r.iter().zip(&src.g).zip(&src.b).enumerate() {
            let dr = (r0 - back.r[i]) as f64;
            let dg = (g0 - back.g[i]) as f64;
            let db = (b0 - back.b[i]) as f64;
            mse += dr * dr + dg * dg + db * db;
        }
        (mse / (src.r.len().max(1) * 3) as f64).sqrt()
    };

    let e_cand = err_at(cand);
    let e_half = err_at(test_half);
    let e_higher = err_at(mid_up);

    let valley_boost = if e_cand <= 0.018 && e_half > e_cand * 1.35 {
        0.35
    } else if e_cand > 0.045 {
        -0.30
    } else {
        0.10
    };
    let flat_penalty = if e_higher <= e_cand * 0.45 && e_cand > 0.025 {
        -0.20
    } else {
        0.0
    };

    let final_conf = (est.confidence + valley_boost + flat_penalty).clamp(0.0, 1.0);
    (final_conf, cand)
}
