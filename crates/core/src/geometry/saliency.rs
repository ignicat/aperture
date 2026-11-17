use crate::color::{WorkingSpace, linear_rgb_to_cie_lab};
use crate::geometry::trim::CropRect;
use crate::pixel::PlanarF32;
use crate::restore::guided::box_mean_plane;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SafeZone {
    pub top: f32,
    pub bottom: f32,
    pub left: f32,
    pub right: f32,
}

impl Default for SafeZone {
    fn default() -> Self {
        Self {
            top: 0.0,
            bottom: 0.0,
            left: 0.0,
            right: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SaliencyMap {
    pub width: usize,
    pub height: usize,
    pub scores: Vec<f32>,
    pub integral: Vec<f64>,
    pub total_mass: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SaliencyCropDecision {
    pub rect: CropRect,
    pub border_loss_ratio: f32,
    pub peak_x: f32,
    pub peak_y: f32,
}

pub fn compute_saliency_map(src: &PlanarF32) -> SaliencyMap {
    let mut work = src.clone();
    work.convert_working_space(WorkingSpace::LinearSrgb);

    let max_side = 192.0f32;
    let scale = (work.width.max(work.height) as f32 / max_side).max(1.0);
    let pw = ((work.width as f32 / scale).round() as usize).clamp(8, 192);
    let ph = ((work.height as f32 / scale).round() as usize).clamp(8, 192);

    let (pr, pg, pb) = downsample_proxy(&work, pw, ph);
    let mut lum = vec![0.0f32; pw * ph];
    let mut feature = vec![0.0f32; pw * ph];

    for i in 0..pw * ph {
        lum[i] = 0.2126 * pr[i] + 0.7152 * pg[i] + 0.0722 * pb[i];
    }

    let spec_res = spectral_residual_64(&lum, pw, ph);

    for y in 0..ph {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(ph - 1);
        for x in 0..pw {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(pw - 1);
            let idx = y * pw + x;

            let gx = (lum[ym * pw + xp] + 2.0 * lum[y * pw + xp] + lum[yp * pw + xp])
                - (lum[ym * pw + xm] + 2.0 * lum[y * pw + xm] + lum[yp * pw + xm]);
            let gy = (lum[yp * pw + xm] + 2.0 * lum[yp * pw + x] + lum[yp * pw + xp])
                - (lum[ym * pw + xm] + 2.0 * lum[ym * pw + x] + lum[ym * pw + xp]);
            let edge = (gx * gx + gy * gy).sqrt().min(2.0) * 0.5;

            let r = pr[idx];
            let g = pg[idx];
            let b = pb[idx];
            let max_c = r.max(g).max(b);
            let min_c = r.min(g).min(b);
            let sat = if max_c - min_c > 1e-4 {
                let sum = max_c + min_c;
                let denom = if sum > 1.0 { 2.0 - sum } else { sum };
                if denom > 1e-4 { (max_c - min_c) / denom } else { 0.0 }
            } else {
                0.0
            };

            let (lab_l, lab_a, lab_b) = linear_rgb_to_cie_lab(r, g, b);
            let skin = if (20.0..=92.0).contains(&lab_l) && (2.0..=24.0).contains(&lab_a) && (4.0..=32.0).contains(&lab_b) {
                let da = (lab_a - 12.5) / 11.5;
                let db = (lab_b - 18.0) / 14.0;
                (1.0 - (da * da + db * db).min(1.0)).max(0.0)
            } else {
                0.0
            };

            let sr = spec_res[idx];
            feature[idx] = edge * 0.35 + skin * 0.30 + sat * 0.15 + sr * 0.40;
        }
    }

    let smoothed = box_mean_plane(&feature, pw, ph, 3);
    let mut integral = vec![0.0f64; (pw + 1) * (ph + 1)];
    for y in 0..ph {
        let mut row_sum = 0.0f64;
        for x in 0..pw {
            row_sum += smoothed[y * pw + x].max(0.0) as f64;
            integral[(y + 1) * (pw + 1) + (x + 1)] =
                integral[y * (pw + 1) + (x + 1)] + row_sum;
        }
    }
    let total_mass = integral[ph * (pw + 1) + pw];

    SaliencyMap {
        width: pw,
        height: ph,
        scores: smoothed,
        integral,
        total_mass,
    }
}

impl SaliencyMap {
    pub fn rect_sum(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        let ax0 = x0.min(self.width);
        let ax1 = x1.clamp(ax0, self.width);
        let ay0 = y0.min(self.height);
        let ay1 = y1.clamp(ay0, self.height);
        let stride = self.width + 1;
        self.integral[ay1 * stride + ax1]
            - self.integral[ay0 * stride + ax1]
            - self.integral[ay1 * stride + ax0]
            + self.integral[ay0 * stride + ax0]
    }

    pub fn sample_norm(&self, nx: f32, ny: f32) -> f32 {
        let x = ((nx.clamp(0.0, 1.0)) * (self.width - 1) as f32).round() as usize;
        let y = ((ny.clamp(0.0, 1.0)) * (self.height - 1) as f32).round() as usize;
        self.scores[y * self.width + x]
    }
}

pub fn choose_smart_crop(
    src: &PlanarF32,
    target_w: u32,
    target_h: u32,
    safe: SafeZone,
) -> SaliencyCropDecision {
    let sw = src.width;
    let sh = src.height;
    let target_ar = target_w as f64 / target_h as f64;
    let src_ar = sw as f64 / sh as f64;

    let (crop_w, crop_h) = if src_ar > target_ar {
        let cw = ((sh as f64 * target_ar).round() as u32).clamp(1, sw);
        (cw, sh)
    } else {
        let ch = ((sw as f64 / target_ar).round() as u32).clamp(1, sh);
        (sw, ch)
    };

    if crop_w == sw && crop_h == sh {
        return SaliencyCropDecision {
            rect: CropRect {
                x: 0,
                y: 0,
                width: sw,
                height: sh,
            },
            border_loss_ratio: 0.0,
            peak_x: 0.5,
            peak_y: 0.5,
        };
    }

    let map = compute_saliency_map(src);
    let pw = map.width;
    let ph = map.height;

    let win_w = ((crop_w as f64 / sw as f64) * pw as f64)
        .round()
        .clamp(1.0, pw as f64) as usize;
    let win_h = ((crop_h as f64 / sh as f64) * ph as f64)
        .round()
        .clamp(1.0, ph as f64) as usize;

    let inner_x0 = ((safe.left.clamp(0.0, 0.4)) * win_w as f32).round() as usize;
    let inner_x1 = win_w.saturating_sub(((safe.right.clamp(0.0, 0.4)) * win_w as f32).round() as usize);
    let inner_y0 = ((safe.top.clamp(0.0, 0.4)) * win_h as f32).round() as usize;
    let inner_y1 = win_h.saturating_sub(((safe.bottom.clamp(0.0, 0.4)) * win_h as f32).round() as usize);

    let mut best_score = f64::MIN;
    let mut best_px = (pw - win_w) / 2;
    let mut best_py = (ph - win_h) / 2;
    let mut best_inside = 0.0f64;

    for py in 0..=ph.saturating_sub(win_h) {
        for px in 0..=pw.saturating_sub(win_w) {
            let inside = map.rect_sum(px, py, px + win_w, py + win_h);
            let safe_inside = map.rect_sum(
                px + inner_x0,
                py + inner_y0,
                px + inner_x1.max(inner_x0 + 1),
                py + inner_y1.max(inner_y0 + 1),
            );
            let cx = (px as f64 + win_w as f64 * 0.5) / pw as f64;
            let cy = (py as f64 + win_h as f64 * 0.5) / ph as f64;
            let center_bias = 1.0 - 0.08 * ((cx - 0.5).abs() + (cy - 0.5).abs());
            let thirds_bonus = 0.04
                * ((-((cx - 0.333).abs().min((cx - 0.667).abs())) * 12.0).exp()
                    + (-((cy - 0.333).abs().min((cy - 0.667).abs())) * 12.0).exp());
            let score = (inside * 0.65 + safe_inside * 0.35) * (center_bias + thirds_bonus);
            if score > best_score {
                best_score = score;
                best_px = px;
                best_py = py;
                best_inside = inside;
            }
        }
    }

    let mut peak_val = -1.0f32;
    let mut peak_x = 0.5f32;
    let mut peak_y = 0.5f32;
    for y in 0..ph {
        for x in 0..pw {
            let s = map.scores[y * pw + x];
            if s > peak_val {
                peak_val = s;
                peak_x = (x as f32 + 0.5) / pw as f32;
                peak_y = (y as f32 + 0.5) / ph as f32;
            }
        }
    }

    let out_x = if pw > win_w {
        ((best_px as f64 / (pw - win_w) as f64) * (sw - crop_w) as f64)
            .round()
            .clamp(0.0, (sw - crop_w) as f64) as u32
    } else {
        0
    };
    let out_y = if ph > win_h {
        ((best_py as f64 / (ph - win_h) as f64) * (sh - crop_h) as f64)
            .round()
            .clamp(0.0, (sh - crop_h) as f64) as u32
    } else {
        0
    };

    let border_loss_ratio = if map.total_mass > 1e-6 {
        ((map.total_mass - best_inside) / map.total_mass).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };

    SaliencyCropDecision {
        rect: CropRect {
            x: out_x,
            y: out_y,
            width: crop_w,
            height: crop_h,
        },
        border_loss_ratio,
        peak_x,
        peak_y,
    }
}

fn downsample_proxy(src: &PlanarF32, pw: usize, ph: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let sw = src.width as usize;
    let sh = src.height as usize;
    let mut r = vec![0.0f32; pw * ph];
    let mut g = vec![0.0f32; pw * ph];
    let mut b = vec![0.0f32; pw * ph];

    for py in 0..ph {
        let y0 = (py * sh) / ph;
        let y1 = (((py + 1) * sh) / ph).max(y0 + 1).min(sh);
        for px in 0..pw {
            let x0 = (px * sw) / pw;
            let x1 = (((px + 1) * sw) / pw).max(x0 + 1).min(sw);
            let mut sr = 0.0f32;
            let mut sg = 0.0f32;
            let mut sb = 0.0f32;
            let mut cnt = 0usize;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let i = sy * sw + sx;
                    sr += src.r[i];
                    sg += src.g[i];
                    sb += src.b[i];
                    cnt += 1;
                }
            }
            let inv = 1.0 / cnt.max(1) as f32;
            let oi = py * pw + px;
            r[oi] = sr * inv;
            g[oi] = sg * inv;
            b[oi] = sb * inv;
        }
    }
    (r, g, b)
}

fn spectral_residual_64(lum: &[f32], pw: usize, ph: usize) -> Vec<f32> {
    const N: usize = 64;
    let mut grid = vec![Complex::new(0.0f32, 0.0f32); N * N];
    for gy in 0..N {
        let sy = (gy * ph / N).min(ph - 1);
        for gx in 0..N {
            let sx = (gx * pw / N).min(pw - 1);
            grid[gy * N + gx] = Complex::new(lum[sy * pw + sx], 0.0);
        }
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft_fwd = planner.plan_fft_forward(N);
    let fft_inv = planner.plan_fft_inverse(N);

    for y in 0..N {
        fft_fwd.process(&mut grid[y * N..(y + 1) * N]);
    }
    let mut col = vec![Complex::new(0.0f32, 0.0f32); N];
    for x in 0..N {
        for y in 0..N {
            col[y] = grid[y * N + x];
        }
        fft_fwd.process(&mut col);
        for y in 0..N {
            grid[y * N + x] = col[y];
        }
    }

    let mut log_amp = vec![0.0f32; N * N];
    let mut phase = vec![(0.0f32, 0.0f32); N * N];
    for i in 0..N * N {
        let mag = grid[i].norm().max(1e-6);
        log_amp[i] = mag.ln();
        phase[i] = (grid[i].re / mag, grid[i].im / mag);
    }

    let avg_log = box_mean_plane(&log_amp, N, N, 1);
    for i in 0..N * N {
        let res = (log_amp[i] - avg_log[i]).exp();
        grid[i] = Complex::new(res * phase[i].0, res * phase[i].1);
    }

    for y in 0..N {
        fft_inv.process(&mut grid[y * N..(y + 1) * N]);
    }
    for x in 0..N {
        for y in 0..N {
            col[y] = grid[y * N + x];
        }
        fft_inv.process(&mut col);
        for y in 0..N {
            grid[y * N + x] = col[y];
        }
    }

    let norm = 1.0 / (N * N) as f32;
    let mut sal_64 = vec![0.0f32; N * N];
    let mut max_v = 1e-6f32;
    for i in 0..N * N {
        let v = (grid[i].norm() * norm).powi(2);
        sal_64[i] = v;
        max_v = max_v.max(v);
    }
    let sal_blur = box_mean_plane(&sal_64, N, N, 2);

    let mut out = vec![0.0f32; pw * ph];
    for y in 0..ph {
        let gy = (y * N / ph).min(N - 1);
        for x in 0..pw {
            let gx = (x * N / pw).min(N - 1);
            out[y * pw + x] = (sal_blur[gy * N + gx] / max_v).clamp(0.0, 1.0);
        }
    }
    out
}
