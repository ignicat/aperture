use crate::color::linear_rgb_to_xyb;
use crate::error::{EngineError, EngineResult};
use crate::pixel::PlanarF32;
use crate::restore::guided::box_mean_plane;

#[derive(Debug, Clone, PartialEq)]
pub struct QualityMetrics {
    pub width: u32,
    pub height: u32,
    pub mse: f64,
    pub psnr_db: f64,
    pub ssim: f64,
    pub ssim2_est: f64,
    pub butteraugli_est: Option<f64>,
    pub dssim_est: Option<f64>,
}

pub fn compare_images(reference: &PlanarF32, candidate: &PlanarF32) -> EngineResult<QualityMetrics> {
    if reference.width != candidate.width || reference.height != candidate.height {
        return Err(EngineError::InvalidDimensions {
            width: candidate.width,
            height: candidate.height,
        });
    }

    let n = reference.r.len().max(1);
    let w = reference.width as usize;
    let h = reference.height as usize;

    let mut mse_sum = 0.0f64;
    let mut ref_y = Vec::with_capacity(n);
    let mut cmp_y = Vec::with_capacity(n);
    let mut xyb_err_sum = 0.0f64;

    for i in 0..n {
        let r0 = reference.r[i];
        let g0 = reference.g[i];
        let b0 = reference.b[i];
        let r1 = candidate.r[i];
        let g1 = candidate.g[i];
        let b1 = candidate.b[i];

        let dr = (r0 - r1) as f64;
        let dg = (g0 - g1) as f64;
        let db = (b0 - b1) as f64;
        mse_sum += (dr * dr + dg * dg + db * db) / 3.0;

        let y0 = 0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0;
        let y1 = 0.2126 * r1 + 0.7152 * g1 + 0.0722 * b1;
        ref_y.push(y0);
        cmp_y.push(y1);

        let (x0, yy0, b_0) = linear_rgb_to_xyb(r0.max(0.0), g0.max(0.0), b0.max(0.0));
        let (x1, yy1, b_1) = linear_rgb_to_xyb(r1.max(0.0), g1.max(0.0), b1.max(0.0));
        let dx = (x0 - x1) as f64;
        let dy = (yy0 - yy1) as f64;
        let dz = (b_0 - b_1) as f64;
        xyb_err_sum += (dx * dx * 2.4 + dy * dy * 1.4 + dz * dz * 0.9).sqrt();
    }

    let mse = mse_sum / n as f64;
    let psnr_db = if mse <= 1e-12 {
        99.99
    } else {
        10.0 * (1.0 / mse).log10()
    };

    let ssim = compute_windowed_ssim(&ref_y, &cmp_y, w, h);
    let mean_xyb_delta = xyb_err_sum / n as f64;
    let ms_ssim = compute_multiscale_ssim(&ref_y, &cmp_y, w, h);

    let ssim2_est = if mse <= 1e-12 {
        100.0
    } else {
        let structural = (ms_ssim.clamp(0.0, 1.0)).powf(2.2) * 100.0;
        let xyb_penalty = (mean_xyb_delta * 195.0).min(45.0);
        (structural - xyb_penalty).clamp(0.0, 100.0)
    };

    let dssim_est = ((1.0 / ssim.max(1e-4)) - 1.0).max(0.0);
    let butteraugli_est = (mean_xyb_delta * 38.0).max(0.0);

    Ok(QualityMetrics {
        width: reference.width,
        height: reference.height,
        mse,
        psnr_db,
        ssim,
        ssim2_est,
        butteraugli_est: Some(butteraugli_est),
        dssim_est: Some(dssim_est),
    })
}

pub fn compute_windowed_ssim(a: &[f32], b: &[f32], w: usize, h: usize) -> f64 {
    let n = w * h;
    if n == 0 || a.len() != n || b.len() != n {
        return 1.0;
    }
    let r = 3usize.min(w / 2).min(h / 2).max(1);
    let mu_a = box_mean_plane(a, w, h, r);
    let mu_b = box_mean_plane(b, w, h, r);

    let a2: Vec<f32> = a.iter().map(|&v| v * v).collect();
    let b2: Vec<f32> = b.iter().map(|&v| v * v).collect();
    let ab: Vec<f32> = a.iter().zip(b).map(|(&x, &y)| x * y).collect();

    let sig_a2 = box_mean_plane(&a2, w, h, r);
    let sig_b2 = box_mean_plane(&b2, w, h, r);
    let sig_ab = box_mean_plane(&ab, w, h, r);

    let c1 = 0.01f64 * 0.01;
    let c2 = 0.03f64 * 0.03;
    let mut sum = 0.0f64;

    for i in 0..n {
        let ma = mu_a[i] as f64;
        let mb = mu_b[i] as f64;
        let va = (sig_a2[i] as f64 - ma * ma).max(0.0);
        let vb = (sig_b2[i] as f64 - mb * mb).max(0.0);
        let cov = sig_ab[i] as f64 - ma * mb;
        let num = (2.0 * ma * mb + c1) * (2.0 * cov + c2);
        let den = (ma * ma + mb * mb + c1) * (va + vb + c2);
        sum += (num / den.max(1e-12)).clamp(0.0, 1.0);
    }

    sum / n as f64
}

fn compute_multiscale_ssim(a: &[f32], b: &[f32], mut w: usize, mut h: usize) -> f64 {
    let weights = [0.45, 0.35, 0.20];
    let mut cur_a = a.to_vec();
    let mut cur_b = b.to_vec();
    let mut score = 0.0f64;
    let mut w_sum = 0.0f64;

    for &wt in &weights {
        if w < 4 || h < 4 {
            break;
        }
        let s = compute_windowed_ssim(&cur_a, &cur_b, w, h);
        score += wt * s;
        w_sum += wt;
        let nw = w / 2;
        let nh = h / 2;
        cur_a = downsample_2x_avg(&cur_a, w, h, nw, nh);
        cur_b = downsample_2x_avg(&cur_b, w, h, nw, nh);
        w = nw;
        h = nh;
    }

    if w_sum > 0.0 { score / w_sum } else { compute_windowed_ssim(a, b, w.max(1), h.max(1)) }
}

fn downsample_2x_avg(src: &[f32], w: usize, _h: usize, nw: usize, nh: usize) -> Vec<f32> {
    let mut dst = Vec::with_capacity(nw * nh);
    for y in 0..nh {
        let sy = y * 2;
        for x in 0..nw {
            let sx = x * 2;
            let avg = 0.25
                * (src[sy * w + sx]
                    + src[sy * w + sx + 1]
                    + src[(sy + 1) * w + sx]
                    + src[(sy + 1) * w + sx + 1]);
            dst.push(avg);
        }
    }
    dst
}

pub fn export_palette_json(colors: &[[u8; 3]]) -> String {
    let mut out = String::from("{\n  \"palette\": [\n");
    for (idx, c) in colors.iter().enumerate() {
        let comma = if idx + 1 < colors.len() { "," } else { "" };
        out.push_str(&format!(
            "    {{ \"hex\": \"#{:02x}{:02x}{:02x}\", \"rgb\": [{}, {}, {}] }}{}\n",
            c[0], c[1], c[2], c[0], c[1], c[2], comma
        ));
    }
    out.push_str("  ]\n}");
    out
}

pub fn export_palette_css(colors: &[[u8; 3]]) -> String {
    let mut out = String::from(":root {\n");
    for (idx, c) in colors.iter().enumerate() {
        out.push_str(&format!(
            "  --palette-{}: #{:02x}{:02x}{:02x};\n",
            idx + 1,
            c[0],
            c[1],
            c[2]
        ));
    }
    out.push('}');
    out
}
