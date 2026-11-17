use crate::error::{EngineError, EngineResult};
use moxcms::{ColorProfile, Layout, TransformOptions};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    Srgb,
    LinearSrgb,
    DisplayP3,
    AdobeRgb,
    Rec2020,
    Rec2020Pq,
    Rec2020Hlg,
    Gray,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkingSpace {
    LinearSrgb,
    Oklab,
    Sigmoidal(f32),
    PerceptualSrgb,
}

pub struct ColorLuts {
    pub u8_to_linear: [f32; 256],
    pub u8_to_u16_linear: [u16; 256],
    pub u16_linear_to_u8: Vec<u8>,
}

impl ColorLuts {
    pub fn build() -> Self {
        let mut u8_to_linear = [0.0f32; 256];
        let mut u8_to_u16_linear = [0u16; 256];
        for i in 0..256 {
            let lin = srgb_to_linear_f32(i as f32 / 255.0);
            u8_to_linear[i] = lin;
            u8_to_u16_linear[i] = (lin.clamp(0.0, 1.0) * 65535.0).round() as u16;
        }
        let mut u16_linear_to_u8 = vec![0u8; 65536];
        for (i, slot) in u16_linear_to_u8.iter_mut().enumerate() {
            let srgb = linear_to_srgb_f32(i as f32 / 65535.0);
            *slot = (srgb.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        Self {
            u8_to_linear,
            u8_to_u16_linear,
            u16_linear_to_u8,
        }
    }
}

static GLOBAL_LUTS: OnceLock<ColorLuts> = OnceLock::new();

pub fn global_luts() -> &'static ColorLuts {
    GLOBAL_LUTS.get_or_init(ColorLuts::build)
}

#[inline]
pub fn srgb_to_linear_f32(v: f32) -> f32 {
    let sign = v.signum();
    let abs = v.abs();
    let lin = if abs <= 0.04045 {
        abs / 12.92
    } else {
        ((abs + 0.055) / 1.055).powf(2.4)
    };
    if v == 0.0 { 0.0 } else { sign * lin }
}

#[inline]
pub fn linear_to_srgb_f32(v: f32) -> f32 {
    let sign = v.signum();
    let abs = v.abs();
    let srgb = if abs <= 0.0031308 {
        abs * 12.92
    } else {
        1.055 * abs.powf(1.0 / 2.4) - 0.055
    };
    if v == 0.0 { 0.0 } else { sign * srgb }
}

#[inline]
pub fn sigmoidal_forward(x: f32, contrast: f32) -> f32 {
    if contrast <= 0.01 {
        return x;
    }
    let clamped = x.clamp(0.0, 1.0);
    let phi = |t: f32| 1.0 / (1.0 + (contrast * (0.5 - t)).exp());
    let lo = phi(0.0);
    let hi = phi(1.0);
    let denom = hi - lo;
    if denom.abs() < 1e-6 {
        return clamped;
    }
    (phi(clamped) - lo) / denom
}

#[inline]
pub fn sigmoidal_inverse(s: f32, contrast: f32) -> f32 {
    if contrast <= 0.01 {
        return s;
    }
    let clamped = s.clamp(0.0, 1.0);
    let phi = |t: f32| 1.0 / (1.0 + (contrast * (0.5 - t)).exp());
    let lo = phi(0.0);
    let hi = phi(1.0);
    let y = (lo + clamped * (hi - lo)).clamp(1e-6, 1.0 - 1e-6);
    (0.5 - ((1.0 / y) - 1.0).ln() / contrast).clamp(0.0, 1.0)
}

#[inline]
pub fn linear_rgb_to_oklab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let l = 0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b;
    let m = 0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b;
    let s = 0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b;
    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();
    (
        0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_,
        1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_,
        0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_,
    )
}

#[inline]
pub fn oklab_to_linear_rgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let l_ = l + 0.396_337_78 * a + 0.215_803_76 * b;
    let m_ = l - 0.105_561_346 * a - 0.063_854_17 * b;
    let s_ = l - 0.089_484_18 * a - 1.291_485_5 * b;
    let l3 = l_ * l_ * l_;
    let m3 = m_ * m_ * m_;
    let s3 = s_ * s_ * s_;
    (
        4.076_741_7 * l3 - 3.307_711_6 * m3 + 0.230_969_94 * s3,
        -1.268_438 * l3 + 2.609_757_4 * m3 - 0.341_319_38 * s3,
        -0.004_196_086_3 * l3 - 0.703_418_6 * m3 + 1.707_614_7 * s3,
    )
}

#[inline]
pub fn oklab_to_linear_rgb_gamut_clipped(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    if l <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    if l >= 1.0 {
        return (1.0, 1.0, 1.0);
    }
    let (r, g, bl) = oklab_to_linear_rgb(l, a, b);
    if (0.0..=1.0).contains(&r) && (0.0..=1.0).contains(&g) && (0.0..=1.0).contains(&bl) {
        return (r, g, bl);
    }
    let mut lo = 0.0f32;
    let mut hi = 1.0f32;
    for _ in 0..14 {
        let mid = 0.5 * (lo + hi);
        let (cr, cg, cb) = oklab_to_linear_rgb(l, a * mid, b * mid);
        if (0.0..=1.0).contains(&cr) && (0.0..=1.0).contains(&cg) && (0.0..=1.0).contains(&cb) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (cr, cg, cb) = oklab_to_linear_rgb(l, a * lo, b * lo);
    (cr.clamp(0.0, 1.0), cg.clamp(0.0, 1.0), cb.clamp(0.0, 1.0))
}

#[inline]
pub fn linear_rgb_to_cie_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let x = (0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b) / 0.950_47;
    let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175_0 * b;
    let z = (0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b) / 1.088_83;
    let f = |t: f32| {
        if t > 0.008_856_452 {
            t.cbrt()
        } else {
            7.787_037 * t + 16.0 / 116.0
        }
    };
    let fx = f(x);
    let fy = f(y);
    let fz = f(z);
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

#[inline]
pub fn linear_rgb_to_xyb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    const BIAS: f32 = 0.003_793_073_3;
    let l = (0.3 * r + 0.622 * g + 0.078 * b + BIAS).max(0.0).cbrt();
    let m = (0.23 * r + 0.692 * g + 0.078 * b + BIAS).max(0.0).cbrt();
    let s = (0.243_422_69 * r + 0.204_767_45 * g + 0.551_809_85 * b + BIAS)
        .max(0.0)
        .cbrt();
    let c0 = BIAS.cbrt();
    let l0 = l - c0;
    let m0 = m - c0;
    let s0 = s - c0;
    (0.5 * (l0 - m0), 0.5 * (l0 + m0), s0)
}

pub fn mox_profile_for_space(space: ColorSpace) -> ColorProfile {
    match space {
        ColorSpace::DisplayP3 => ColorProfile::new_display_p3(),
        ColorSpace::AdobeRgb => ColorProfile::new_adobe_rgb(),
        ColorSpace::Rec2020 => ColorProfile::new_bt2020(),
        ColorSpace::Rec2020Pq => ColorProfile::new_bt2020_pq(),
        ColorSpace::Rec2020Hlg => ColorProfile::new_bt2020_hlg(),
        ColorSpace::Srgb
        | ColorSpace::LinearSrgb
        | ColorSpace::Gray
        | ColorSpace::Unknown => ColorProfile::new_srgb(),
    }
}

pub fn transform_icc_u8(
    src: &[u8],
    src_layout: Layout,
    src_icc: &[u8],
    dst_space: ColorSpace,
    dst_layout: Layout,
) -> EngineResult<Vec<u8>> {
    let src_profile = ColorProfile::new_from_slice(src_icc)
        .map_err(|e| EngineError::ColorTransform(format!("{e:?}")))?;
    let dst_profile = mox_profile_for_space(dst_space);
    let exec = src_profile
        .create_transform_8bit(src_layout, &dst_profile, dst_layout, TransformOptions::default())
        .map_err(|e| EngineError::ColorTransform(format!("{e:?}")))?;
    let src_ch = src_layout.channels();
    if src_ch == 0 || !src.len().is_multiple_of(src_ch) {
        return Err(EngineError::ColorTransform(
            "source buffer length does not align with layout channels".to_string(),
        ));
    }
    let pixels = src.len() / src_ch;
    let mut dst = vec![0u8; pixels * dst_layout.channels()];
    exec.transform(src, &mut dst)
        .map_err(|e| EngineError::ColorTransform(format!("{e:?}")))?;
    Ok(dst)
}

pub fn transform_icc_u16(
    src: &[u16],
    src_layout: Layout,
    src_icc: &[u8],
    dst_space: ColorSpace,
    dst_layout: Layout,
) -> EngineResult<Vec<u16>> {
    let src_profile = ColorProfile::new_from_slice(src_icc)
        .map_err(|e| EngineError::ColorTransform(format!("{e:?}")))?;
    let dst_profile = mox_profile_for_space(dst_space);
    let exec = src_profile
        .create_transform_16bit(src_layout, &dst_profile, dst_layout, TransformOptions::default())
        .map_err(|e| EngineError::ColorTransform(format!("{e:?}")))?;
    let src_ch = src_layout.channels();
    if src_ch == 0 || !src.len().is_multiple_of(src_ch) {
        return Err(EngineError::ColorTransform(
            "source buffer length does not align with layout channels".to_string(),
        ));
    }
    let pixels = src.len() / src_ch;
    let mut dst = vec![0u16; pixels * dst_layout.channels()];
    exec.transform(src, &mut dst)
        .map_err(|e| EngineError::ColorTransform(format!("{e:?}")))?;
    Ok(dst)
}

pub fn transform_icc_f32(
    src: &[f32],
    src_layout: Layout,
    src_icc: &[u8],
    dst_space: ColorSpace,
    dst_layout: Layout,
) -> EngineResult<Vec<f32>> {
    let src_profile = ColorProfile::new_from_slice(src_icc)
        .map_err(|e| EngineError::ColorTransform(format!("{e:?}")))?;
    let dst_profile = mox_profile_for_space(dst_space);
    let exec = src_profile
        .create_transform_f32(src_layout, &dst_profile, dst_layout, TransformOptions::default())
        .map_err(|e| EngineError::ColorTransform(format!("{e:?}")))?;
    let src_ch = src_layout.channels();
    if src_ch == 0 || !src.len().is_multiple_of(src_ch) {
        return Err(EngineError::ColorTransform(
            "source buffer length does not align with layout channels".to_string(),
        ));
    }
    let pixels = src.len() / src_ch;
    let mut dst = vec![0.0f32; pixels * dst_layout.channels()];
    exec.transform(src, &mut dst)
        .map_err(|e| EngineError::ColorTransform(format!("{e:?}")))?;
    Ok(dst)
}
