use crate::arena::CompiledAxisPlan;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResampleFilter {
    Auto,
    Adaptive,
    Fsr,
    Ibp,
    EwaLanczos,
    EwaLanczosSharp,
    EwaRobidoux,
    EwaRobidouxSharp,
    EwaGinseng,
    Lanczos3,
    Lanczos4,
    SsimDown,
    SsimUp,
    Icbi,
    Nedi,
    LineArt,
    Mmpx,
    Xbrz,
    ScaleFx,
    Mitchell,
    CatmullRom,
    Hermite,
    Box,
    Nearest,
}

impl ResampleFilter {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "adaptive" => Some(Self::Adaptive),
            "fsr" | "easu" => Some(Self::Fsr),
            "ibp" => Some(Self::Ibp),
            "ewa-lanczos" | "ewalanczos" => Some(Self::EwaLanczos),
            "ewa-lanczossharp" | "ewa_lanczossharp" | "lanczossharp" => Some(Self::EwaLanczosSharp),
            "ewa-robidoux" | "robidoux" => Some(Self::EwaRobidoux),
            "ewa-robidouxsharp" | "robidouxsharp" => Some(Self::EwaRobidouxSharp),
            "ewa-ginseng" | "ginseng" => Some(Self::EwaGinseng),
            "lanczos" | "lanczos3" => Some(Self::Lanczos3),
            "lanczos4" => Some(Self::Lanczos4),
            "ssim-down" | "ssimdown" => Some(Self::SsimDown),
            "ssim-up" | "ssimup" | "ssimsuperres" => Some(Self::SsimUp),
            "icbi" => Some(Self::Icbi),
            "nedi" => Some(Self::Nedi),
            "lineart" | "anime" => Some(Self::LineArt),
            "mmpx" => Some(Self::Mmpx),
            "xbrz" => Some(Self::Xbrz),
            "scalefx" => Some(Self::ScaleFx),
            "mitchell" => Some(Self::Mitchell),
            "catmull" | "catmull-rom" | "catmullrom" | "bicubic" => Some(Self::CatmullRom),
            "hermite" => Some(Self::Hermite),
            "box" | "area" => Some(Self::Box),
            "nearest" | "point" => Some(Self::Nearest),
            _ => None,
        }
    }

    pub fn id(self) -> u32 {
        match self {
            Self::Auto => 0,
            Self::Adaptive => 1,
            Self::Fsr => 2,
            Self::Ibp => 3,
            Self::EwaLanczos => 4,
            Self::EwaLanczosSharp => 5,
            Self::EwaRobidoux => 6,
            Self::EwaRobidouxSharp => 7,
            Self::EwaGinseng => 8,
            Self::Lanczos3 => 9,
            Self::Lanczos4 => 10,
            Self::SsimDown => 11,
            Self::SsimUp => 12,
            Self::Icbi => 13,
            Self::Nedi => 14,
            Self::LineArt => 15,
            Self::Mmpx => 16,
            Self::Xbrz => 17,
            Self::ScaleFx => 18,
            Self::Mitchell => 19,
            Self::CatmullRom => 20,
            Self::Hermite => 21,
            Self::Box => 22,
            Self::Nearest => 23,
        }
    }

    pub fn base_radius(self) -> f64 {
        match self {
            Self::Nearest => 0.5,
            Self::Box => 0.5,
            Self::Hermite => 1.0,
            Self::Mitchell | Self::CatmullRom | Self::EwaRobidoux | Self::EwaRobidouxSharp => 2.0,
            Self::Lanczos3 | Self::EwaLanczos | Self::EwaLanczosSharp | Self::EwaGinseng => 3.0,
            Self::Lanczos4 => 4.0,
            _ => 3.0,
        }
    }
}

#[inline]
pub fn sinc(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 1e-8 {
        return 1.0;
    }
    let px = PI * ax;
    px.sin() / px
}

#[inline]
pub fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    let result = if ax < 8.0 {
        let y = x * x;
        let num = x
            * (72_362_614_232.0
                + y * (-7_895_059_235.0
                    + y * (242_396_853.1
                        + y * (-2_972_611.439
                            + y * (15_704.482_60 + y * (-30.160_366_06))))));
        let den = 144_725_228_442.0
            + y * (2_300_535_178.0
                + y * (18_583_304.74
                    + y * (99_447.433_94 + y * (376.999_139_7 + y))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356_194_491;
        let p = 1.0
            + y * (0.183_105e-2
                + y * (-0.351_639_649_6e-4
                    + y * (0.245_752_017_4e-5 + y * (-0.240_337_019e-6))));
        let q = 0.046_874_999_95
            + y * (-0.200_269_087_3e-3
                + y * (0.844_919_909_6e-5
                    + y * (-0.882_289_87e-6 + y * 0.105_787_412e-6)));
        (std::f64::consts::FRAC_2_PI / ax).sqrt() * (xx.cos() * p - z * xx.sin() * q)
    };
    if x < 0.0 { -result.abs() * x.signum() } else { result }
}

#[inline]
pub fn jinc(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 1e-8 {
        return 1.0;
    }
    let px = PI * ax;
    2.0 * bessel_j1(px) / px
}

#[inline]
pub fn keys_cubic(x: f64, b: f64, c: f64) -> f64 {
    let ax = x.abs();
    if ax < 1.0 {
        let x2 = ax * ax;
        let x3 = x2 * ax;
        ((12.0 - 9.0 * b - 6.0 * c) * x3 + (-18.0 + 12.0 * b + 6.0 * c) * x2 + (6.0 - 2.0 * b))
            / 6.0
    } else if ax < 2.0 {
        let x2 = ax * ax;
        let x3 = x2 * ax;
        ((-b - 6.0 * c) * x3 + (6.0 * b + 30.0 * c) * x2 + (-12.0 * b - 48.0 * c) * ax
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

#[inline]
pub fn eval_separable_kernel(filter: ResampleFilter, dist: f64) -> f64 {
    let ax = dist.abs();
    match filter {
        ResampleFilter::Nearest => {
            if ax < 0.5 { 1.0 } else { 0.0 }
        }
        ResampleFilter::Box => {
            if ax <= 0.5 { 1.0 } else { 0.0 }
        }
        ResampleFilter::Hermite => keys_cubic(ax, 0.0, 0.0),
        ResampleFilter::Mitchell => keys_cubic(ax, 1.0 / 3.0, 1.0 / 3.0),
        ResampleFilter::CatmullRom => keys_cubic(ax, 0.0, 0.5),
        ResampleFilter::EwaRobidoux => keys_cubic(ax, 0.378_215_755, 0.310_892_122),
        ResampleFilter::EwaRobidouxSharp => keys_cubic(ax, 0.262_014_512, 0.368_992_744),
        ResampleFilter::Lanczos4 => {
            if ax >= 4.0 { 0.0 } else { sinc(ax) * sinc(ax / 4.0) }
        }
        _ => {
            if ax >= 3.0 { 0.0 } else { sinc(ax) * sinc(ax / 3.0) }
        }
    }
}

#[inline]
pub fn eval_ewa_kernel(filter: ResampleFilter, r: f64) -> f64 {
    let ar = r.abs();
    match filter {
        ResampleFilter::EwaLanczos => {
            if ar >= 3.238_315_484 {
                0.0
            } else {
                jinc(ar) * jinc(ar / 3.238_315_484)
            }
        }
        ResampleFilter::EwaLanczosSharp => {
            let scaled = ar / 0.981_250_583_722_370_7;
            if scaled >= 3.238_315_484 {
                0.0
            } else {
                jinc(scaled) * jinc(scaled / 3.238_315_484)
            }
        }
        ResampleFilter::EwaGinseng => {
            if ar >= 3.238_315_484 {
                0.0
            } else {
                jinc(ar) * sinc(ar / 3.238_315_484)
            }
        }
        ResampleFilter::EwaRobidoux => keys_cubic(ar, 0.378_215_755, 0.310_892_122),
        ResampleFilter::EwaRobidouxSharp => keys_cubic(ar, 0.262_014_512, 0.368_992_744),
        _ => eval_separable_kernel(filter, ar),
    }
}

pub fn build_axis_plan(src_len: u32, dst_len: u32, filter: ResampleFilter) -> CompiledAxisPlan {
    let scale = src_len as f64 / dst_len as f64;
    let downsample_scale = scale.max(1.0);
    let base_r = filter.base_radius();
    let support = base_r * downsample_scale;

    let mut src_starts = Vec::with_capacity(dst_len as usize);
    let mut tap_counts = Vec::with_capacity(dst_len as usize);
    let mut weight_offsets = Vec::with_capacity(dst_len as usize);
    let mut weights = Vec::with_capacity((dst_len as usize) * (support.ceil() as usize * 2 + 2));

    for dst_i in 0..dst_len {
        let center = (dst_i as f64 + 0.5) * scale - 0.5;
        let mut left = (center - support).ceil() as i64;
        let mut right = (center + support).floor() as i64;
        if left < 0 {
            left = 0;
        }
        if right >= src_len as i64 {
            right = src_len as i64 - 1;
        }
        if right < left {
            let clamped = center.round().clamp(0.0, (src_len - 1) as f64) as i64;
            left = clamped;
            right = clamped;
        }

        let offset = weights.len() as u32;
        let mut sum = 0.0f64;
        let count_start = weights.len();
        for src_i in left..=right {
            let dist = (src_i as f64 - center) / downsample_scale;
            let w = eval_separable_kernel(filter, dist);
            sum += w;
            weights.push(w as f32);
        }
        let taps = weights.len() - count_start;
        if sum.abs() > 1e-12 {
            let inv = (1.0 / sum) as f32;
            for w in &mut weights[count_start..] {
                *w *= inv;
            }
        } else if taps > 0 {
            weights[count_start] = 1.0;
            for w in &mut weights[count_start + 1..] {
                *w = 0.0;
            }
        }
        src_starts.push(left as u32);
        tap_counts.push(taps as u16);
        weight_offsets.push(offset);
    }

    CompiledAxisPlan {
        src_len,
        dst_len,
        src_starts,
        tap_counts,
        weight_offsets,
        weights,
    }
}
