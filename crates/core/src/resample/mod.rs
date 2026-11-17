pub mod easu;
pub mod edge;
pub mod ewa;
pub mod kernels;
pub mod lineart;
pub mod pixelart;
pub mod separable;
pub mod simd;
pub mod ssim;

use crate::arena::PlanCache;
use crate::classify::{ContentKind, classify_image, classify_tiles_8x8};
use crate::color::WorkingSpace;
use crate::error::{EngineError, EngineResult};
use crate::pixel::PlanarF32;
use crate::pool::ThreadPool;
use kernels::ResampleFilter;

#[derive(Debug, Clone)]
pub struct ResampleConfig {
    pub filter: ResampleFilter,
    pub antiring: f32,
    pub rcas_sharpness: f32,
    pub ibp_iters: u8,
    pub sigmoidal_strength: Option<f32>,
    pub working_space: WorkingSpace,
    pub use_krig_chroma: bool,
    pub premultiply: bool,
    pub fix_alpha_bleed: bool,
}

impl Default for ResampleConfig {
    fn default() -> Self {
        Self {
            filter: ResampleFilter::Auto,
            antiring: 0.65,
            rcas_sharpness: 0.35,
            ibp_iters: 2,
            sigmoidal_strength: None,
            working_space: WorkingSpace::LinearSrgb,
            use_krig_chroma: false,
            premultiply: true,
            fix_alpha_bleed: true,
        }
    }
}

pub fn resample_planar(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    cfg: &ResampleConfig,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    if dst_w == 0 || dst_h == 0 {
        return Err(EngineError::InvalidDimensions {
            width: dst_w,
            height: dst_h,
        });
    }

    let mut work = src.clone();
    if work.has_alpha {
        if work.premultiplied {
            work.unpremultiply();
        }
        if cfg.fix_alpha_bleed {
            work.fix_transparent_bleeding(0.01);
        }
    }

    let target_space = if let Some(sig) = cfg.sigmoidal_strength {
        WorkingSpace::Sigmoidal(sig)
    } else {
        cfg.working_space
    };
    work.convert_working_space(target_space);
    if cfg.premultiply && work.has_alpha {
        work.premultiply();
    }

    if work.width == dst_w && work.height == dst_h {
        if work.premultiplied {
            work.unpremultiply();
        }
        work.convert_working_space(WorkingSpace::LinearSrgb);
        return Ok(work);
    }

    let is_upscale = dst_w > work.width || dst_h > work.height;
    let chosen = match cfg.filter {
        ResampleFilter::Auto => {
            let profile = classify_image(&work);
            match (profile.kind, is_upscale) {
                (ContentKind::PixelArt, _) => {
                    if dst_w.is_multiple_of(work.width) && dst_h.is_multiple_of(work.height) {
                        ResampleFilter::Mmpx
                    } else {
                        ResampleFilter::ScaleFx
                    }
                }
                (ContentKind::LineArt, true) => ResampleFilter::LineArt,
                (ContentKind::UiText, true) => ResampleFilter::Fsr,
                (ContentKind::Photo, true) => ResampleFilter::Fsr,
                (_, false) => ResampleFilter::SsimDown,
            }
        }
        other => other,
    };

    let mut out = match chosen {
        ResampleFilter::Adaptive => {
            resample_adaptive_hybrid(&work, dst_w, dst_h, cfg, plan_cache, pool)?
        }
        ResampleFilter::Fsr => {
            let base = easu::upscale_fsr_easu_rcas(&work, dst_w, dst_h, cfg.rcas_sharpness, pool)?;
            if cfg.use_krig_chroma && is_upscale {
                ssim::krig_bilateral_chroma(&base, 1.0, 0.08, pool)?
            } else {
                base
            }
        }
        ResampleFilter::Ibp => {
            ssim::irani_peleg_ibp(&work, dst_w, dst_h, cfg.ibp_iters, plan_cache, pool)?
        }
        ResampleFilter::EwaLanczos
        | ResampleFilter::EwaLanczosSharp
        | ResampleFilter::EwaRobidoux
        | ResampleFilter::EwaRobidouxSharp
        | ResampleFilter::EwaGinseng => {
            ewa::resample_ewa_planar(&work, dst_w, dst_h, chosen, cfg.antiring, pool)?
        }
        ResampleFilter::SsimDown => {
            ssim::ssim_downscale(&work, dst_w, dst_h, 0.55, plan_cache, pool)?
        }
        ResampleFilter::SsimUp => {
            let base = ewa::resample_ewa_planar(
                &work,
                dst_w,
                dst_h,
                ResampleFilter::EwaLanczosSharp,
                cfg.antiring,
                pool,
            )?;
            ssim::ssim_super_res(&work, &base, 0.8, plan_cache, pool)?
        }
        ResampleFilter::Icbi => edge::upscale_icbi(&work, dst_w, dst_h, 3, pool)?,
        ResampleFilter::Nedi => edge::upscale_nedi(&work, dst_w, dst_h, pool)?,
        ResampleFilter::LineArt => lineart::upscale_lineart(&work, dst_w, dst_h, 0.85, pool)?,
        ResampleFilter::Mmpx => pixelart::upscale_mmpx(&work, dst_w, dst_h, pool)?,
        ResampleFilter::Xbrz => pixelart::upscale_xbrz_or_scalefx(&work, dst_w, dst_h, false, pool)?,
        ResampleFilter::ScaleFx => pixelart::upscale_xbrz_or_scalefx(&work, dst_w, dst_h, true, pool)?,
        ResampleFilter::Nearest => pixelart::nearest_resize(&work, dst_w, dst_h, pool)?,
        ResampleFilter::Lanczos3
        | ResampleFilter::Lanczos4
        | ResampleFilter::Mitchell
        | ResampleFilter::CatmullRom
        | ResampleFilter::Hermite
        | ResampleFilter::Box
        | ResampleFilter::Auto => {
            separable::resample_separable(&work, dst_w, dst_h, chosen, cfg.antiring, plan_cache, pool)?
        }
    };

    if out.premultiplied {
        out.unpremultiply();
    }
    out.convert_working_space(WorkingSpace::LinearSrgb);
    Ok(out)
}

fn resample_adaptive_hybrid(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    cfg: &ResampleConfig,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    if dst_w <= src.width && dst_h <= src.height {
        return ssim::ssim_downscale(src, dst_w, dst_h, 0.5, plan_cache, pool);
    }

    let sharp = easu::upscale_fsr_easu_rcas(src, dst_w, dst_h, cfg.rcas_sharpness, pool)?;
    let smooth = ewa::resample_ewa_planar(src, dst_w, dst_h, ResampleFilter::EwaRobidoux, 0.7, pool)?;

    let (tx_count, ty_count, tiles) = classify_tiles_8x8(src);
    let dw = dst_w as usize;
    let dh = dst_h as usize;
    let mut out = sharp.clone();

    pool.parallel_row_stripes_4_mut(
        &mut out.r,
        &mut out.g,
        &mut out.b,
        &mut Vec::new(),
        dw,
        |y0, sr, sg, sb, _| {
            let rows = sr.len() / dw;
            for ly in 0..rows {
                let y = y0 + ly;
                let ty = ((y * src.height as usize) / dh / 8).min(ty_count.saturating_sub(1));
                for x in 0..dw {
                    let tx = ((x * src.width as usize) / dw / 8).min(tx_count.saturating_sub(1));
                    let metric = tiles[ty * tx_count + tx];
                    let alpha = (metric.edge_strength * 14.0).clamp(0.0, 1.0);
                    let idx = y * dw + x;
                    let oi = ly * dw + x;
                    sr[oi] = smooth.r[idx] + (sharp.r[idx] - smooth.r[idx]) * alpha;
                    sg[oi] = smooth.g[idx] + (sharp.g[idx] - smooth.g[idx]) * alpha;
                    sb[oi] = smooth.b[idx] + (sharp.b[idx] - smooth.b[idx]) * alpha;
                }
            }
        },
    );

    Ok(out)
}
