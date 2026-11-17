use crate::arena::{MemoryBudget, PlanCache};
use crate::classify::{compute_dhash_64, fnv1a_128};
use crate::codec::anim::{decode_animation, encode_animation};
use crate::codec::metadata::ContainerMetadata;
use crate::codec::{DecodeOptions, EncodeOptions, ImageFormat, decode_image, detect_format, encode_image};
use crate::color::{WorkingSpace, global_luts};
use crate::error::{EngineError, EngineResult};
use crate::geometry::fit::{FitConfig, extract_oklab_palette, fit_planar};
use crate::geometry::spec::{ResolvedDimensions, TargetGeometry};
use crate::geometry::trim::{crop_planar, detect_solid_borders};
use crate::inspect::metrics::compare_images;
use crate::inspect::resdet::descale_if_suspected;
use crate::pixel::{Frame, FrameSequence, PixelFormat, PlanarF32};
use crate::pool::ThreadPool;
use crate::resample::kernels::ResampleFilter;
use crate::resample::{ResampleConfig, resample_planar};
use crate::restore::deband::deband_stochastic;
use crate::restore::deblock::deblock_shifted_dct;
use crate::restore::dither::{DitherMode, apply_dither_and_grain};
use crate::restore::guided::guided_filter_denoise;

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub targets: Vec<TargetGeometry>,
    pub fit: FitConfig,
    pub resample: ResampleConfig,
    pub decode: DecodeOptions,
    pub encode: EncodeOptions,
    pub output_format: Option<ImageFormat>,
    pub trim_borders: bool,
    pub auto_descale: bool,
    pub descale_min_confidence: f32,
    pub denoise_strength: f32,
    pub dejpeg_strength: f32,
    pub deband_strength: f32,
    pub dither_mode: DitherMode,
    pub grain_amount: f32,
    pub target_ssim2: Option<f64>,
    pub skip_if_fits: bool,
    pub extract_palette: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            targets: Vec::new(),
            fit: FitConfig::default(),
            resample: ResampleConfig::default(),
            decode: DecodeOptions::default(),
            encode: EncodeOptions::default(),
            output_format: None,
            trim_borders: false,
            auto_descale: false,
            descale_min_confidence: 0.48,
            denoise_strength: 0.0,
            dejpeg_strength: 0.0,
            deband_strength: 0.0,
            dither_mode: DitherMode::None,
            grain_amount: 0.0,
            target_ssim2: None,
            skip_if_fits: false,
            extract_palette: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProcessedImageOutput {
    pub bytes: Vec<u8>,
    pub format: ImageFormat,
    pub src_width: u32,
    pub src_height: u32,
    pub dst_width: u32,
    pub dst_height: u32,
    pub skipped: bool,
    pub was_descaled: bool,
    pub dhash_64: u64,
    pub exact_hash_128: u128,
    pub palette: Vec<[u8; 3]>,
    pub estimated_ssim2: Option<f64>,
}

pub fn process_image_bytes(
    input_bytes: &[u8],
    ext_hint: Option<&str>,
    cfg: &PipelineConfig,
    budget: &MemoryBudget,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<Vec<ProcessedImageOutput>> {
    let exact_hash_128 = fnv1a_128(input_bytes);
    let sniffed = detect_format(input_bytes).or_else(|| ext_hint.and_then(ImageFormat::from_extension));
    let src_fmt = sniffed.ok_or_else(|| {
        EngineError::UnsupportedFormat("unable to determine input image format".to_string())
    })?;

    let decoded = decode_image(input_bytes, ext_hint, &cfg.decode)?;
    let est_bytes = (decoded.image.width as usize)
        .saturating_mul(decoded.image.height as usize)
        .saturating_mul(16);
    let _lease = budget.try_reserve(est_bytes)?;

    if decoded.is_animated && decoded.frame_count > 1 {
        return process_animated_sequence(
            input_bytes,
            src_fmt,
            decoded.image.format,
            &decoded.metadata,
            exact_hash_128,
            cfg,
            plan_cache,
            pool,
        );
    }

    let mut base = PlanarF32::from_packed(&decoded.image, WorkingSpace::LinearSrgb, global_luts())?;
    let src_width = base.width;
    let src_height = base.height;

    let mut was_descaled = false;
    if cfg.trim_borders {
        let rect = detect_solid_borders(&base, 0.015);
        if rect.width != base.width || rect.height != base.height {
            base = crop_planar(&base, rect)?;
        }
    }

    if cfg.auto_descale {
        let (clean, _, descaled) =
            descale_if_suspected(&base, cfg.descale_min_confidence, plan_cache, pool)?;
        if descaled {
            base = clean;
            was_descaled = true;
        }
    }

    if cfg.dejpeg_strength > 0.005 {
        let s = cfg.dejpeg_strength.clamp(0.0, 1.0);
        base = deblock_shifted_dct(&base, s, false)?;
    }

    if cfg.denoise_strength > 0.005 {
        let s = cfg.denoise_strength.clamp(0.0, 1.0);
        let radius = if s > 0.65 { 3 } else { 2 };
        let eps = 0.002 + s * 0.035;
        base = guided_filter_denoise(&base, radius, eps, s * 0.75)?;
    }

    let out_fmt = cfg
        .output_format
        .unwrap_or_else(|| src_fmt.default_output_format(cfg.encode.lossless));

    let targets: Vec<Option<TargetGeometry>> = if cfg.targets.is_empty() {
        vec![None]
    } else {
        cfg.targets.iter().copied().map(Some).collect()
    };

    let dhash_64 = compute_dhash_64(&base);
    let mut results = Vec::with_capacity(targets.len());

    for tgt in targets {
        let resolved = match tgt {
            Some(geom) => geom.resolve(src_width, src_height)?,
            None => ResolvedDimensions {
                width: base.width,
                height: base.height,
                skipped: false,
                exact_stretch: false,
                cover_mode: false,
            },
        };

        let no_filters = !cfg.trim_borders
            && !was_descaled
            && cfg.dejpeg_strength <= 0.005
            && cfg.denoise_strength <= 0.005
            && cfg.deband_strength <= 0.005
            && cfg.dither_mode == DitherMode::None
            && cfg.grain_amount <= 0.005;

        if (resolved.skipped && cfg.skip_if_fits)
            || (cfg.skip_if_fits
                && resolved.width == src_width
                && resolved.height == src_height
                && out_fmt == src_fmt
                && !cfg.encode.lossless
                && !cfg.encode.optimize
                && no_filters)
        {
            results.push(ProcessedImageOutput {
                bytes: input_bytes.to_vec(),
                format: src_fmt,
                src_width,
                src_height,
                dst_width: src_width,
                dst_height: src_height,
                skipped: true,
                was_descaled: false,
                dhash_64,
                exact_hash_128,
                palette: if cfg.extract_palette > 0 {
                    extract_oklab_palette(&base, cfg.extract_palette)
                } else {
                    Vec::new()
                },
                estimated_ssim2: Some(100.0),
            });
            continue;
        }

        let mut out_planar = transform_planar_to_target(
            &base,
            resolved,
            out_fmt,
            decoded.image.format,
            cfg,
            plan_cache,
            pool,
        )?;

        let palette = if cfg.extract_palette > 0 {
            extract_oklab_palette(&out_planar, cfg.extract_palette)
        } else {
            Vec::new()
        };

        let target_px_fmt = choose_output_pixel_format(decoded.image.format, out_planar.has_alpha, out_fmt);
        let packed_out = out_planar.to_packed(target_px_fmt, global_luts())?;

        let (bytes, est_ssim2) = encode_with_optional_ssim2_search(
            &packed_out,
            &mut out_planar,
            out_fmt,
            cfg,
            &decoded.metadata,
        )?;

        results.push(ProcessedImageOutput {
            bytes,
            format: out_fmt,
            src_width,
            src_height,
            dst_width: packed_out.width,
            dst_height: packed_out.height,
            skipped: false,
            was_descaled,
            dhash_64,
            exact_hash_128,
            palette,
            estimated_ssim2: est_ssim2,
        });
    }

    Ok(results)
}

#[allow(clippy::too_many_arguments)]
fn process_animated_sequence(
    input_bytes: &[u8],
    src_fmt: ImageFormat,
    src_px_fmt: PixelFormat,
    metadata: &ContainerMetadata,
    exact_hash_128: u128,
    cfg: &PipelineConfig,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<Vec<ProcessedImageOutput>> {
    let seq = decode_animation(input_bytes, src_fmt, cfg.decode.max_pixels)?;
    if seq.frames.is_empty() {
        return Err(EngineError::DecodeFailed(
            "animated image contains no frames".to_string(),
        ));
    }
    let out_fmt = match cfg
        .output_format
        .unwrap_or_else(|| src_fmt.default_output_format(cfg.encode.lossless))
    {
        ImageFormat::Gif => ImageFormat::Gif,
        ImageFormat::Png => ImageFormat::Png,
        _ => ImageFormat::Gif,
    };

    let targets: Vec<Option<TargetGeometry>> = if cfg.targets.is_empty() {
        vec![None]
    } else {
        cfg.targets.iter().copied().map(Some).collect()
    };

    let mut out_list = Vec::with_capacity(targets.len());
    for tgt in targets {
        let resolved = match tgt {
            Some(geom) => geom.resolve(seq.canvas_width, seq.canvas_height)?,
            None => ResolvedDimensions {
                width: seq.canvas_width,
                height: seq.canvas_height,
                skipped: false,
                exact_stretch: false,
                cover_mode: false,
            },
        };

        if resolved.skipped && cfg.skip_if_fits {
            let first_p =
                PlanarF32::from_packed(&seq.frames[0].image, WorkingSpace::LinearSrgb, global_luts())?;
            out_list.push(ProcessedImageOutput {
                bytes: input_bytes.to_vec(),
                format: src_fmt,
                src_width: seq.canvas_width,
                src_height: seq.canvas_height,
                dst_width: seq.canvas_width,
                dst_height: seq.canvas_height,
                skipped: true,
                was_descaled: false,
                dhash_64: compute_dhash_64(&first_p),
                exact_hash_128,
                palette: Vec::new(),
                estimated_ssim2: Some(100.0),
            });
            continue;
        }

        let mut out_frames = Vec::with_capacity(seq.frames.len());
        let mut first_hash = 0u64;
        let mut palette = Vec::new();

        for (idx, frame) in seq.frames.iter().enumerate() {
            let mut planar =
                PlanarF32::from_packed(&frame.image, WorkingSpace::LinearSrgb, global_luts())?;
            if cfg.denoise_strength > 0.005 {
                planar = guided_filter_denoise(
                    &planar,
                    2,
                    0.002 + cfg.denoise_strength * 0.025,
                    cfg.denoise_strength * 0.5,
                )?;
            }
            let out_p = transform_planar_to_target(
                &planar,
                resolved,
                out_fmt,
                src_px_fmt,
                cfg,
                plan_cache,
                pool,
            )?;
            if idx == 0 {
                first_hash = compute_dhash_64(&out_p);
                if cfg.extract_palette > 0 {
                    palette = extract_oklab_palette(&out_p, cfg.extract_palette);
                }
            }
            let packed = out_p.to_packed(PixelFormat::Rgba8, global_luts())?;
            out_frames.push(Frame {
                image: packed,
                meta: frame.meta,
            });
        }

        let out_seq = FrameSequence {
            canvas_width: resolved.width,
            canvas_height: resolved.height,
            loop_count: seq.loop_count,
            frames: out_frames,
        };
        let bytes = encode_animation(&out_seq, out_fmt)?;
        let _ = metadata;

        out_list.push(ProcessedImageOutput {
            bytes,
            format: out_fmt,
            src_width: seq.canvas_width,
            src_height: seq.canvas_height,
            dst_width: resolved.width,
            dst_height: resolved.height,
            skipped: false,
            was_descaled: false,
            dhash_64: first_hash,
            exact_hash_128,
            palette,
            estimated_ssim2: None,
        });
    }

    Ok(out_list)
}

fn transform_planar_to_target(
    base: &PlanarF32,
    resolved: ResolvedDimensions,
    out_fmt: ImageFormat,
    src_px_fmt: PixelFormat,
    cfg: &PipelineConfig,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    let mut scaled = if resolved.exact_stretch {
        let mut exact_fit = cfg.fit.clone();
        exact_fit.mode = crate::geometry::fit::FitMode::Stretch;
        fit_planar(
            base,
            resolved.width,
            resolved.height,
            &exact_fit,
            &cfg.resample,
            plan_cache,
            pool,
        )?
    } else if base.width == resolved.width && base.height == resolved.height {
        base.clone()
    } else {
        let src_ar = base.width as f64 / base.height as f64;
        let dst_ar = resolved.width as f64 / resolved.height as f64;
        let same_ar = (src_ar / dst_ar - 1.0).abs() < 0.015;
        if same_ar && cfg.fit.foreground_dim <= 0.001 && cfg.fit.vignette <= 0.001 {
            resample_planar(
                base,
                resolved.width,
                resolved.height,
                &cfg.resample,
                plan_cache,
                pool,
            )?
        } else {
            fit_planar(
                base,
                resolved.width,
                resolved.height,
                &cfg.fit,
                &cfg.resample,
                plan_cache,
                pool,
            )?
        }
    };

    if scaled.width > base.width
        && scaled.height > base.height
        && cfg.resample.rcas_sharpness > 0.01
        && matches!(
            cfg.resample.filter,
            ResampleFilter::Auto | ResampleFilter::EwaLanczos | ResampleFilter::EwaLanczosSharp
        )
    {
        scaled = crate::resample::easu::rcas_pass(
            &scaled,
            cfg.resample.rcas_sharpness * 0.5,
            pool,
        )?;
    }

    if cfg.deband_strength > 0.005 {
        let s = cfg.deband_strength.clamp(0.0, 1.0);
        let range = 10.0 + s * 16.0;
        let iters = if s > 0.55 { 2 } else { 1 };
        scaled = deband_stochastic(&scaled, s * 0.75, iters, range, pool)?;
    }

    if cfg.dither_mode != DitherMode::None || cfg.grain_amount > 0.005 {
        let target_px = choose_output_pixel_format(src_px_fmt, scaled.has_alpha, out_fmt);
        let levels = if target_px.is_16bit() { 65535.0 } else { 255.0 };
        let grain = (cfg.grain_amount * 0.04).clamp(0.0, 0.08);
        scaled = apply_dither_and_grain(&scaled, cfg.dither_mode, grain, levels)?;
    }

    Ok(scaled)
}

fn choose_output_pixel_format(src_px: PixelFormat, has_alpha: bool, out_fmt: ImageFormat) -> PixelFormat {
    match out_fmt {
        ImageFormat::Png | ImageFormat::Tiff | ImageFormat::Farbfeld if src_px.is_16bit() => {
            if has_alpha {
                PixelFormat::Rgba16
            } else {
                PixelFormat::Rgb16
            }
        }
        ImageFormat::Exr | ImageFormat::Hdr => {
            if has_alpha && out_fmt == ImageFormat::Exr {
                PixelFormat::RgbaF32
            } else {
                PixelFormat::RgbF32
            }
        }
        _ => {
            if has_alpha {
                PixelFormat::Rgba8
            } else {
                PixelFormat::Rgb8
            }
        }
    }
}

fn encode_with_optional_ssim2_search(
    packed: &crate::pixel::PackedImage,
    ref_planar: &mut PlanarF32,
    out_fmt: ImageFormat,
    cfg: &PipelineConfig,
    metadata: &ContainerMetadata,
) -> EngineResult<(Vec<u8>, Option<f64>)> {
    if let Some(target_score) = cfg.target_ssim2
        && !cfg.encode.lossless
        && matches!(out_fmt, ImageFormat::Jpeg | ImageFormat::Webp)
    {
        if out_fmt == ImageFormat::Jpeg && !cfg.encode.lossless {
            ref_planar.convert_working_space(WorkingSpace::LinearSrgb);
            let dec_opts = DecodeOptions {
                max_pixels: cfg.decode.max_pixels,
                auto_orient: false,
                convert_icc_to_srgb: false,
                target_svg_size: None,
            };

            let mut lo = 35u8;
            let mut hi = 98u8;
            let mut best_bytes: Option<Vec<u8>> = None;
            let mut best_score = 0.0f64;

            for _ in 0..7 {
                if lo > hi {
                    break;
                }
                let mid = lo + (hi - lo) / 2;
                let trial_opts = EncodeOptions {
                    quality: mid,
                    max_bytes: None,
                    ..cfg.encode.clone()
                };
                let encoded = encode_image(packed, out_fmt, &trial_opts, Some(metadata))?;
                let dec = decode_image(&encoded, Some(out_fmt.canonical_extension()), &dec_opts)?;
                let cmp_planar =
                    PlanarF32::from_packed(&dec.image, WorkingSpace::LinearSrgb, global_luts())?;
                let metrics = compare_images(ref_planar, &cmp_planar)?;
                let under_bytes = cfg
                    .encode
                    .max_bytes
                    .is_none_or(|cap| (encoded.len() as u64) <= cap);

                if metrics.ssim2_est >= target_score && under_bytes {
                    best_score = metrics.ssim2_est;
                    best_bytes = Some(encoded);
                    if mid == 0 {
                        break;
                    }
                    hi = mid.saturating_sub(1);
                } else if !under_bytes {
                    if mid == 0 {
                        break;
                    }
                    hi = mid.saturating_sub(1);
                } else {
                    lo = mid.saturating_add(1);
                }
            }

            if let Some(bytes) = best_bytes {
                return Ok((bytes, Some(best_score)));
            }
        } else {
            let bytes = encode_image(packed, out_fmt, &cfg.encode, Some(metadata))?;
            return Ok((bytes, None));
        }
    }

    let bytes = encode_image(packed, out_fmt, &cfg.encode, Some(metadata))?;
    Ok((bytes, None))
}
