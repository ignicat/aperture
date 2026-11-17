pub mod archive;
pub mod arena;
pub mod classify;
pub mod codec;
pub mod color;
pub mod error;
pub mod geometry;
pub mod inspect;
pub mod pipeline;
pub mod pixel;
pub mod pool;
pub mod resample;
pub mod restore;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::arena::{
        CompiledAxisPlan, CompiledResamplePlan, MemoryBudget, PlanCache, ResamplePlanKey,
        WorkerArena,
    };
    use super::codec::metadata::{ContainerMetadata, MetadataPolicy, read_exif_orientation};
    use super::codec::{DecodeOptions, EncodeOptions, ImageFormat, decode_image, encode_image};
    use super::color::{
        ColorSpace, WorkingSpace, global_luts, linear_rgb_to_cie_lab, linear_rgb_to_oklab,
        linear_rgb_to_xyb, linear_to_srgb_f32, oklab_to_linear_rgb,
        oklab_to_linear_rgb_gamut_clipped, sigmoidal_forward, sigmoidal_inverse,
        srgb_to_linear_f32,
    };
    use super::geometry::spec::TargetGeometry;
    use super::pixel::{PackedImage, PixelFormat, PlanarF32, StripWindow};
    use super::pool::ThreadPool;

    #[test]
    fn srgb_and_sigmoidal_roundtrip() {
        let luts = global_luts();
        for v in 0..=255u8 {
            let lin = luts.u8_to_linear[v as usize];
            let idx = (lin * 65535.0).round() as usize;
            assert_eq!(luts.u16_linear_to_u8[idx], v);
        }
        for i in 0..=20 {
            let x = i as f32 / 20.0;
            let back = linear_to_srgb_f32(srgb_to_linear_f32(x));
            assert!((back - x).abs() < 1e-5);
            let sig = sigmoidal_inverse(sigmoidal_forward(x, 6.5), 6.5);
            assert!((sig - x).abs() < 1e-4);
        }
    }

    #[test]
    fn oklab_roundtrip_and_gamut_clip() {
        let (l, a, b) = linear_rgb_to_oklab(0.7, 0.2, 0.4);
        let (r, g, bl) = oklab_to_linear_rgb(l, a, b);
        assert!((r - 0.7).abs() < 1e-4);
        assert!((g - 0.2).abs() < 1e-4);
        assert!((bl - 0.4).abs() < 1e-4);

        let (cr, cg, cb) = oklab_to_linear_rgb_gamut_clipped(0.6, 0.35, 0.35);
        assert!((0.0..=1.0).contains(&cr));
        assert!((0.0..=1.0).contains(&cg));
        assert!((0.0..=1.0).contains(&cb));

        let (lab_l, _, _) = linear_rgb_to_cie_lab(1.0, 1.0, 1.0);
        assert!((lab_l - 100.0).abs() < 0.1);
        let (x, y, zb) = linear_rgb_to_xyb(0.5, 0.5, 0.5);
        assert!(x.abs() < 1e-4);
        assert!(y > 0.0 && zb > 0.0);
    }

    #[test]
    fn alpha_bleed_fix_and_premultiply() {
        let data = vec![
            255, 0, 0, 255,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 255, 255,
        ];
        let packed = PackedImage::new(2, 2, PixelFormat::Rgba8, ColorSpace::Srgb, data);
        assert!(packed.is_ok());
        if let Ok(img) = packed {
            let luts = global_luts();
            let planar = PlanarF32::from_packed(&img, WorkingSpace::LinearSrgb, luts);
            assert!(planar.is_ok());
            if let Ok(mut p) = planar {
                p.fix_transparent_bleeding(0.01);
                assert!(p.r[1] > 0.9 || p.b[1] > 0.9);
                assert_eq!(p.a[1], 0.0);

                p.premultiply();
                assert!(p.premultiplied);
                assert_eq!(p.r[1], 0.0);
                p.unpremultiply();
                assert!(!p.premultiplied);
                assert!((p.r[0] - 1.0).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn memory_budget_arena_pool_and_strip_window() {
        let budget = MemoryBudget::new(1024);
        let lease1 = budget.try_reserve(600);
        assert!(lease1.is_ok());
        let lease2 = budget.try_reserve(600);
        assert!(lease2.is_err());
        drop(lease1);
        assert_eq!(budget.used_bytes(), 0);
        assert!(budget.try_reserve(800).is_ok());

        let mut arena = WorkerArena::new(4096);
        let buf_len = arena.ensure_u8(512).len();
        assert_eq!(buf_len, 512);
        let (a, _, _, _) = arena.ensure_f32_planes(64);
        assert_eq!(a.len(), 64);

        let cache = PlanCache::new(4);
        let key = ResamplePlanKey {
            src_w: 10,
            src_h: 10,
            dst_w: 20,
            dst_h: 20,
            kernel_id: 1,
            quantized_param: 0,
        };
        let _ = cache.get_or_build(key, || CompiledResamplePlan {
            horiz: CompiledAxisPlan {
                src_len: 10,
                dst_len: 20,
                src_starts: vec![0; 20],
                tap_counts: vec![2; 20],
                weight_offsets: vec![0; 20],
                weights: vec![0.5; 40],
            },
            vert: CompiledAxisPlan {
                src_len: 10,
                dst_len: 20,
                src_starts: vec![0; 20],
                tap_counts: vec![2; 20],
                weight_offsets: vec![0; 20],
                weights: vec![0.5; 40],
            },
        });
        assert_eq!(cache.len(), 1);

        let pool = ThreadPool::new(4);
        let nums: Vec<usize> = (0..100).collect();
        let doubled = pool.parallel_map(&nums, |_, &x| x * 2);
        assert_eq!(doubled[42], 84);

        if let Ok(mut win) = StripWindow::new(4, 5, 3, false) {
            for r in 0..5usize {
                let row = [r as f32; 4];
                assert!(win.push_row(&row, &row, &row, None).is_ok());
            }
            if let Some(row) = win.row_slices_clamped(4) {
                assert_eq!(row.r[0], 4.0);
            }
            if let Some(row) = win.row_slices_clamped(0) {
                assert_eq!(row.r[0], 2.0);
            }
        }
    }

    #[test]
    fn target_geometry_dsl_parsing_and_resolution() {
        let cyr = TargetGeometry::parse("1920х1080");
        assert!(cyr.is_ok());
        if let Ok(spec) = cyr
            && let Ok(res) = spec.resolve(1280, 720)
        {
            assert_eq!((res.width, res.height, res.skipped), (1920, 1080, false));
        }

        if let Ok(shrink) = TargetGeometry::parse("1920x1080>")
            && let Ok(res) = shrink.resolve(1280, 720)
        {
            assert!(res.skipped);
        }

        if let Ok(uw) = TargetGeometry::parse("uwqhd")
            && let Ok(res) = uw.resolve(1920, 1080)
        {
            assert_eq!((res.width, res.height), (3440, 1440));
        }

        if let Ok(list) = TargetGeometry::parse_list("4k, 1920w, 2x, ar:21:9, mp:8") {
            assert_eq!(list.len(), 5);
        } else {
            assert_eq!(0, 1);
        }
    }

    #[test]
    fn codec_roundtrips_and_exif_orientation() {
        let w = 16u32;
        let h = 12u32;
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                rgba.push((x * 16) as u8);
                rgba.push((y * 20) as u8);
                rgba.push(((x + y) * 8) as u8);
                rgba.push(if x < 4 { 200 } else { 255 });
            }
        }
        let img = PackedImage::new(w, h, PixelFormat::Rgba8, ColorSpace::Srgb, rgba);
        assert!(img.is_ok());
        let Ok(img) = img else {
            return;
        };

        let formats = [
            ImageFormat::Png,
            ImageFormat::Webp,
            ImageFormat::Gif,
            ImageFormat::Tiff,
            ImageFormat::Bmp,
            ImageFormat::Tga,
            ImageFormat::Pnm,
            ImageFormat::Qoi,
            ImageFormat::Farbfeld,
            ImageFormat::Hdr,
            ImageFormat::Exr,
            ImageFormat::Ico,
        ];
        let dec_opts = DecodeOptions::default();
        let enc_opts = EncodeOptions::default();
        for fmt in formats {
            let encoded = encode_image(&img, fmt, &enc_opts, None);
            assert!(encoded.is_ok(), "encode failed for {fmt:?}");
            if let Ok(bytes) = encoded {
                let decoded = decode_image(&bytes, Some(fmt.canonical_extension()), &dec_opts);
                assert!(decoded.is_ok(), "decode failed for {fmt:?}");
                if let Ok(d) = decoded
                    && fmt != ImageFormat::Ico
                {
                    assert_eq!((d.image.width, d.image.height), (w, h));
                }
            }
        }

        let exif_tiff = b"Exif\0\0II\x2A\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0".to_vec();
        assert_eq!(read_exif_orientation(&exif_tiff), 6);
        let jpg_opts = EncodeOptions {
            quality: 95,
            ..Default::default()
        };
        let encoded_plain = encode_image(&img, ImageFormat::Jpeg, &jpg_opts, None);
        assert!(encoded_plain.is_ok());
        if let Ok(plain_bytes) = encoded_plain {
            use img_parts::ImageEXIF;
            use img_parts::jpeg::Jpeg;
            let parsed = Jpeg::from_bytes(img_parts::Bytes::from(plain_bytes));
            assert!(parsed.is_ok());
            if let Ok(mut jpeg_part) = parsed {
                jpeg_part.set_exif(Some(img_parts::Bytes::copy_from_slice(&exif_tiff)));
                let mut raw_with_orient6 = Vec::new();
                assert!(jpeg_part.encoder().write_to(&mut raw_with_orient6).is_ok());
                let decoded = decode_image(&raw_with_orient6, Some("jpg"), &dec_opts);
                assert!(decoded.is_ok());
                if let Ok(d) = decoded {
                    assert_eq!((d.image.width, d.image.height), (h, w));
                    let re_encoded = encode_image(&d.image, ImageFormat::Jpeg, &jpg_opts, Some(&d.metadata));
                    assert!(re_encoded.is_ok());
                    if let Ok(re_bytes) = re_encoded {
                        let re_dec = decode_image(&re_bytes, Some("jpg"), &dec_opts);
                        assert!(re_dec.is_ok());
                        if let Ok(rd) = re_dec
                            && let Some(ex) = rd.metadata.exif
                        {
                            assert_eq!(read_exif_orientation(&ex), 1);
                        }
                    }
                }
            }
        }

        let jpg_meta = ContainerMetadata {
            icc_profile: None,
            exif: Some(exif_tiff),
            xmp: None,
            orientation: 6,
        };

        let quant_opts = EncodeOptions {
            png_quant_colors: Some(16),
            ..Default::default()
        };
        assert!(encode_image(&img, ImageFormat::Png, &quant_opts, None).is_ok());

        let strip_opts = EncodeOptions {
            meta_policy: MetadataPolicy::StripPrivacy,
            max_bytes: Some(4096),
            ..Default::default()
        };
        let small_jpg = encode_image(&img, ImageFormat::Jpeg, &strip_opts, Some(&jpg_meta));
        assert!(small_jpg.is_ok());
        if let Ok(jb) = small_jpg {
            assert!(jb.len() <= 4096);
        }
    }

    #[test]
    fn resampling_algorithms_and_scalar_simd_parity() {
        use super::classify::{compute_dhash_64, dhash_distance};
        use super::resample::kernels::{ResampleFilter, build_axis_plan};
        use super::resample::simd::vertical_row_scalar;
        use super::resample::{ResampleConfig, resample_planar};

        let w = 24u32;
        let h = 18u32;
        let mut img = PlanarF32::new(w, h, true, WorkingSpace::LinearSrgb).ok();
        assert!(img.is_some());
        let Some(ref mut src) = img else {
            return;
        };
        for y in 0..h as usize {
            for x in 0..w as usize {
                let idx = y * w as usize + x;
                let edge = if x > y + 2 { 0.85 } else { 0.15 };
                src.r[idx] = edge;
                src.g[idx] = (y as f32 / h as f32) * 0.8 + 0.1;
                src.b[idx] = (x as f32 / w as f32) * 0.8 + 0.1;
                src.a[idx] = if x < 4 { 0.5 } else { 1.0 };
            }
        }

        let cache = PlanCache::new(32);
        let pool = ThreadPool::new(4);

        let algos = [
            ResampleFilter::Auto,
            ResampleFilter::Adaptive,
            ResampleFilter::Fsr,
            ResampleFilter::Ibp,
            ResampleFilter::EwaLanczos,
            ResampleFilter::EwaLanczosSharp,
            ResampleFilter::EwaRobidoux,
            ResampleFilter::EwaGinseng,
            ResampleFilter::Lanczos3,
            ResampleFilter::Lanczos4,
            ResampleFilter::SsimUp,
            ResampleFilter::Icbi,
            ResampleFilter::Nedi,
            ResampleFilter::LineArt,
            ResampleFilter::Mmpx,
            ResampleFilter::Xbrz,
            ResampleFilter::ScaleFx,
            ResampleFilter::Mitchell,
            ResampleFilter::CatmullRom,
            ResampleFilter::Nearest,
        ];

        for &algo in &algos {
            let cfg = ResampleConfig {
                filter: algo,
                ..Default::default()
            };
            let up = resample_planar(src, 48, 36, &cfg, &cache, &pool);
            assert!(up.is_ok(), "upscale failed for {algo:?}");
            if let Ok(u) = up {
                assert_eq!((u.width, u.height), (48, 36));
                for &v in &u.r {
                    assert!(v.is_finite());
                }
            }
        }

        let down_cfg = ResampleConfig {
            filter: ResampleFilter::SsimDown,
            ..Default::default()
        };
        let down = resample_planar(src, 12, 9, &down_cfg, &cache, &pool);
        assert!(down.is_ok());

        let krig_cfg = ResampleConfig {
            filter: ResampleFilter::Fsr,
            use_krig_chroma: true,
            ..Default::default()
        };
        assert!(resample_planar(src, 48, 36, &krig_cfg, &cache, &pool).is_ok());

        let h_src = compute_dhash_64(src);
        if let Ok(up_fsr) = resample_planar(src, 48, 36, &ResampleConfig::default(), &cache, &pool) {
            let h_up = compute_dhash_64(&up_fsr);
            assert!(dhash_distance(h_src, h_up) <= 10);
        }

        let plan = build_axis_plan(h, 36, ResampleFilter::Lanczos3);
        let mut scalar_row = vec![0.0f32; w as usize];
        let taps = plan.tap_counts[5] as usize;
        let woff = plan.weight_offsets[5] as usize;
        vertical_row_scalar(
            &src.r,
            &mut scalar_row,
            w as usize,
            plan.src_starts[5] as usize,
            &plan.weights[woff..woff + taps],
            0.0,
        );
        assert!(scalar_row.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn restoration_trim_deblock_deband_dither() {
        use super::geometry::trim::{crop_planar, detect_solid_borders};
        use super::restore::deband::deband_stochastic;
        use super::restore::deblock::deblock_shifted_dct;
        use super::restore::dither::{DitherMode, apply_dither_and_grain};
        use super::restore::guided::guided_filter_denoise;

        let w = 32u32;
        let h = 24u32;
        let mut p = PlanarF32::new(w, h, false, WorkingSpace::LinearSrgb).ok();
        assert!(p.is_some());
        let Some(ref mut img) = p else {
            return;
        };
        for y in 0..h as usize {
            for x in 0..w as usize {
                let idx = y * w as usize + x;
                if (4..28).contains(&x) && (3..21).contains(&y) {
                    img.r[idx] = ((x / 4) as f32) * 0.1;
                    img.g[idx] = (y as f32 / h as f32) * 0.7;
                    img.b[idx] = 0.4;
                }
            }
        }

        let rect = detect_solid_borders(img, 0.01);
        assert_eq!((rect.x, rect.y, rect.width, rect.height), (4, 3, 24, 18));
        let cropped = crop_planar(img, rect);
        assert!(cropped.is_ok());

        let pool = ThreadPool::new(2);
        let guided = guided_filter_denoise(img, 2, 0.01, 0.25);
        assert!(guided.is_ok());

        let deblocked = deblock_shifted_dct(img, 0.5, false);
        assert!(deblocked.is_ok());

        let debanded = deband_stochastic(img, 0.6, 2, 10.0, &pool);
        assert!(debanded.is_ok());

        for mode in [DitherMode::Bayer, DitherMode::FloydSteinberg, DitherMode::BlueNoise] {
            let dith = apply_dither_and_grain(img, mode, 0.15, 255.0);
            assert!(dith.is_ok());
            if let Ok(d) = dith {
                assert!(d.r.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
            }
        }
    }

    #[test]
    fn phase5_smart_fit_seam_carve_and_animations() {
        use super::codec::ImageFormat;
        use super::codec::anim::{decode_animation, encode_animation};
        use super::geometry::fit::{
            FitConfig, FitMode, Gravity, MonitorRegion, extract_oklab_palette, fit_planar,
            parse_hex_color, split_multi_monitor,
        };
        use super::geometry::saliency::{SafeZone, compute_saliency_map};
        use super::geometry::seam::build_vertical_visibility_map;
        use super::pixel::{Frame, FrameMeta, FrameSequence, PackedImage, PixelFormat};
        use super::resample::ResampleConfig;

        let w = 40u32;
        let h = 24u32;
        let p = PlanarF32::new(w, h, false, WorkingSpace::LinearSrgb).ok();
        assert!(p.is_some());
        let Some(ref src) = p else {
            return;
        };
        let mut src = src.clone();
        for y in 0..h as usize {
            for x in 0..w as usize {
                let idx = y * w as usize + x;
                let in_subject = (24..34).contains(&x) && (6..18).contains(&y);
                src.r[idx] = if in_subject { 0.85 } else { 0.12 };
                src.g[idx] = if in_subject { 0.55 } else { 0.25 + (y as f32 / h as f32) * 0.2 };
                src.b[idx] = if in_subject { 0.45 } else { 0.55 };
            }
        }

        let sal = compute_saliency_map(&src);
        assert!(sal.total_mass > 0.0);
        let palette = extract_oklab_palette(&src, 3);
        assert_eq!(palette.len(), 3);
        assert!(parse_hex_color("#1e2430").is_some());

        let vmap = build_vertical_visibility_map(&src, 6);
        assert_eq!((vmap.width, vmap.height), (w as usize, h as usize));

        let cache = PlanCache::new(32);
        let pool = ThreadPool::new(2);
        let res_cfg = ResampleConfig::default();

        let fit_modes = [
            FitMode::Auto,
            FitMode::SmartCrop,
            FitMode::SeamCarve,
            FitMode::Ambient,
            FitMode::Extend,
            FitMode::Mirror,
            FitMode::Gradient,
            FitMode::Contain,
            FitMode::Stretch,
            FitMode::Tile,
        ];
        for mode in fit_modes {
            let fcfg = FitConfig {
                mode,
                gravity: Gravity::Auto,
                safe_zone: SafeZone {
                    top: 0.05,
                    bottom: 0.05,
                    left: 0.0,
                    right: 0.0,
                },
                bg_blur_radius: 4,
                corner_radius: 4.0,
                foreground_dim: 0.1,
                vignette: 0.15,
                ..Default::default()
            };
            let fitted = fit_planar(&src, 32, 32, &fcfg, &res_cfg, &cache, &pool);
            assert!(fitted.is_ok(), "fit_planar failed for {mode:?}");
            if let Ok(out) = fitted {
                assert_eq!((out.width, out.height), (32, 32));
                assert!(out.r.iter().all(|v| v.is_finite()));
            }
        }

        let monitors = vec![
            MonitorRegion {
                name: "DP-1".to_string(),
                x: 0,
                y: 0,
                width: 32,
                height: 18,
            },
            MonitorRegion {
                name: "HDMI-1".to_string(),
                x: 32,
                y: 0,
                width: 24,
                height: 18,
            },
        ];
        let splits = split_multi_monitor(&src, &monitors, 4, &FitConfig::default(), &res_cfg, &cache, &pool);
        assert!(splits.is_ok());
        if let Ok(outputs) = splits {
            assert_eq!(outputs.len(), 2);
            assert_eq!((outputs[0].1.width, outputs[0].1.height), (32, 18));
            assert_eq!((outputs[1].1.width, outputs[1].1.height), (24, 18));
        }

        let mut seq_frames = Vec::new();
        for f_idx in 0..2u8 {
            let mut rgba = Vec::with_capacity(8 * 8 * 4);
            for i in 0..64usize {
                rgba.extend_from_slice(&[f_idx.wrapping_mul(100).saturating_add(20), (i * 3) as u8, 128, 255]);
            }
            if let Ok(packed) = PackedImage::new(8, 8, PixelFormat::Rgba8, ColorSpace::Srgb, rgba) {
                seq_frames.push(Frame {
                    image: packed,
                    meta: FrameMeta {
                        duration_ms: 40,
                        left: 0,
                        top: 0,
                    },
                });
            }
        }
        let seq = FrameSequence {
            canvas_width: 8,
            canvas_height: 8,
            loop_count: 0,
            frames: seq_frames,
        };
        for anim_fmt in [ImageFormat::Gif, ImageFormat::Png] {
            let enc = encode_animation(&seq, anim_fmt);
            assert!(enc.is_ok(), "encode_animation failed for {anim_fmt:?}");
            if let Ok(bytes) = enc {
                let dec = decode_animation(&bytes, anim_fmt, 1_000_000);
                assert!(dec.is_ok(), "decode_animation failed for {anim_fmt:?}");
                if let Ok(dseq) = dec {
                    assert_eq!(dseq.frames.len(), 2);
                }
            }
        }
    }

    #[test]
    fn phase6_forensics_resdet_and_quality_metrics() {
        use super::codec::{DecodeOptions, EncodeOptions, ImageFormat, decode_image, encode_image};
        use super::inspect::inspect_image_bytes;
        use super::inspect::metrics::{compare_images, export_palette_css, export_palette_json};
        use super::inspect::resdet::{descale_if_suspected, detect_native_resolution};
        use super::resample::kernels::ResampleFilter;
        use super::resample::separable::resample_separable;

        let cache = PlanCache::new(32);
        let pool = ThreadPool::new(2);

        let nw = 64u32;
        let nh = 48u32;
        let native_opt = PlanarF32::new(nw, nh, false, WorkingSpace::LinearSrgb).ok();
        assert!(native_opt.is_some());
        let Some(mut native) = native_opt else {
            return;
        };
        for y in 0..nh as usize {
            for x in 0..nw as usize {
                let i = y * nw as usize + x;
                let hf = if (x + y) % 2 == 0 { 0.8 } else { 0.15 };
                let wave = ((x as f32 * 1.9).sin() * (y as f32 * 1.7).cos() * 0.35 + 0.5).clamp(0.0, 1.0);
                native.r[i] = (hf * 0.45 + wave * 0.55).clamp(0.0, 1.0);
                native.g[i] = ((x ^ y) as f32 / 63.0).clamp(0.0, 1.0);
                native.b[i] = 1.0 - native.r[i];
            }
        }

        let upscaled_res =
            resample_separable(&native, 128, 96, ResampleFilter::Mitchell, 0.0, &cache, &pool);
        assert!(upscaled_res.is_ok());
        let Ok(fake_upscale) = upscaled_res else {
            return;
        };

        let report_fake = detect_native_resolution(&fake_upscale, &cache, &pool);
        assert!(
            report_fake.is_fake_upscale,
            "expected fake upscale detection, got {report_fake:?}"
        );
        assert!(report_fake.detected_native_width.is_some() || report_fake.detected_native_height.is_some());

        let descaled = descale_if_suspected(&fake_upscale, 0.35, &cache, &pool);
        assert!(descaled.is_ok());
        if let Ok((d_img, _, did_descale)) = descaled {
            assert!(did_descale);
            assert!(d_img.width < fake_upscale.width || d_img.height < fake_upscale.height);
        }

        let report_native = detect_native_resolution(&native, &cache, &pool);
        assert!(!report_native.is_fake_upscale || report_native.confidence_x < 0.7);

        let same_metrics = compare_images(&native, &native);
        assert!(same_metrics.is_ok());
        if let Ok(m) = same_metrics {
            assert!(m.ssim > 0.999);
            assert!(m.ssim2_est >= 99.9);
        }

        if let Ok(packed_up) = fake_upscale.to_packed(PixelFormat::Rgba8, super::color::global_luts())
            && let Ok(bytes) = encode_image(&packed_up, ImageFormat::Png, &EncodeOptions::default(), None)
        {
            let insp = inspect_image_bytes(&bytes, Some("png"), 10_000_000, &cache, &pool);
            assert!(insp.is_ok());
            if let Ok(rep) = insp {
                assert_eq!((rep.width, rep.height), (128, 96));
                assert!(!rep.palette.is_empty());
                assert!(rep.to_json().contains("\"palette\""));
                assert!(rep.to_text().contains("forensics:"));
                assert!(export_palette_json(&rep.palette).contains("#"));
                assert!(export_palette_css(&rep.palette).contains("--palette-1"));
            }
            let _ = decode_image(&bytes, Some("png"), &DecodeOptions::default());
        } else {
            assert_eq!(0, 1);
        }
    }

    #[test]
    fn phase7_pipeline_and_hardened_nested_archives() {
        use super::archive::security::{
            ArchiveBudgetTracker, ArchiveSecurityPolicy, check_non_overlapping_ranges,
            sanitize_archive_path,
        };
        use super::archive::{
            ArchiveEntryOutput, ArchiveFormat, ArchiveMode, ArchiveProcessOptions, DedupMode,
            NonImagePolicy, process_archive_bytes, write_archive_entries,
        };
        use super::codec::{EncodeOptions, ImageFormat, encode_image};
        use super::geometry::spec::TargetGeometry;
        use super::pipeline::{PipelineConfig, process_image_bytes};
        use super::restore::dither::DitherMode;
        use std::io::Cursor;

        assert!(sanitize_archive_path("../../etc/passwd").is_err());
        assert!(sanitize_archive_path("..\\evil.png").is_err());
        assert!(sanitize_archive_path("C:\\windows\\x.png").is_err());
        assert!(sanitize_archive_path("ok\0bad.png").is_err());
        assert_eq!(
            sanitize_archive_path("/sub/dir/./wall.png").ok(),
            Some("sub/dir/wall.png".to_string())
        );

        let mut ranges = [(100, 300), (250, 400)];
        assert!(check_non_overlapping_ranges(&mut ranges).is_err());

        let mut tracker = ArchiveBudgetTracker::new(ArchiveSecurityPolicy {
            max_entry_bytes: 2048,
            max_total_bytes: 8192,
            ..Default::default()
        });
        let zeros = vec![0u8; 4096];
        assert!(tracker.read_bounded(Cursor::new(&zeros), None).is_err());

        let cache = PlanCache::new(32);
        let pool = ThreadPool::new(2);
        let budget = MemoryBudget::new(256 * 1024 * 1024);

        let mut raw_img = vec![0u8; 20 * 16 * 4];
        for y in 0..16usize {
            for x in 0..20usize {
                let i = (y * 20 + x) * 4;
                raw_img[i] = (x * 12) as u8;
                raw_img[i + 1] = (y * 15) as u8;
                raw_img[i + 2] = 180;
                raw_img[i + 3] = 255;
            }
        }
        let packed = PackedImage::new(20, 16, PixelFormat::Rgba8, ColorSpace::Srgb, raw_img);
        assert!(packed.is_ok());
        let Ok(packed) = packed else {
            return;
        };
        let png_res = encode_image(&packed, ImageFormat::Png, &EncodeOptions::default(), None);
        assert!(png_res.is_ok());
        let Ok(png_bytes) = png_res else {
            return;
        };

        let tgt1 = TargetGeometry::parse("40x32").ok();
        let tgt2 = TargetGeometry::parse("20х16").ok();
        assert!(tgt1.is_some() && tgt2.is_some());
        let pcfg = PipelineConfig {
            targets: vec![tgt1.unwrap_or(TargetGeometry::ScaleFactor(2.0))],
            deband_strength: 0.4,
            dither_mode: DitherMode::BlueNoise,
            extract_palette: 4,
            target_ssim2: Some(75.0),
            output_format: Some(ImageFormat::Jpeg),
            ..Default::default()
        };
        let proc_out = process_image_bytes(&png_bytes, Some("png"), &pcfg, &budget, &cache, &pool);
        assert!(proc_out.is_ok());
        if let Ok(list) = proc_out {
            assert_eq!(list.len(), 1);
            assert_eq!((list[0].dst_width, list[0].dst_height), (40, 32));
            assert_eq!(list[0].format, ImageFormat::Jpeg);
            assert_eq!(list[0].palette.len(), 4);
        }

        let mut hero_raw = vec![0u8; 20 * 16 * 4];
        for (idx, px) in hero_raw.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            *px = [((idx * 7) % 255) as u8, 90, 210, 255];
        }
        let hero_packed = PackedImage::new(20, 16, PixelFormat::Rgba8, ColorSpace::Srgb, hero_raw);
        assert!(hero_packed.is_ok());
        let Ok(hero_packed) = hero_packed else {
            return;
        };
        let hero_res = encode_image(&hero_packed, ImageFormat::Png, &EncodeOptions::default(), None);
        assert!(hero_res.is_ok());
        let Ok(hero_bytes) = hero_res else {
            return;
        };

        let inner_entries = vec![ArchiveEntryOutput {
            path: "nested/hero.png".to_string(),
            data: hero_bytes,
            was_image: true,
        }];
        let inner_zst = write_archive_entries(&inner_entries, ArchiveFormat::TarZst);
        assert!(inner_zst.is_ok());
        let Ok(inner_zst_bytes) = inner_zst else {
            return;
        };

        let outer_entries = vec![
            ArchiveEntryOutput {
                path: "cover.png".to_string(),
                data: png_bytes.clone(),
                was_image: true,
            },
            ArchiveEntryOutput {
                path: "dup/cover_copy.png".to_string(),
                data: png_bytes,
                was_image: true,
            },
            ArchiveEntryOutput {
                path: "docs/notes.txt".to_string(),
                data: b"readme metadata".to_vec(),
                was_image: false,
            },
            ArchiveEntryOutput {
                path: "pack.tar.zst".to_string(),
                data: inner_zst_bytes,
                was_image: false,
            },
        ];
        let outer_zip = write_archive_entries(&outer_entries, ArchiveFormat::Zip);
        assert!(outer_zip.is_ok());
        let Ok(outer_zip_bytes) = outer_zip else {
            return;
        };

        let mut evil_zip = outer_zip_bytes.clone();
        if let Some(pos) = evil_zip.windows(9).position(|w| w == b"cover.png") {
            evil_zip[pos..pos + 9].copy_from_slice(b"../xx.png");
        }
        let arc_opts = ArchiveProcessOptions {
            mode: ArchiveMode::Repack,
            output_format: Some(ArchiveFormat::TarZst),
            non_images: NonImagePolicy::Keep,
            dedup: DedupMode::Exact,
            ..Default::default()
        };
        let scale_cfg = PipelineConfig {
            targets: vec![TargetGeometry::ScaleFactor(2.0)],
            ..Default::default()
        };
        let evil_res = process_archive_bytes(
            &evil_zip,
            Some("evil.zip"),
            &scale_cfg,
            &arc_opts,
            &budget,
            &cache,
            &pool,
        );
        assert!(evil_res.is_err());

        let mut evil_central_zip = outer_zip_bytes.clone();
        for pos in (0..evil_central_zip.len().saturating_sub(9)).rev() {
            if &evil_central_zip[pos..pos + 9] == b"cover.png" {
                evil_central_zip[pos..pos + 9].copy_from_slice(b"../xx.png");
            }
        }
        assert!(
            process_archive_bytes(
                &evil_central_zip,
                Some("evil_central.zip"),
                &scale_cfg,
                &arc_opts,
                &budget,
                &cache,
                &pool,
            )
            .is_err()
        );

        let clean_res = process_archive_bytes(
            &outer_zip_bytes,
            Some("bundle.zip"),
            &scale_cfg,
            &arc_opts,
            &budget,
            &cache,
            &pool,
        );
        assert!(clean_res.is_ok());
        if let Ok(outcome) = clean_res {
            assert_eq!(outcome.format, ArchiveFormat::TarZst);
            assert_eq!(outcome.stats.nested_archives, 1);
            assert_eq!(outcome.stats.duplicates_removed, 1);
            assert_eq!(outcome.stats.non_images_kept, 1);
            assert_eq!(outcome.stats.images_processed, 2);
            assert!(outcome.repacked_bytes.is_some());
        }
    }
}
