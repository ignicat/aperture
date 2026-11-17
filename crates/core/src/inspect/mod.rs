pub mod metrics;
pub mod resdet;

use crate::arena::PlanCache;
use crate::classify::{ContentKind, classify_image, compute_dhash_64};
use crate::codec::{DecodeOptions, ImageFormat, decode_image};
use crate::color::{WorkingSpace, global_luts};
use crate::error::EngineResult;
use crate::geometry::fit::extract_oklab_palette;
use crate::inspect::resdet::{NativeResolutionReport, detect_native_resolution};
use crate::pixel::{PixelFormat, PlanarF32};
use crate::pool::ThreadPool;

#[derive(Debug, Clone)]
pub struct InspectReport {
    pub format: Option<ImageFormat>,
    pub width: u32,
    pub height: u32,
    pub pixel_format: PixelFormat,
    pub is_lossless_container: bool,
    pub has_alpha: bool,
    pub is_animated: bool,
    pub frame_count: u32,
    pub icc_bytes: usize,
    pub exif_bytes: usize,
    pub xmp_bytes: usize,
    pub exif_orientation: u8,
    pub content_kind: ContentKind,
    pub highlight_clip_ratio: f32,
    pub shadow_crush_ratio: f32,
    pub banding_risk: f32,
    pub jpeg_blockiness: f32,
    pub dhash_64: u64,
    pub palette: Vec<[u8; 3]>,
    pub native_resolution: NativeResolutionReport,
    pub file_bytes: usize,
}

impl InspectReport {
    pub fn to_json(&self) -> String {
        let hexes: Vec<String> = self
            .palette
            .iter()
            .map(|c| format!("\"#{:02x}{:02x}{:02x}\"", c[0], c[1], c[2]))
            .collect();
        let fmt_str = self
            .format
            .map(|f| f.canonical_extension())
            .unwrap_or("unknown");
        format!(
            concat!(
                "{{\n",
                "  \"format\": \"{}\",\n",
                "  \"width\": {},\n",
                "  \"height\": {},\n",
                "  \"pixel_format\": \"{:?}\",\n",
                "  \"lossless_container\": {},\n",
                "  \"has_alpha\": {},\n",
                "  \"animated\": {},\n",
                "  \"frame_count\": {},\n",
                "  \"icc_bytes\": {},\n",
                "  \"exif_bytes\": {},\n",
                "  \"xmp_bytes\": {},\n",
                "  \"exif_orientation\": {},\n",
                "  \"content_kind\": \"{:?}\",\n",
                "  \"highlight_clip_ratio\": {:.4},\n",
                "  \"shadow_crush_ratio\": {:.4},\n",
                "  \"banding_risk\": {:.4},\n",
                "  \"jpeg_blockiness\": {:.4},\n",
                "  \"dhash_64\": \"{:016x}\",\n",
                "  \"palette\": [{}],\n",
                "  \"is_fake_upscale\": {},\n",
                "  \"detected_native_width\": {},\n",
                "  \"detected_native_height\": {},\n",
                "  \"native_confidence_x\": {:.3},\n",
                "  \"native_confidence_y\": {:.3},\n",
                "  \"file_bytes\": {}\n",
                "}}"
            ),
            fmt_str,
            self.width,
            self.height,
            self.pixel_format,
            self.is_lossless_container,
            self.has_alpha,
            self.is_animated,
            self.frame_count,
            self.icc_bytes,
            self.exif_bytes,
            self.xmp_bytes,
            self.exif_orientation,
            self.content_kind,
            self.highlight_clip_ratio,
            self.shadow_crush_ratio,
            self.banding_risk,
            self.jpeg_blockiness,
            self.dhash_64,
            hexes.join(", "),
            self.native_resolution.is_fake_upscale,
            self.native_resolution
                .detected_native_width
                .map_or("null".to_string(), |v| v.to_string()),
            self.native_resolution
                .detected_native_height
                .map_or("null".to_string(), |v| v.to_string()),
            self.native_resolution.confidence_x,
            self.native_resolution.confidence_y,
            self.file_bytes,
        )
    }

    pub fn to_text(&self) -> String {
        let fmt_str = self
            .format
            .map(|f| f.canonical_extension())
            .unwrap_or("unknown");
        let native_str = match (
            self.native_resolution.detected_native_width,
            self.native_resolution.detected_native_height,
        ) {
            (Some(nw), Some(nh)) => format!(
                "{}x{} (conf {:.2}/{:.2})",
                nw, nh, self.native_resolution.confidence_x, self.native_resolution.confidence_y
            ),
            (Some(nw), None) => format!(
                "{}x? (conf {:.2})",
                nw, self.native_resolution.confidence_x
            ),
            (None, Some(nh)) => format!(
                "?x{} (conf {:.2})",
                nh, self.native_resolution.confidence_y
            ),
            (None, None) => "native (no synthetic upscale detected)".to_string(),
        };
        let pal_str: Vec<String> = self
            .palette
            .iter()
            .map(|c| format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2]))
            .collect();
        format!(
            "format: {} | {}x{} ({:?}) | {} bytes | frames: {}\n\
             content: {:?} | lossless: {} | alpha: {} | dhash: {:016x}\n\
             metadata: icc={}B exif={}B xmp={}B orient={}\n\
             signal: clip={:.2}% crush={:.2}% banding={:.2} blockiness={:.2}\n\
             forensics: fake_upscale={} | native={}\n\
             palette: {}",
            fmt_str,
            self.width,
            self.height,
            self.pixel_format,
            self.file_bytes,
            self.frame_count,
            self.content_kind,
            self.is_lossless_container,
            self.has_alpha,
            self.dhash_64,
            self.icc_bytes,
            self.exif_bytes,
            self.xmp_bytes,
            self.exif_orientation,
            self.highlight_clip_ratio * 100.0,
            self.shadow_crush_ratio * 100.0,
            self.banding_risk,
            self.jpeg_blockiness,
            self.native_resolution.is_fake_upscale,
            native_str,
            pal_str.join(" "),
        )
    }
}

pub fn inspect_image_bytes(
    bytes: &[u8],
    ext_hint: Option<&str>,
    max_pixels: u64,
    cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<InspectReport> {
    let dec_opts = DecodeOptions {
        max_pixels,
        auto_orient: false,
        convert_icc_to_srgb: false,
        target_svg_size: None,
    };
    let decoded = decode_image(bytes, ext_hint, &dec_opts)?;
    let planar = PlanarF32::from_packed(&decoded.image, WorkingSpace::LinearSrgb, global_luts())?;

    let clip_crush = compute_clip_crush_banding_blockiness(&planar);
    let profile = classify_image(&planar);
    let dhash_64 = compute_dhash_64(&planar);
    let palette = extract_oklab_palette(&planar, 5);
    let native_resolution = detect_native_resolution(&planar, cache, pool);

    Ok(InspectReport {
        format: Some(decoded.format),
        width: decoded.image.width,
        height: decoded.image.height,
        pixel_format: decoded.image.format,
        is_lossless_container: !decoded.was_lossy,
        has_alpha: decoded.image.format.has_alpha(),
        is_animated: decoded.is_animated,
        frame_count: decoded.frame_count,
        icc_bytes: decoded.metadata.icc_profile.as_ref().map_or(0, Vec::len),
        exif_bytes: decoded.metadata.exif.as_ref().map_or(0, Vec::len),
        xmp_bytes: decoded.metadata.xmp.as_ref().map_or(0, Vec::len),
        exif_orientation: decoded.metadata.orientation,
        content_kind: profile.kind,
        highlight_clip_ratio: clip_crush.0,
        shadow_crush_ratio: clip_crush.1,
        banding_risk: clip_crush.2,
        jpeg_blockiness: clip_crush.3,
        dhash_64,
        palette,
        native_resolution,
        file_bytes: bytes.len(),
    })
}

fn compute_clip_crush_banding_blockiness(src: &PlanarF32) -> (f32, f32, f32, f32) {
    let w = src.width as usize;
    let h = src.height as usize;
    let n = src.r.len().max(1);
    let mut clipped = 0usize;
    let mut crushed = 0usize;
    let mut small_grad = 0usize;
    let mut smooth_area = 0usize;

    for i in 0..n {
        let r = src.r[i];
        let g = src.g[i];
        let b = src.b[i];
        if r >= 0.992 || g >= 0.992 || b >= 0.992 {
            clipped += 1;
        }
        if r <= 0.004 && g <= 0.004 && b <= 0.004 {
            crushed += 1;
        }
    }

    let mut block_jump = 0.0f64;
    let mut block_samples = 0usize;
    let mut inner_jump = 0.0f64;
    let mut inner_samples = 0usize;

    for y in 0..h {
        for x in 1..w {
            let idx = y * w + x;
            let d = (src.g[idx] - src.g[idx - 1]).abs();
            if d < 0.025 {
                smooth_area += 1;
                if d == 0.0 || (d > 0.003 && d < 0.009) {
                    small_grad += 1;
                }
            }
            if x.is_multiple_of(8) {
                block_jump += d as f64;
                block_samples += 1;
            } else {
                inner_jump += d as f64;
                inner_samples += 1;
            }
        }
    }

    let block_mean = block_jump / block_samples.max(1) as f64;
    let inner_mean = inner_jump / inner_samples.max(1) as f64;
    let blockiness = if inner_mean > 1e-5 {
        ((block_mean / inner_mean) - 1.0).clamp(0.0, 2.0) as f32
    } else {
        0.0
    };
    let banding = if smooth_area > 0 {
        (small_grad as f32 / smooth_area as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };

    (
        clipped as f32 / n as f32,
        crushed as f32 / n as f32,
        banding,
        blockiness,
    )
}
