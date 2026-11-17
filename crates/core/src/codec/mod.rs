pub mod anim;
pub mod bmp_tga;
pub mod metadata;
pub mod pnm_hdr_ff;

use crate::color::{ColorSpace, global_luts, transform_icc_u16, transform_icc_u8};
use crate::error::{EngineError, EngineResult};
use crate::pixel::{PackedImage, PixelFormat};
use metadata::{
    ContainerKind, ContainerMetadata, MetadataPolicy, apply_exif_orientation, extract_metadata,
    inject_metadata,
};
use moxcms::Layout;
use std::io::Cursor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    Png,
    Jpeg,
    Webp,
    Gif,
    Tiff,
    Bmp,
    Tga,
    Pnm,
    Qoi,
    Ico,
    Farbfeld,
    Hdr,
    Exr,
    Jxl,
    Avif,
    Svg,
}

impl ImageFormat {
    pub fn canonical_extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Webp => "webp",
            Self::Gif => "gif",
            Self::Tiff => "tiff",
            Self::Bmp => "bmp",
            Self::Tga => "tga",
            Self::Pnm => "ppm",
            Self::Qoi => "qoi",
            Self::Ico => "ico",
            Self::Farbfeld => "ff",
            Self::Hdr => "hdr",
            Self::Exr => "exr",
            Self::Jxl => "jxl",
            Self::Avif => "avif",
            Self::Svg => "svg",
        }
    }

    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.trim_start_matches('.').to_ascii_lowercase().as_str() {
            "png" | "apng" => Some(Self::Png),
            "jpg" | "jpeg" | "jpe" | "jfif" => Some(Self::Jpeg),
            "webp" => Some(Self::Webp),
            "gif" => Some(Self::Gif),
            "tif" | "tiff" => Some(Self::Tiff),
            "bmp" | "dib" => Some(Self::Bmp),
            "tga" | "icb" | "vda" | "vst" => Some(Self::Tga),
            "pbm" | "pgm" | "ppm" | "pam" | "pnm" => Some(Self::Pnm),
            "qoi" => Some(Self::Qoi),
            "ico" | "cur" => Some(Self::Ico),
            "ff" | "farbfeld" => Some(Self::Farbfeld),
            "hdr" | "pic" | "rgbe" => Some(Self::Hdr),
            "exr" => Some(Self::Exr),
            "jxl" => Some(Self::Jxl),
            "avif" => Some(Self::Avif),
            "svg" | "svgz" => Some(Self::Svg),
            _ => None,
        }
    }

    pub fn is_lossless(self) -> bool {
        matches!(
            self,
            Self::Png
                | Self::Tiff
                | Self::Bmp
                | Self::Tga
                | Self::Pnm
                | Self::Qoi
                | Self::Ico
                | Self::Farbfeld
                | Self::Hdr
                | Self::Exr
        )
    }

    pub fn default_output_format(self, force_lossless: bool) -> Self {
        match self {
            Self::Svg => Self::Png,
            Self::Jpeg if force_lossless => Self::Png,
            other => other,
        }
    }

    pub fn container_kind(self) -> ContainerKind {
        match self {
            Self::Jpeg => ContainerKind::Jpeg,
            Self::Png => ContainerKind::Png,
            Self::Webp => ContainerKind::Webp,
            _ => ContainerKind::Other,
        }
    }
}

pub fn detect_format(bytes: &[u8]) -> Option<ImageFormat> {
    if bytes.len() >= 8 && bytes[0..8] == [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
        return Some(ImageFormat::Png);
    }
    if bytes.len() >= 3 && bytes[0..3] == [0xFF, 0xD8, 0xFF] {
        return Some(ImageFormat::Jpeg);
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some(ImageFormat::Webp);
    }
    if bytes.len() >= 6 && (&bytes[0..6] == b"GIF87a" || &bytes[0..6] == b"GIF89a") {
        return Some(ImageFormat::Gif);
    }
    if bytes.len() >= 4
        && (&bytes[0..4] == b"II*\0"
            || &bytes[0..4] == b"MM\0*"
            || &bytes[0..4] == b"II+\0"
            || &bytes[0..4] == b"MM\0+")
    {
        return Some(ImageFormat::Tiff);
    }
    if bytes.len() >= 2 && &bytes[0..2] == b"BM" {
        return Some(ImageFormat::Bmp);
    }
    if bytes.len() >= 4 && &bytes[0..4] == b"qoif" {
        return Some(ImageFormat::Qoi);
    }
    if bytes.len() >= 8 && &bytes[0..8] == b"farbfeld" {
        return Some(ImageFormat::Farbfeld);
    }
    if bytes.len() >= 4 && bytes[0..4] == [0x76, 0x2F, 0x31, 0x01] {
        return Some(ImageFormat::Exr);
    }
    if bytes.starts_with(b"#?RADIANCE") || bytes.starts_with(b"#?RGBE") {
        return Some(ImageFormat::Hdr);
    }
    if bytes.len() >= 26 && &bytes[bytes.len() - 18..] == b"TRUEVISION-XFILE.\0" {
        return Some(ImageFormat::Tga);
    }
    if bytes.len() >= 22 && bytes[0] == 0 && bytes[1] == 0 && (bytes[2] == 1 || bytes[2] == 2) && bytes[3] == 0 {
        let count = u16::from_le_bytes([bytes[4], bytes[5]]) as usize;
        if count >= 1 && 6 + count * 16 <= bytes.len() {
            let size0 = u32::from_le_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]) as usize;
            let off0 = u32::from_le_bytes([bytes[18], bytes[19], bytes[20], bytes[21]]) as usize;
            if off0 >= 6 + count * 16 && off0.saturating_add(size0) <= bytes.len() {
                return Some(ImageFormat::Ico);
            }
        }
    }
    if bytes.len() >= 3 && bytes[0] == b'P' && matches!(bytes[1], b'1'..=b'7') && bytes[2].is_ascii_whitespace() {
        return Some(ImageFormat::Pnm);
    }
    if bytes.len() >= 2 && bytes[0..2] == [0xFF, 0x0A]
        || (bytes.len() >= 12 && bytes[0..12] == [0, 0, 0, 12, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A])
    {
        return Some(ImageFormat::Jxl);
    }
    if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" && (&bytes[8..12] == b"avif" || &bytes[8..12] == b"avis") {
        return Some(ImageFormat::Avif);
    }
    let trim_start = bytes.iter().position(|b| !b.is_ascii_whitespace()).unwrap_or(bytes.len());
    let head = &bytes[trim_start..bytes.len().min(trim_start + 256)];
    if head.starts_with(b"<svg") || (head.starts_with(b"<?xml") && head.windows(4).any(|w| w == b"<svg")) {
        return Some(ImageFormat::Svg);
    }
    None
}

#[derive(Debug, Clone)]
pub struct DecodeOptions {
    pub max_pixels: u64,
    pub auto_orient: bool,
    pub convert_icc_to_srgb: bool,
    pub target_svg_size: Option<(u32, u32)>,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            max_pixels: 268_435_456,
            auto_orient: true,
            convert_icc_to_srgb: false,
            target_svg_size: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub format: ImageFormat,
    pub image: PackedImage,
    pub metadata: ContainerMetadata,
    pub was_lossy: bool,
    pub is_animated: bool,
    pub frame_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSubsampling {
    Auto,
    S444,
    S422,
    S420,
}

#[derive(Debug, Clone)]
pub struct EncodeOptions {
    pub quality: u8,
    pub lossless: bool,
    pub optimize: bool,
    pub png_compression: u8,
    pub png_quant_colors: Option<u16>,
    pub chroma_subsampling: ChromaSubsampling,
    pub meta_policy: MetadataPolicy,
    pub ico_sizes: Vec<u32>,
    pub max_bytes: Option<u64>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            quality: 98,
            lossless: false,
            optimize: true,
            png_compression: 6,
            png_quant_colors: None,
            chroma_subsampling: ChromaSubsampling::S444,
            meta_policy: MetadataPolicy::Keep,
            ico_sizes: vec![16, 32, 48, 64, 128, 256],
            max_bytes: None,
        }
    }
}

fn check_pixel_limit(width: u32, height: u32, max_pixels: u64) -> EngineResult<()> {
    if width == 0 || height == 0 {
        return Err(EngineError::InvalidDimensions { width, height });
    }
    let px = (width as u64).saturating_mul(height as u64);
    if max_pixels > 0 && px > max_pixels {
        return Err(EngineError::SecurityLimit(format!(
            "image dimensions {width}x{height} ({px} px) exceed max_pixels {max_pixels}"
        )));
    }
    Ok(())
}

pub fn decode_image(
    bytes: &[u8],
    ext_hint: Option<&str>,
    opts: &DecodeOptions,
) -> EngineResult<DecodedImage> {
    let format = detect_format(bytes)
        .or_else(|| ext_hint.and_then(ImageFormat::from_extension))
        .ok_or_else(|| EngineError::UnsupportedFormat("unrecognized image signature".to_string()))?;

    let mut meta = extract_metadata(format.container_kind(), bytes);
    let mut was_lossy = matches!(format, ImageFormat::Jpeg);
    let mut is_animated = false;
    let mut frame_count = 1u32;

    let mut img = match format {
        ImageFormat::Png => {
            let mut decoder = png::Decoder::new(Cursor::new(bytes));
            decoder.set_transformations(png::Transformations::EXPAND);
            let mut reader = decoder
                .read_info()
                .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
            let (width, height) = (reader.info().width, reader.info().height);
            check_pixel_limit(width, height, opts.max_pixels)?;
            if let Some(fc) = reader.info().animation_control() {
                is_animated = fc.num_frames > 1;
                frame_count = fc.num_frames;
            }
            if let Some(icc) = &reader.info().icc_profile
                && meta.icc_profile.is_none()
            {
                meta.icc_profile = Some(icc.to_vec());
            }
            let buf_size = reader
                .output_buffer_size()
                .ok_or_else(|| EngineError::DecodeFailed("PNG output size overflow".to_string()))?;
            let mut buf = vec![0u8; buf_size];
            let out_info = reader
                .next_frame(&mut buf)
                .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
            buf.truncate(out_info.buffer_size());
            let fmt = match (out_info.color_type, out_info.bit_depth) {
                (png::ColorType::Grayscale, png::BitDepth::Eight) => PixelFormat::Gray8,
                (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight) => PixelFormat::GrayAlpha8,
                (png::ColorType::Rgb, png::BitDepth::Eight) => PixelFormat::Rgb8,
                (png::ColorType::Rgba, png::BitDepth::Eight) => PixelFormat::Rgba8,
                (png::ColorType::Grayscale, png::BitDepth::Sixteen) => {
                    be16_to_ne(&mut buf);
                    PixelFormat::Gray16
                }
                (png::ColorType::GrayscaleAlpha, png::BitDepth::Sixteen) => {
                    be16_to_ne(&mut buf);
                    PixelFormat::GrayAlpha16
                }
                (png::ColorType::Rgb, png::BitDepth::Sixteen) => {
                    be16_to_ne(&mut buf);
                    PixelFormat::Rgb16
                }
                (png::ColorType::Rgba, png::BitDepth::Sixteen) => {
                    be16_to_ne(&mut buf);
                    PixelFormat::Rgba16
                }
                _ => return Err(EngineError::UnsupportedFormat("PNG color depth".to_string())),
            };
            let cs = if fmt.is_gray() { ColorSpace::Gray } else { ColorSpace::Srgb };
            PackedImage::new(out_info.width, out_info.height, fmt, cs, buf)?
        }
        ImageFormat::Jpeg => decode_jpeg_with_cmyk(bytes, opts, &mut meta)?,
        ImageFormat::Webp => {
            let mut decoder = image_webp::WebPDecoder::new(Cursor::new(bytes))
                .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
            let (width, height) = decoder.dimensions();
            check_pixel_limit(width, height, opts.max_pixels)?;
            was_lossy = decoder.is_lossy();
            is_animated = decoder.is_animated();
            frame_count = decoder.num_frames().max(1);
            if let Ok(Some(icc)) = decoder.icc_profile()
                && meta.icc_profile.is_none()
            {
                meta.icc_profile = Some(icc);
            }
            if let Ok(Some(exif)) = decoder.exif_metadata()
                && meta.exif.is_none()
            {
                meta.orientation = metadata::read_exif_orientation(&exif);
                meta.exif = Some(exif);
            }
            let buf_len = decoder
                .output_buffer_size()
                .ok_or_else(|| EngineError::DecodeFailed("WebP buffer size overflow".to_string()))?;
            let mut buf = vec![0u8; buf_len];
            decoder
                .read_image(&mut buf)
                .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
            let fmt = if decoder.has_alpha() {
                PixelFormat::Rgba8
            } else {
                PixelFormat::Rgb8
            };
            PackedImage::new(width, height, fmt, ColorSpace::Srgb, buf)?
        }
        ImageFormat::Gif => decode_gif_first_frame(bytes, opts, &mut is_animated, &mut frame_count)?,
        ImageFormat::Tiff => decode_tiff_first_page(bytes, opts, &mut meta)?,
        ImageFormat::Bmp => {
            let decoded = bmp_tga::decode_bmp(bytes)?;
            check_pixel_limit(decoded.width, decoded.height, opts.max_pixels)?;
            decoded
        }
        ImageFormat::Tga => {
            let decoded = bmp_tga::decode_tga(bytes)?;
            check_pixel_limit(decoded.width, decoded.height, opts.max_pixels)?;
            decoded
        }
        ImageFormat::Pnm => {
            let decoded = pnm_hdr_ff::decode_pnm(bytes)?;
            check_pixel_limit(decoded.width, decoded.height, opts.max_pixels)?;
            decoded
        }
        ImageFormat::Qoi => {
            let (hdr, buf) = qoi::decode_to_vec(bytes)
                .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
            check_pixel_limit(hdr.width, hdr.height, opts.max_pixels)?;
            let fmt = match hdr.channels {
                qoi::Channels::Rgb => PixelFormat::Rgb8,
                qoi::Channels::Rgba => PixelFormat::Rgba8,
            };
            PackedImage::new(hdr.width, hdr.height, fmt, ColorSpace::Srgb, buf)?
        }
        ImageFormat::Ico => {
            let dir = ico::IconDir::read(Cursor::new(bytes))
                .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
            let entry = dir
                .entries()
                .iter()
                .max_by_key(|e| (e.width() as u64) * (e.height() as u64))
                .ok_or_else(|| EngineError::DecodeFailed("empty ICO directory".to_string()))?;
            check_pixel_limit(entry.width(), entry.height(), opts.max_pixels)?;
            let icon = entry
                .decode()
                .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
            let w = icon.width();
            let h = icon.height();
            let rgba = icon.rgba_data().to_vec();
            PackedImage::new(w, h, PixelFormat::Rgba8, ColorSpace::Srgb, rgba)?
        }
        ImageFormat::Farbfeld => {
            let decoded = pnm_hdr_ff::decode_farbfeld(bytes)?;
            check_pixel_limit(decoded.width, decoded.height, opts.max_pixels)?;
            decoded
        }
        ImageFormat::Hdr => {
            let decoded = pnm_hdr_ff::decode_radiance_hdr(bytes)?;
            check_pixel_limit(decoded.width, decoded.height, opts.max_pixels)?;
            decoded
        }
        ImageFormat::Exr => decode_exr(bytes, opts)?,
        ImageFormat::Jxl => decode_jxl_opt(bytes, opts)?,
        ImageFormat::Avif => {
            was_lossy = true;
            decode_avif_opt(bytes, opts)?
        }
        ImageFormat::Svg => decode_svg_opt(bytes, opts)?,
    };

    if opts.convert_icc_to_srgb
        && let Some(icc) = &meta.icc_profile
    {
        if let Ok(converted) = convert_packed_icc_to_srgb(&img, icc) {
            img = converted;
            meta.icc_profile = None;
        }
    } else if img.icc_profile.is_none() {
        img.icc_profile = meta.icc_profile.clone();
    }

    if opts.auto_orient && meta.orientation > 1 {
        img = apply_exif_orientation(img, meta.orientation)?;
    }

    Ok(DecodedImage {
        format,
        image: img,
        metadata: meta,
        was_lossy,
        is_animated,
        frame_count,
    })
}

fn decode_jpeg_with_cmyk(
    bytes: &[u8],
    opts: &DecodeOptions,
    meta: &mut ContainerMetadata,
) -> EngineResult<PackedImage> {
    let mut decoder = zune_jpeg::JpegDecoder::new(Cursor::new(bytes));
    decoder
        .decode_headers()
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
    let info = decoder
        .info()
        .ok_or_else(|| EngineError::DecodeFailed("missing JPEG info".to_string()))?;
    check_pixel_limit(info.width as u32, info.height as u32, opts.max_pixels)?;
    if meta.icc_profile.is_none() {
        meta.icc_profile = decoder.icc_profile();
    }
    if meta.exif.is_none()
        && let Some(exif) = decoder.exif()
    {
        meta.orientation = metadata::read_exif_orientation(exif);
        meta.exif = Some(exif.clone());
    }
    if meta.xmp.is_none()
        && let Some(xmp) = decoder.xmp()
    {
        meta.xmp = Some(xmp.clone());
    }

    let in_cs = decoder.input_colorspace();
    let pixels = decoder
        .decode()
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
    let width = info.width as u32;
    let height = info.height as u32;

    match in_cs {
        Some(zune_jpeg::zune_core::colorspace::ColorSpace::Luma) => {
            PackedImage::new(width, height, PixelFormat::Gray8, ColorSpace::Gray, pixels)
        }
        Some(zune_jpeg::zune_core::colorspace::ColorSpace::CMYK | zune_jpeg::zune_core::colorspace::ColorSpace::YCCK) => {
            if let Some(icc) = &meta.icc_profile
                && let Ok(rgb) = transform_icc_u8(&pixels, Layout::Rgba, icc, ColorSpace::Srgb, Layout::Rgb)
            {
                meta.icc_profile = None;
                return PackedImage::new(width, height, PixelFormat::Rgb8, ColorSpace::Srgb, rgb);
            }
            let mut rgb = vec![0u8; (width as usize) * (height as usize) * 3];
            for (dst, cmyk) in rgb.as_chunks_mut::<3>().0.iter_mut().zip(pixels.as_chunks::<4>().0) {
                let c = cmyk[0] as f32 / 255.0;
                let m = cmyk[1] as f32 / 255.0;
                let y = cmyk[2] as f32 / 255.0;
                let k = cmyk[3] as f32 / 255.0;
                dst[0] = (c * k * 255.0).round().clamp(0.0, 255.0) as u8;
                dst[1] = (m * k * 255.0).round().clamp(0.0, 255.0) as u8;
                dst[2] = (y * k * 255.0).round().clamp(0.0, 255.0) as u8;
            }
            PackedImage::new(width, height, PixelFormat::Rgb8, ColorSpace::Srgb, rgb)
        }
        _ => PackedImage::new(width, height, PixelFormat::Rgb8, ColorSpace::Srgb, pixels),
    }
}

fn decode_gif_first_frame(
    bytes: &[u8],
    opts: &DecodeOptions,
    is_animated: &mut bool,
    frame_count: &mut u32,
) -> EngineResult<PackedImage> {
    let mut gopts = gif::DecodeOptions::new();
    gopts.set_color_output(gif::ColorOutput::RGBA);
    let mut dec = gopts
        .read_info(Cursor::new(bytes))
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
    let w = dec.width() as u32;
    let h = dec.height() as u32;
    check_pixel_limit(w, h, opts.max_pixels)?;
    let first = dec
        .read_next_frame()
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?
        .ok_or_else(|| EngineError::DecodeFailed("GIF contains no frames".to_string()))?;
    let mut canvas = vec![0u8; (w as usize) * (h as usize) * 4];
    let fw = first.width as usize;
    let fh = first.height as usize;
    let fx = first.left as usize;
    let fy = first.top as usize;
    for row in 0..fh {
        let dy = fy + row;
        if dy >= h as usize {
            break;
        }
        for col in 0..fw {
            let dx = fx + col;
            if dx >= w as usize {
                break;
            }
            let si = (row * fw + col) * 4;
            let di = (dy * w as usize + dx) * 4;
            canvas[di..di + 4].copy_from_slice(&first.buffer[si..si + 4]);
        }
    }
    let mut count = 1u32;
    while let Ok(Some(_)) = dec.next_frame_info() {
        count += 1;
    }
    *frame_count = count;
    *is_animated = count > 1;
    PackedImage::new(w, h, PixelFormat::Rgba8, ColorSpace::Srgb, canvas)
}

fn decode_tiff_first_page(
    bytes: &[u8],
    opts: &DecodeOptions,
    meta: &mut ContainerMetadata,
) -> EngineResult<PackedImage> {
    let mut dec = tiff::decoder::Decoder::new(Cursor::new(bytes))
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
    let (w, h) = dec
        .dimensions()
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
    check_pixel_limit(w, h, opts.max_pixels)?;
    if meta.icc_profile.is_none()
        && let Ok(icc) = dec.get_tag_u8_vec(tiff::tags::Tag::Unknown(34675))
    {
        meta.icc_profile = Some(icc);
    }
    if let Ok(orient) = dec.get_tag_u32(tiff::tags::Tag::Orientation)
        && (1..=8).contains(&orient)
    {
        meta.orientation = orient as u8;
    }
    let ct = dec
        .colortype()
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
    let res = dec
        .read_image()
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
    match (ct, res) {
        (tiff::ColorType::Gray(8), tiff::decoder::DecodingResult::U8(v)) => {
            PackedImage::new(w, h, PixelFormat::Gray8, ColorSpace::Gray, v)
        }
        (tiff::ColorType::GrayA(8), tiff::decoder::DecodingResult::U8(v)) => {
            PackedImage::new(w, h, PixelFormat::GrayAlpha8, ColorSpace::Gray, v)
        }
        (tiff::ColorType::RGB(8), tiff::decoder::DecodingResult::U8(v)) => {
            PackedImage::new(w, h, PixelFormat::Rgb8, ColorSpace::Srgb, v)
        }
        (tiff::ColorType::RGBA(8), tiff::decoder::DecodingResult::U8(v)) => {
            PackedImage::new(w, h, PixelFormat::Rgba8, ColorSpace::Srgb, v)
        }
        (tiff::ColorType::Gray(16), tiff::decoder::DecodingResult::U16(v)) => {
            PackedImage::new(w, h, PixelFormat::Gray16, ColorSpace::Gray, u16_slice_to_ne_bytes(&v))
        }
        (tiff::ColorType::GrayA(16), tiff::decoder::DecodingResult::U16(v)) => {
            PackedImage::new(w, h, PixelFormat::GrayAlpha16, ColorSpace::Gray, u16_slice_to_ne_bytes(&v))
        }
        (tiff::ColorType::RGB(16), tiff::decoder::DecodingResult::U16(v)) => {
            PackedImage::new(w, h, PixelFormat::Rgb16, ColorSpace::Srgb, u16_slice_to_ne_bytes(&v))
        }
        (tiff::ColorType::RGBA(16), tiff::decoder::DecodingResult::U16(v)) => {
            PackedImage::new(w, h, PixelFormat::Rgba16, ColorSpace::Srgb, u16_slice_to_ne_bytes(&v))
        }
        (tiff::ColorType::CMYK(8), tiff::decoder::DecodingResult::U8(v)) => {
            let mut rgb = vec![0u8; (w as usize) * (h as usize) * 3];
            for (dst, cmyk) in rgb.as_chunks_mut::<3>().0.iter_mut().zip(v.as_chunks::<4>().0) {
                let c = 1.0 - cmyk[0] as f32 / 255.0;
                let m = 1.0 - cmyk[1] as f32 / 255.0;
                let y = 1.0 - cmyk[2] as f32 / 255.0;
                let k = 1.0 - cmyk[3] as f32 / 255.0;
                dst[0] = (c * k * 255.0).round().clamp(0.0, 255.0) as u8;
                dst[1] = (m * k * 255.0).round().clamp(0.0, 255.0) as u8;
                dst[2] = (y * k * 255.0).round().clamp(0.0, 255.0) as u8;
            }
            PackedImage::new(w, h, PixelFormat::Rgb8, ColorSpace::Srgb, rgb)
        }
        _ => Err(EngineError::UnsupportedFormat(format!("TIFF {ct:?}"))),
    }
}

fn decode_exr(bytes: &[u8], opts: &DecodeOptions) -> EngineResult<PackedImage> {
    use exr::prelude::*;
    let flat = read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .first_valid_layer()
        .all_attributes()
        .from_buffered(Cursor::new(bytes))
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
    let size = flat.layer_data.size;
    check_pixel_limit(size.width() as u32, size.height() as u32, opts.max_pixels)?;

    let mut rgba_img = read()
        .no_deep_data()
        .largest_resolution_level()
        .rgba_channels(
            |res, _| (res.width(), vec![0.0f32; res.width() * res.height() * 4]),
            |(w, buf), pos, (r, g, b, a): (f32, f32, f32, f32)| {
                let idx = (pos.y() * *w + pos.x()) * 4;
                buf[idx] = r;
                buf[idx + 1] = g;
                buf[idx + 2] = b;
                buf[idx + 3] = a;
            },
        )
        .first_valid_layer()
        .all_attributes()
        .from_buffered(Cursor::new(bytes))
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
    let (_, floats) = std::mem::take(&mut rgba_img.layer_data.channel_data.pixels);
    let mut out = Vec::with_capacity(floats.len() * 4);
    for f in floats {
        out.extend_from_slice(&f.to_ne_bytes());
    }
    PackedImage::new(
        size.width() as u32,
        size.height() as u32,
        PixelFormat::RgbaF32,
        ColorSpace::LinearSrgb,
        out,
    )
}

#[cfg(feature = "jxl")]
fn decode_jxl_opt(bytes: &[u8], opts: &DecodeOptions) -> EngineResult<PackedImage> {
    let image = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(bytes))
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
    let w = image.width();
    let h = image.height();
    check_pixel_limit(w, h, opts.max_pixels)?;
    let render = image
        .render_frame(0)
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
    let fb = render.image_all_channels();
    let channels = fb.channels();
    let buf = fb.buf();
    let mut out = Vec::with_capacity((w as usize) * (h as usize) * 16);
    if channels >= 4 {
        for px in buf.chunks_exact(channels) {
            out.extend_from_slice(&px[0].to_ne_bytes());
            out.extend_from_slice(&px[1].to_ne_bytes());
            out.extend_from_slice(&px[2].to_ne_bytes());
            out.extend_from_slice(&px[3].to_ne_bytes());
        }
        PackedImage::new(w, h, PixelFormat::RgbaF32, ColorSpace::Srgb, out)
    } else if channels == 3 {
        for px in buf.as_chunks::<3>().0 {
            out.extend_from_slice(&px[0].to_ne_bytes());
            out.extend_from_slice(&px[1].to_ne_bytes());
            out.extend_from_slice(&px[2].to_ne_bytes());
        }
        PackedImage::new(w, h, PixelFormat::RgbF32, ColorSpace::Srgb, out)
    } else {
        for &v in buf {
            out.extend_from_slice(&v.to_ne_bytes());
            out.extend_from_slice(&v.to_ne_bytes());
            out.extend_from_slice(&v.to_ne_bytes());
        }
        PackedImage::new(w, h, PixelFormat::RgbF32, ColorSpace::Srgb, out)
    }
}

#[cfg(not(feature = "jxl"))]
fn decode_jxl_opt(_bytes: &[u8], _opts: &DecodeOptions) -> EngineResult<PackedImage> {
    Err(EngineError::UnsupportedFormat(
        "JPEG XL requires the `jxl` feature".to_string(),
    ))
}

#[cfg(feature = "avif")]
fn decode_avif_opt(bytes: &[u8], opts: &DecodeOptions) -> EngineResult<PackedImage> {
    let dec = avif_decode::Decoder::from_avif(bytes)
        .map_err(|e| EngineError::DecodeFailed(format!("{e:?}")))?;
    match dec
        .to_image()
        .map_err(|e| EngineError::DecodeFailed(format!("{e:?}")))?
    {
        avif_decode::Image::Rgb8(img) => {
            let (buf, w, h) = img.into_contiguous_buf();
            check_pixel_limit(w as u32, h as u32, opts.max_pixels)?;
            let raw: Vec<u8> = buf.into_iter().flat_map(|p| [p.r, p.g, p.b]).collect();
            PackedImage::new(w as u32, h as u32, PixelFormat::Rgb8, ColorSpace::Srgb, raw)
        }
        avif_decode::Image::Rgba8(img) => {
            let (buf, w, h) = img.into_contiguous_buf();
            check_pixel_limit(w as u32, h as u32, opts.max_pixels)?;
            let raw: Vec<u8> = buf.into_iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
            PackedImage::new(w as u32, h as u32, PixelFormat::Rgba8, ColorSpace::Srgb, raw)
        }
        avif_decode::Image::Rgb16(img) => {
            let (buf, w, h) = img.into_contiguous_buf();
            check_pixel_limit(w as u32, h as u32, opts.max_pixels)?;
            let mut raw = Vec::with_capacity(buf.len() * 6);
            for p in buf {
                raw.extend_from_slice(&p.r.to_ne_bytes());
                raw.extend_from_slice(&p.g.to_ne_bytes());
                raw.extend_from_slice(&p.b.to_ne_bytes());
            }
            PackedImage::new(w as u32, h as u32, PixelFormat::Rgb16, ColorSpace::Srgb, raw)
        }
        avif_decode::Image::Rgba16(img) => {
            let (buf, w, h) = img.into_contiguous_buf();
            check_pixel_limit(w as u32, h as u32, opts.max_pixels)?;
            let mut raw = Vec::with_capacity(buf.len() * 8);
            for p in buf {
                raw.extend_from_slice(&p.r.to_ne_bytes());
                raw.extend_from_slice(&p.g.to_ne_bytes());
                raw.extend_from_slice(&p.b.to_ne_bytes());
                raw.extend_from_slice(&p.a.to_ne_bytes());
            }
            PackedImage::new(w as u32, h as u32, PixelFormat::Rgba16, ColorSpace::Srgb, raw)
        }
        avif_decode::Image::Gray8(img) => {
            let (buf, w, h) = img.into_contiguous_buf();
            check_pixel_limit(w as u32, h as u32, opts.max_pixels)?;
            let raw: Vec<u8> = buf.into_iter().map(|p| p.value()).collect();
            PackedImage::new(w as u32, h as u32, PixelFormat::Gray8, ColorSpace::Gray, raw)
        }
        avif_decode::Image::Gray16(img) => {
            let (buf, w, h) = img.into_contiguous_buf();
            check_pixel_limit(w as u32, h as u32, opts.max_pixels)?;
            let mut raw = Vec::with_capacity(buf.len() * 2);
            for p in buf {
                raw.extend_from_slice(&p.value().to_ne_bytes());
            }
            PackedImage::new(w as u32, h as u32, PixelFormat::Gray16, ColorSpace::Gray, raw)
        }
    }
}

#[cfg(not(feature = "avif"))]
fn decode_avif_opt(_bytes: &[u8], _opts: &DecodeOptions) -> EngineResult<PackedImage> {
    Err(EngineError::UnsupportedFormat(
        "AVIF requires the `avif` feature".to_string(),
    ))
}

#[cfg(feature = "svg")]
fn decode_svg_opt(bytes: &[u8], opts: &DecodeOptions) -> EngineResult<PackedImage> {
    let tree = resvg::usvg::Tree::from_data(bytes, &resvg::usvg::Options::default())
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
    let src_size = tree.size();
    let (tw, th) = opts.target_svg_size.unwrap_or((
        src_size.width().round().max(1.0) as u32,
        src_size.height().round().max(1.0) as u32,
    ));
    check_pixel_limit(tw, th, opts.max_pixels)?;
    let mut pixmap = resvg::tiny_skia::Pixmap::new(tw, th)
        .ok_or(EngineError::InvalidDimensions { width: tw, height: th })?;
    let sx = tw as f32 / src_size.width().max(1e-4);
    let sy = th as f32 / src_size.height().max(1e-4);
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(sx, sy),
        &mut pixmap.as_mut(),
    );
    let mut rgba = pixmap.take();
    for px in rgba.as_chunks_mut::<4>().0 {
        let a = px[3];
        if a > 0 && a < 255 {
            let af = a as f32 / 255.0;
            px[0] = ((px[0] as f32 / af).round()).clamp(0.0, 255.0) as u8;
            px[1] = ((px[1] as f32 / af).round()).clamp(0.0, 255.0) as u8;
            px[2] = ((px[2] as f32 / af).round()).clamp(0.0, 255.0) as u8;
        }
    }
    PackedImage::new(tw, th, PixelFormat::Rgba8, ColorSpace::Srgb, rgba)
}

#[cfg(not(feature = "svg"))]
fn decode_svg_opt(_bytes: &[u8], _opts: &DecodeOptions) -> EngineResult<PackedImage> {
    Err(EngineError::UnsupportedFormat(
        "SVG rasterization requires the `svg` feature".to_string(),
    ))
}

fn convert_packed_icc_to_srgb(img: &PackedImage, icc: &[u8]) -> EngineResult<PackedImage> {
    match img.format {
        PixelFormat::Rgb8 => {
            let out = transform_icc_u8(&img.data, Layout::Rgb, icc, ColorSpace::Srgb, Layout::Rgb)?;
            PackedImage::new(img.width, img.height, PixelFormat::Rgb8, ColorSpace::Srgb, out)
        }
        PixelFormat::Rgba8 => {
            let out = transform_icc_u8(&img.data, Layout::Rgba, icc, ColorSpace::Srgb, Layout::Rgba)?;
            PackedImage::new(img.width, img.height, PixelFormat::Rgba8, ColorSpace::Srgb, out)
        }
        PixelFormat::Rgb16 => {
            let samples = ne_bytes_to_u16(&img.data);
            let conv = transform_icc_u16(&samples, Layout::Rgb, icc, ColorSpace::Srgb, Layout::Rgb)?;
            PackedImage::new(
                img.width,
                img.height,
                PixelFormat::Rgb16,
                ColorSpace::Srgb,
                u16_slice_to_ne_bytes(&conv),
            )
        }
        PixelFormat::Rgba16 => {
            let samples = ne_bytes_to_u16(&img.data);
            let conv = transform_icc_u16(&samples, Layout::Rgba, icc, ColorSpace::Srgb, Layout::Rgba)?;
            PackedImage::new(
                img.width,
                img.height,
                PixelFormat::Rgba16,
                ColorSpace::Srgb,
                u16_slice_to_ne_bytes(&conv),
            )
        }
        _ => Ok(img.clone()),
    }
}

pub fn encode_image(
    img: &PackedImage,
    format: ImageFormat,
    opts: &EncodeOptions,
    meta: Option<&ContainerMetadata>,
) -> EngineResult<Vec<u8>> {
    if let Some(limit) = opts.max_bytes
        && matches!(format, ImageFormat::Jpeg | ImageFormat::Avif)
    {
        return encode_under_byte_budget(img, format, opts, meta, limit);
    }

    let raw = match format {
        ImageFormat::Png => encode_png(img, opts)?,
        ImageFormat::Jpeg => encode_jpeg(img, opts)?,
        ImageFormat::Webp => encode_webp(img)?,
        ImageFormat::Gif => encode_gif(img)?,
        ImageFormat::Tiff => encode_tiff(img)?,
        ImageFormat::Bmp => bmp_tga::encode_bmp(img)?,
        ImageFormat::Tga => bmp_tga::encode_tga(img)?,
        ImageFormat::Pnm => pnm_hdr_ff::encode_pnm(img)?,
        ImageFormat::Qoi => encode_qoi(img)?,
        ImageFormat::Ico => encode_ico(img, &opts.ico_sizes)?,
        ImageFormat::Farbfeld => pnm_hdr_ff::encode_farbfeld(img)?,
        ImageFormat::Hdr => pnm_hdr_ff::encode_radiance_hdr(img)?,
        ImageFormat::Exr => encode_exr(img)?,
        ImageFormat::Avif => encode_avif_opt(img, opts)?,
        ImageFormat::Jxl => {
            return Err(EngineError::UnsupportedFormat(
                "JPEG XL encoding is not yet enabled; decode supported".to_string(),
            ));
        }
        ImageFormat::Svg => {
            return Err(EngineError::UnsupportedFormat(
                "SVG is rasterized to PNG/raster outputs".to_string(),
            ));
        }
    };

    if let Some(m) = meta {
        inject_metadata(format.container_kind(), raw, m, opts.meta_policy)
    } else {
        Ok(raw)
    }
}

fn encode_under_byte_budget(
    img: &PackedImage,
    format: ImageFormat,
    opts: &EncodeOptions,
    meta: Option<&ContainerMetadata>,
    limit: u64,
) -> EngineResult<Vec<u8>> {
    let mut lo = 10u8;
    let mut hi = opts.quality.clamp(10, 100);
    let mut best: Option<Vec<u8>> = None;
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let mut trial_opts = opts.clone();
        trial_opts.quality = mid;
        trial_opts.max_bytes = None;
        let bytes = encode_image(img, format, &trial_opts, meta)?;
        if bytes.len() as u64 <= limit {
            best = Some(bytes);
            if mid == 100 {
                break;
            }
            lo = mid + 1;
        } else {
            if mid <= 10 {
                if best.is_none() {
                    best = Some(bytes);
                }
                break;
            }
            hi = mid - 1;
        }
    }
    best.ok_or_else(|| EngineError::EncodeFailed("failed to fit byte budget".to_string()))
}

fn encode_png(img: &PackedImage, opts: &EncodeOptions) -> EngineResult<Vec<u8>> {
    let (target_fmt, data, palette) = prepare_png_payload(img, opts);
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, img.width, img.height);
        let (ct, bd) = match target_fmt {
            PngPayloadKind::Gray8 => (png::ColorType::Grayscale, png::BitDepth::Eight),
            PngPayloadKind::GrayAlpha8 => (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight),
            PngPayloadKind::Rgb8 => (png::ColorType::Rgb, png::BitDepth::Eight),
            PngPayloadKind::Rgba8 => (png::ColorType::Rgba, png::BitDepth::Eight),
            PngPayloadKind::Gray16 => (png::ColorType::Grayscale, png::BitDepth::Sixteen),
            PngPayloadKind::GrayAlpha16 => (png::ColorType::GrayscaleAlpha, png::BitDepth::Sixteen),
            PngPayloadKind::Rgb16 => (png::ColorType::Rgb, png::BitDepth::Sixteen),
            PngPayloadKind::Rgba16 => (png::ColorType::Rgba, png::BitDepth::Sixteen),
            PngPayloadKind::Indexed8 => (png::ColorType::Indexed, png::BitDepth::Eight),
        };
        encoder.set_color(ct);
        encoder.set_depth(bd);
        if let Some((pal, trns)) = palette {
            encoder.set_palette(pal);
            if !trns.is_empty() {
                encoder.set_trns(trns);
            }
        }
        let comp = if opts.optimize || opts.png_compression >= 7 {
            png::Compression::High
        } else if opts.png_compression >= 3 {
            png::Compression::Balanced
        } else {
            png::Compression::Fast
        };
        encoder.set_compression(comp);
        encoder.set_filter(if opts.optimize {
            png::Filter::Adaptive
        } else {
            png::Filter::Paeth
        });
        let mut writer = encoder
            .write_header()
            .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
        writer
            .write_image_data(&data)
            .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
    }
    Ok(out)
}

#[derive(Clone, Copy)]
enum PngPayloadKind {
    Gray8,
    GrayAlpha8,
    Rgb8,
    Rgba8,
    Gray16,
    GrayAlpha16,
    Rgb16,
    Rgba16,
    Indexed8,
}

type PalettePayload = (Vec<u8>, Vec<u8>);

fn prepare_png_payload(
    img: &PackedImage,
    opts: &EncodeOptions,
) -> (PngPayloadKind, Vec<u8>, Option<PalettePayload>) {
    let working = if img.format.is_f32() {
        let luts = global_luts();
        match crate::pixel::PlanarF32::from_packed(img, crate::color::WorkingSpace::LinearSrgb, luts)
            .and_then(|p| p.to_packed(if img.format.has_alpha() { PixelFormat::Rgba16 } else { PixelFormat::Rgb16 }, luts))
        {
            Ok(p) => std::borrow::Cow::Owned(p),
            Err(_) => std::borrow::Cow::Borrowed(img),
        }
    } else {
        std::borrow::Cow::Borrowed(img)
    };

    if let Some(max_colors) = opts.png_quant_colors
        && !working.format.is_16bit()
    {
        let (pal, trns, indices) = quantize_to_palette(&working, max_colors.clamp(2, 256) as usize);
        return (PngPayloadKind::Indexed8, indices, Some((pal, trns)));
    }

    if opts.optimize {
        match working.format {
            PixelFormat::Rgba8 => {
                let all_opaque = working.data.as_chunks::<4>().0.iter().all(|px| px[3] == 255);
                if all_opaque {
                    let all_gray = working
                        .data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|px| px[0] == px[1] && px[1] == px[2]);
                    if all_gray {
                        let gray: Vec<u8> =
                            working.data.as_chunks::<4>().0.iter().map(|px| px[0]).collect();
                        return (PngPayloadKind::Gray8, gray, None);
                    }
                    if let Some(pal) = try_exact_palette_rgba(&working.data) {
                        return (PngPayloadKind::Indexed8, pal.2, Some((pal.0, pal.1)));
                    }
                    let mut rgb = Vec::with_capacity(working.pixel_count() * 3);
                    for px in working.data.as_chunks::<4>().0 {
                        rgb.extend_from_slice(&px[0..3]);
                    }
                    return (PngPayloadKind::Rgb8, rgb, None);
                } else if let Some(pal) = try_exact_palette_rgba(&working.data) {
                    return (PngPayloadKind::Indexed8, pal.2, Some((pal.0, pal.1)));
                }
            }
            PixelFormat::Rgb8 => {
                let all_gray = working
                    .data
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .all(|px| px[0] == px[1] && px[1] == px[2]);
                if all_gray {
                    let gray: Vec<u8> =
                        working.data.as_chunks::<3>().0.iter().map(|px| px[0]).collect();
                    return (PngPayloadKind::Gray8, gray, None);
                }
            }
            _ => {}
        }
    }

    match working.format {
        PixelFormat::Gray8 => (PngPayloadKind::Gray8, working.data.clone(), None),
        PixelFormat::GrayAlpha8 => (PngPayloadKind::GrayAlpha8, working.data.clone(), None),
        PixelFormat::Rgb8 => (PngPayloadKind::Rgb8, working.data.clone(), None),
        PixelFormat::Rgba8 => (PngPayloadKind::Rgba8, working.data.clone(), None),
        PixelFormat::Gray16 => (
            PngPayloadKind::Gray16,
            ne_bytes_to_be_bytes(&working.data),
            None,
        ),
        PixelFormat::GrayAlpha16 => (
            PngPayloadKind::GrayAlpha16,
            ne_bytes_to_be_bytes(&working.data),
            None,
        ),
        PixelFormat::Rgb16 => (
            PngPayloadKind::Rgb16,
            ne_bytes_to_be_bytes(&working.data),
            None,
        ),
        PixelFormat::Rgba16 => (
            PngPayloadKind::Rgba16,
            ne_bytes_to_be_bytes(&working.data),
            None,
        ),
        _ => (PngPayloadKind::Rgba8, working.data.clone(), None),
    }
}

fn try_exact_palette_rgba(data: &[u8]) -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut palette: Vec<[u8; 4]> = Vec::new();
    let mut indices = Vec::with_capacity(data.len() / 4);
    for &px in data.as_chunks::<4>().0 {
        if let Some(pos) = palette.iter().position(|&c| c == px) {
            indices.push(pos as u8);
        } else {
            if palette.len() >= 256 {
                return None;
            }
            indices.push(palette.len() as u8);
            palette.push(px);
        }
    }
    let mut rgb_pal = Vec::with_capacity(palette.len() * 3);
    let mut trns = Vec::with_capacity(palette.len());
    let mut has_alpha = false;
    for c in &palette {
        rgb_pal.extend_from_slice(&c[0..3]);
        trns.push(c[3]);
        if c[3] != 255 {
            has_alpha = true;
        }
    }
    if !has_alpha {
        trns.clear();
    }
    Some((rgb_pal, trns, indices))
}

fn quantize_to_palette(img: &PackedImage, max_colors: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let rgba = to_rgba8_bytes(img);
    let w = img.width as usize;
    let h = img.height as usize;
    let mut samples: Vec<[u8; 4]> = rgba.as_chunks::<4>().0.to_vec();
    let mut boxes: Vec<Vec<[u8; 4]>> = vec![std::mem::take(&mut samples)];
    while boxes.len() < max_colors {
        let Some((split_idx, _)) = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() > 1)
            .max_by_key(|(_, b)| color_box_range(b).0)
        else {
            break;
        };
        let mut bucket = boxes.swap_remove(split_idx);
        let (_, axis) = color_box_range(&bucket);
        bucket.sort_unstable_by_key(|c| c[axis]);
        let mid = bucket.len() / 2;
        let right = bucket.split_off(mid);
        boxes.push(bucket);
        boxes.push(right);
    }
    let mut palette: Vec<[u8; 4]> = Vec::with_capacity(boxes.len());
    for bucket in boxes {
        if bucket.is_empty() {
            continue;
        }
        let mut sr = 0u64;
        let mut sg = 0u64;
        let mut sb = 0u64;
        let mut sa = 0u64;
        for px in &bucket {
            sr += px[0] as u64;
            sg += px[1] as u64;
            sb += px[2] as u64;
            sa += px[3] as u64;
        }
        let n = bucket.len() as u64;
        palette.push([
            (sr / n) as u8,
            (sg / n) as u8,
            (sb / n) as u8,
            (sa / n) as u8,
        ]);
    }
    let mut err_buf: Vec<[f32; 4]> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32])
        .collect();
    let mut indices = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let cur = err_buf[idx];
            let best_i = nearest_palette_index(cur, &palette);
            indices[idx] = best_i as u8;
            let chosen = palette[best_i];
            let er = cur[0] - chosen[0] as f32;
            let eg = cur[1] - chosen[1] as f32;
            let eb = cur[2] - chosen[2] as f32;
            let ea = cur[3] - chosen[3] as f32;
            let mut distribute = |nx: usize, ny: usize, factor: f32| {
                let ni = ny * w + nx;
                err_buf[ni][0] = (err_buf[ni][0] + er * factor).clamp(0.0, 255.0);
                err_buf[ni][1] = (err_buf[ni][1] + eg * factor).clamp(0.0, 255.0);
                err_buf[ni][2] = (err_buf[ni][2] + eb * factor).clamp(0.0, 255.0);
                err_buf[ni][3] = (err_buf[ni][3] + ea * factor).clamp(0.0, 255.0);
            };
            if x + 1 < w {
                distribute(x + 1, y, 7.0 / 16.0);
            }
            if y + 1 < h {
                if x > 0 {
                    distribute(x - 1, y + 1, 3.0 / 16.0);
                }
                distribute(x, y + 1, 5.0 / 16.0);
                if x + 1 < w {
                    distribute(x + 1, y + 1, 1.0 / 16.0);
                }
            }
        }
    }
    let mut rgb_pal = Vec::with_capacity(palette.len() * 3);
    let mut trns = Vec::with_capacity(palette.len());
    let mut has_alpha = false;
    for c in &palette {
        rgb_pal.extend_from_slice(&c[0..3]);
        trns.push(c[3]);
        if c[3] != 255 {
            has_alpha = true;
        }
    }
    if !has_alpha {
        trns.clear();
    }
    (rgb_pal, trns, indices)
}

fn color_box_range(bucket: &[[u8; 4]]) -> (u8, usize) {
    let mut min_c = [255u8; 4];
    let mut max_c = [0u8; 4];
    for px in bucket {
        for ch in 0..4 {
            min_c[ch] = min_c[ch].min(px[ch]);
            max_c[ch] = max_c[ch].max(px[ch]);
        }
    }
    let mut best_range = 0u8;
    let mut best_axis = 0usize;
    for ch in 0..4 {
        let range = max_c[ch] - min_c[ch];
        if range >= best_range {
            best_range = range;
            best_axis = ch;
        }
    }
    (best_range, best_axis)
}

fn nearest_palette_index(c: [f32; 4], palette: &[[u8; 4]]) -> usize {
    let mut best_i = 0usize;
    let mut best_dist = f32::MAX;
    for (i, p) in palette.iter().enumerate() {
        let dr = c[0] - p[0] as f32;
        let dg = c[1] - p[1] as f32;
        let db = c[2] - p[2] as f32;
        let da = c[3] - p[3] as f32;
        let d = dr * dr * 0.299 + dg * dg * 0.587 + db * db * 0.114 + da * da;
        if d < best_dist {
            best_dist = d;
            best_i = i;
        }
    }
    best_i
}

fn encode_jpeg(img: &PackedImage, opts: &EncodeOptions) -> EngineResult<Vec<u8>> {
    let quality = if opts.lossless { 100 } else { opts.quality.clamp(1, 100) };
    let mut out = Vec::new();
    let mut encoder = jpeg_encoder::Encoder::new(&mut out, quality);
    let sampling = match opts.chroma_subsampling {
        ChromaSubsampling::S444 | ChromaSubsampling::Auto if quality >= 85 => {
            jpeg_encoder::SamplingFactor::R_4_4_4
        }
        ChromaSubsampling::S422 => jpeg_encoder::SamplingFactor::R_4_2_2,
        _ => jpeg_encoder::SamplingFactor::R_4_2_0,
    };
    encoder.set_sampling_factor(sampling);
    encoder.set_optimized_huffman_tables(opts.optimize);
    encoder.set_progressive(opts.optimize || quality >= 85);

    let (ct, buf): (jpeg_encoder::ColorType, Vec<u8>) = match img.format {
        PixelFormat::Gray8 => (jpeg_encoder::ColorType::Luma, img.data.clone()),
        PixelFormat::Rgb8 => (jpeg_encoder::ColorType::Rgb, img.data.clone()),
        _ => (jpeg_encoder::ColorType::Rgb, to_rgb8_bytes(img)),
    };
    encoder
        .encode(&buf, img.width as u16, img.height as u16, ct)
        .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
    Ok(out)
}

fn encode_webp(img: &PackedImage) -> EngineResult<Vec<u8>> {
    let (ct, data) = match img.format {
        PixelFormat::Gray8 => (image_webp::ColorType::L8, img.data.clone()),
        PixelFormat::GrayAlpha8 => (image_webp::ColorType::La8, img.data.clone()),
        PixelFormat::Rgb8 => (image_webp::ColorType::Rgb8, img.data.clone()),
        PixelFormat::Rgba8 => (image_webp::ColorType::Rgba8, img.data.clone()),
        _ if img.format.has_alpha() => (image_webp::ColorType::Rgba8, to_rgba8_bytes(img)),
        _ => (image_webp::ColorType::Rgb8, to_rgb8_bytes(img)),
    };
    let mut out = Vec::new();
    let enc = image_webp::WebPEncoder::new(&mut out);
    enc.encode(&data, img.width, img.height, ct)
        .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
    Ok(out)
}

fn encode_gif(img: &PackedImage) -> EngineResult<Vec<u8>> {
    let mut rgba = to_rgba8_bytes(img);
    let w = img.width.min(u16::MAX as u32) as u16;
    let h = img.height.min(u16::MAX as u32) as u16;
    let mut out = Vec::new();
    {
        let mut enc = gif::Encoder::new(&mut out, w, h, &[])
            .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
        let frame = gif::Frame::from_rgba_speed(w, h, &mut rgba, 10);
        enc.write_frame(&frame)
            .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
    }
    Ok(out)
}

fn encode_tiff(img: &PackedImage) -> EngineResult<Vec<u8>> {
    let mut out = Cursor::new(Vec::new());
    let mut enc = tiff::encoder::TiffEncoder::new(&mut out)
        .map_err(|e| EngineError::EncodeFailed(e.to_string()))?
        .with_compression(tiff::encoder::Compression::Deflate(Default::default()));
    match img.format {
        PixelFormat::Gray8 => enc
            .write_image::<tiff::encoder::colortype::Gray8>(img.width, img.height, &img.data)
            .map_err(|e| EngineError::EncodeFailed(e.to_string()))?,
        PixelFormat::Gray16 => {
            let s = ne_bytes_to_u16(&img.data);
            enc.write_image::<tiff::encoder::colortype::Gray16>(img.width, img.height, &s)
                .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
        }
        PixelFormat::Rgb8 => enc
            .write_image::<tiff::encoder::colortype::RGB8>(img.width, img.height, &img.data)
            .map_err(|e| EngineError::EncodeFailed(e.to_string()))?,
        PixelFormat::Rgba8 => enc
            .write_image::<tiff::encoder::colortype::RGBA8>(img.width, img.height, &img.data)
            .map_err(|e| EngineError::EncodeFailed(e.to_string()))?,
        PixelFormat::Rgb16 => {
            let s = ne_bytes_to_u16(&img.data);
            enc.write_image::<tiff::encoder::colortype::RGB16>(img.width, img.height, &s)
                .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
        }
        PixelFormat::Rgba16 => {
            let s = ne_bytes_to_u16(&img.data);
            enc.write_image::<tiff::encoder::colortype::RGBA16>(img.width, img.height, &s)
                .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
        }
        _ => {
            let rgba = to_rgba8_bytes(img);
            enc.write_image::<tiff::encoder::colortype::RGBA8>(img.width, img.height, &rgba)
                .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
        }
    }
    Ok(out.into_inner())
}

fn encode_qoi(img: &PackedImage) -> EngineResult<Vec<u8>> {
    match img.format {
        PixelFormat::Rgb8 => {
            qoi::encode_to_vec(&img.data, img.width, img.height).map_err(|e| EngineError::EncodeFailed(e.to_string()))
        }
        PixelFormat::Rgba8 => {
            qoi::encode_to_vec(&img.data, img.width, img.height).map_err(|e| EngineError::EncodeFailed(e.to_string()))
        }
        _ if img.format.has_alpha() => {
            let rgba = to_rgba8_bytes(img);
            qoi::encode_to_vec(&rgba, img.width, img.height).map_err(|e| EngineError::EncodeFailed(e.to_string()))
        }
        _ => {
            let rgb = to_rgb8_bytes(img);
            qoi::encode_to_vec(&rgb, img.width, img.height).map_err(|e| EngineError::EncodeFailed(e.to_string()))
        }
    }
}

fn encode_ico(img: &PackedImage, sizes: &[u32]) -> EngineResult<Vec<u8>> {
    let mut dir = ico::IconDir::new(ico::ResourceType::Icon);
    let target_sizes = if sizes.is_empty() { &[img.width.clamp(1, 256)] } else { sizes };
    let src_rgba = to_rgba8_bytes(img);
    let Ok(src_fir) = fast_image_resize::images::Image::from_vec_u8(
        img.width,
        img.height,
        src_rgba,
        fast_image_resize::PixelType::U8x4,
    ) else {
        return Err(EngineError::EncodeFailed("ICO source setup failed".to_string()));
    };
    let mut resizer = fast_image_resize::Resizer::new();
    for &sz in target_sizes {
        let clamped = sz.clamp(1, 256);
        let mut dst_fir = fast_image_resize::images::Image::new(
            clamped,
            clamped,
            fast_image_resize::PixelType::U8x4,
        );
        let opts = fast_image_resize::ResizeOptions::new()
            .resize_alg(fast_image_resize::ResizeAlg::Convolution(
                fast_image_resize::FilterType::Lanczos3,
            ))
            .use_alpha(true);
        resizer
            .resize(&src_fir, &mut dst_fir, Some(&opts))
            .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
        let icon_img = ico::IconImage::from_rgba_data(clamped, clamped, dst_fir.into_vec());
        let entry = ico::IconDirEntry::encode(&icon_img)
            .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
        dir.add_entry(entry);
    }
    let mut out = Vec::new();
    dir.write(&mut out)
        .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
    Ok(out)
}

fn encode_exr(img: &PackedImage) -> EngineResult<Vec<u8>> {
    use exr::prelude::*;
    let luts = global_luts();
    let planar = crate::pixel::PlanarF32::from_packed(
        img,
        crate::color::WorkingSpace::LinearSrgb,
        luts,
    )?;
    let w = img.width as usize;
    let h = img.height as usize;
    let mut out = Cursor::new(Vec::new());
    let channels = SpecificChannels::rgba(|Vec2(x, y)| {
        let idx = y * w + x;
        let a = if planar.has_alpha { planar.a[idx] } else { 1.0 };
        (planar.r[idx], planar.g[idx], planar.b[idx], a)
    });
    Image::from_channels((w, h), channels)
        .write()
        .to_buffered(&mut out)
        .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
    Ok(out.into_inner())
}

#[cfg(feature = "avif")]
fn encode_avif_opt(img: &PackedImage, opts: &EncodeOptions) -> EngineResult<Vec<u8>> {
    let rgba = to_rgba8_bytes(img);
    let pixels: Vec<ravif::RGBA8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| ravif::RGBA8::new(p[0], p[1], p[2], p[3]))
        .collect();
    let q = if opts.lossless { 100.0 } else { opts.quality.clamp(1, 100) as f32 };
    let res = ravif::Encoder::new()
        .with_quality(q)
        .with_speed(6)
        .encode_rgba(ravif::Img::new(&pixels, img.width as usize, img.height as usize))
        .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
    Ok(res.avif_file)
}

#[cfg(not(feature = "avif"))]
fn encode_avif_opt(_img: &PackedImage, _opts: &EncodeOptions) -> EngineResult<Vec<u8>> {
    Err(EngineError::UnsupportedFormat(
        "AVIF encoding requires the `avif` feature".to_string(),
    ))
}

pub fn to_rgb8_bytes(img: &PackedImage) -> Vec<u8> {
    if img.format == PixelFormat::Rgb8 {
        return img.data.clone();
    }
    let luts = global_luts();
    if let Ok(p) = crate::pixel::PlanarF32::from_packed(img, crate::color::WorkingSpace::LinearSrgb, luts)
        && let Ok(conv) = p.to_packed(PixelFormat::Rgb8, luts)
    {
        return conv.data;
    }
    Vec::new()
}

pub fn to_rgba8_bytes(img: &PackedImage) -> Vec<u8> {
    if img.format == PixelFormat::Rgba8 {
        return img.data.clone();
    }
    let luts = global_luts();
    if let Ok(p) = crate::pixel::PlanarF32::from_packed(img, crate::color::WorkingSpace::LinearSrgb, luts)
        && let Ok(conv) = p.to_packed(PixelFormat::Rgba8, luts)
    {
        return conv.data;
    }
    Vec::new()
}

fn be16_to_ne(buf: &mut [u8]) {
    for c in buf.as_chunks_mut::<2>().0 {
        let v = u16::from_be_bytes(*c);
        *c = v.to_ne_bytes();
    }
}

fn ne_bytes_to_be_bytes(buf: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(buf.len());
    for c in buf.as_chunks::<2>().0 {
        let v = u16::from_ne_bytes(*c);
        out.extend_from_slice(&v.to_be_bytes());
    }
    out
}

fn u16_slice_to_ne_bytes(samples: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        out.extend_from_slice(&s.to_ne_bytes());
    }
    out
}

fn ne_bytes_to_u16(buf: &[u8]) -> Vec<u16> {
    buf.as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_ne_bytes(*c))
        .collect()
}
