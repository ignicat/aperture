use crate::error::{EngineError, EngineResult};
use crate::pixel::PackedImage;
use img_parts::jpeg::{Jpeg, JpegSegment, markers};
use img_parts::png::{Png, PngChunk};
use img_parts::riff::{RiffChunk, RiffContent};
use img_parts::webp::WebP;
use img_parts::{Bytes, ImageEXIF, ImageICC};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataPolicy {
    Keep,
    StripPrivacy,
    StripAll,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContainerMetadata {
    pub icc_profile: Option<Vec<u8>>,
    pub exif: Option<Vec<u8>>,
    pub xmp: Option<Vec<u8>>,
    pub orientation: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerKind {
    Jpeg,
    Png,
    Webp,
    Other,
}

pub fn read_exif_orientation(exif: &[u8]) -> u8 {
    let tiff = if let Some(rest) = exif.strip_prefix(b"Exif\0\0") {
        rest
    } else {
        exif
    };
    if tiff.len() < 8 {
        return 1;
    }
    let little_endian = match &tiff[0..4] {
        b"II\x2A\0" => true,
        b"MM\0\x2A" => false,
        _ => return 1,
    };
    let read_u16 = |off: usize| -> Option<u16> {
        let slice = tiff.get(off..off + 2)?;
        Some(if little_endian {
            u16::from_le_bytes([slice[0], slice[1]])
        } else {
            u16::from_be_bytes([slice[0], slice[1]])
        })
    };
    let read_u32 = |off: usize| -> Option<u32> {
        let slice = tiff.get(off..off + 4)?;
        Some(if little_endian {
            u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]])
        } else {
            u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]])
        })
    };
    let Some(ifd0_off_u32) = read_u32(4) else {
        return 1;
    };
    let ifd0_off = ifd0_off_u32 as usize;
    let Some(count) = read_u16(ifd0_off) else {
        return 1;
    };
    let entries_start = ifd0_off + 2;
    for idx in 0..(count as usize).min(256) {
        let entry_off = entries_start + idx * 12;
        let Some(tag) = read_u16(entry_off) else {
            break;
        };
        if tag == 0x0112 {
            let Some(ty) = read_u16(entry_off + 2) else {
                break;
            };
            let Some(cnt) = read_u32(entry_off + 4) else {
                break;
            };
            if ty == 3
                && cnt >= 1
                && let Some(val) = read_u16(entry_off + 8)
                && (1..=8).contains(&val)
            {
                return val as u8;
            }
        }
    }
    1
}

pub fn normalize_exif_orientation(exif: &[u8]) -> Option<Vec<u8>> {
    let prefix_len = if exif.starts_with(b"Exif\0\0") { 6 } else { 0 };
    if exif.len() < prefix_len + 8 {
        return None;
    }
    let mut out = exif.to_vec();
    let tiff = &mut out[prefix_len..];
    let little_endian = match &tiff[0..4] {
        b"II\x2A\0" => true,
        b"MM\0\x2A" => false,
        _ => return None,
    };
    let read_u16 = |buf: &[u8], off: usize| -> Option<u16> {
        let s = buf.get(off..off + 2)?;
        Some(if little_endian {
            u16::from_le_bytes([s[0], s[1]])
        } else {
            u16::from_be_bytes([s[0], s[1]])
        })
    };
    let read_u32 = |buf: &[u8], off: usize| -> Option<u32> {
        let s = buf.get(off..off + 4)?;
        Some(if little_endian {
            u32::from_le_bytes([s[0], s[1], s[2], s[3]])
        } else {
            u32::from_be_bytes([s[0], s[1], s[2], s[3]])
        })
    };
    let ifd0_off = read_u32(tiff, 4)? as usize;
    let count = read_u16(tiff, ifd0_off)?;
    let entries_start = ifd0_off + 2;
    for idx in 0..(count as usize).min(256) {
        let entry_off = entries_start + idx * 12;
        if let Some(0x0112) = read_u16(tiff, entry_off)
            && let Some(3) = read_u16(tiff, entry_off + 2)
            && let Some(slot) = tiff.get_mut(entry_off + 8..entry_off + 10)
        {
            let one = if little_endian {
                1u16.to_le_bytes()
            } else {
                1u16.to_be_bytes()
            };
            slot.copy_from_slice(&one);
            return Some(out);
        }
    }
    Some(out)
}

pub fn apply_exif_orientation(img: PackedImage, orientation: u8) -> EngineResult<PackedImage> {
    if orientation <= 1 || orientation > 8 {
        return Ok(img);
    }
    let w = img.width as usize;
    let h = img.height as usize;
    let bpp = img.format.bytes_per_pixel();
    let (new_w, new_h) = match orientation {
        5..=8 => (h, w),
        _ => (w, h),
    };
    let mut dst = vec![0u8; img.data.len()];
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = match orientation {
                2 => (w - 1 - x, y),
                3 => (w - 1 - x, h - 1 - y),
                4 => (x, h - 1 - y),
                5 => (y, x),
                6 => (h - 1 - y, x),
                7 => (h - 1 - y, w - 1 - x),
                8 => (y, w - 1 - x),
                _ => (x, y),
            };
            let src_i = (y * w + x) * bpp;
            let dst_i = (dy * new_w + dx) * bpp;
            dst[dst_i..dst_i + bpp].copy_from_slice(&img.data[src_i..src_i + bpp]);
        }
    }
    let mut out = PackedImage::new(new_w as u32, new_h as u32, img.format, img.color_space, dst)?;
    out.icc_profile = img.icc_profile;
    Ok(out)
}

const XMP_JPEG_HEADER: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
const XMP_PNG_PREFIX: &[u8] = b"XML:com.adobe.xmp\0\0\0\0\0";

pub fn extract_metadata(kind: ContainerKind, raw: &[u8]) -> ContainerMetadata {
    let mut meta = ContainerMetadata {
        orientation: 1,
        ..ContainerMetadata::default()
    };
    match kind {
        ContainerKind::Jpeg => {
            if let Ok(jpeg) = Jpeg::from_bytes(Bytes::copy_from_slice(raw)) {
                meta.icc_profile = jpeg.icc_profile().map(|b| b.to_vec());
                meta.exif = jpeg.exif().map(|b| b.to_vec());
                for seg in jpeg.segments_by_marker(markers::APP1) {
                    if let Some(xmp_body) = seg.contents().strip_prefix(XMP_JPEG_HEADER) {
                        meta.xmp = Some(xmp_body.to_vec());
                        break;
                    }
                }
            }
        }
        ContainerKind::Png => {
            if let Ok(png) = Png::from_bytes(Bytes::copy_from_slice(raw)) {
                meta.icc_profile = png.icc_profile().map(|b| b.to_vec());
                meta.exif = png.exif().map(|b| b.to_vec());
                for chunk in png.chunks_by_type(*b"iTXt") {
                    if let Some(xmp_body) = chunk.contents().strip_prefix(XMP_PNG_PREFIX) {
                        meta.xmp = Some(xmp_body.to_vec());
                        break;
                    }
                }
            }
        }
        ContainerKind::Webp => {
            if let Ok(webp) = WebP::from_bytes(Bytes::copy_from_slice(raw)) {
                meta.icc_profile = webp.icc_profile().map(|b| b.to_vec());
                meta.exif = webp.exif().map(|b| b.to_vec());
                if let Some(xmp) = webp.chunk_by_id(*b"XMP ")
                    && let RiffContent::Data(data) = xmp.content()
                {
                    meta.xmp = Some(data.to_vec());
                }
            }
        }
        ContainerKind::Other => {}
    }
    if let Some(exif) = &meta.exif {
        meta.orientation = read_exif_orientation(exif);
    }
    meta
}

pub fn inject_metadata(
    kind: ContainerKind,
    encoded: Vec<u8>,
    meta: &ContainerMetadata,
    policy: MetadataPolicy,
) -> EngineResult<Vec<u8>> {
    if policy == MetadataPolicy::StripAll {
        return Ok(encoded);
    }
    let icc = meta.icc_profile.as_ref().map(|b| Bytes::copy_from_slice(b));
    let exif = if policy == MetadataPolicy::Keep {
        meta.exif
            .as_deref()
            .and_then(|raw| normalize_exif_orientation(raw).or_else(|| Some(raw.to_vec())))
            .map(Bytes::from)
    } else {
        None
    };
    let xmp = if policy == MetadataPolicy::Keep {
        meta.xmp.as_ref()
    } else {
        None
    };
    if icc.is_none() && exif.is_none() && xmp.is_none() {
        return Ok(encoded);
    }

    match kind {
        ContainerKind::Jpeg => {
            let mut jpeg = Jpeg::from_bytes(Bytes::from(encoded))
                .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
            if let Some(p) = icc {
                jpeg.set_icc_profile(Some(p));
            }
            if let Some(ex) = exif {
                jpeg.set_exif(Some(ex));
            }
            if let Some(x) = xmp {
                let mut payload = Vec::with_capacity(XMP_JPEG_HEADER.len() + x.len());
                payload.extend_from_slice(XMP_JPEG_HEADER);
                payload.extend_from_slice(x);
                let seg = JpegSegment::new_with_contents(markers::APP1, Bytes::from(payload));
                let pos = jpeg.segments().len().min(1);
                jpeg.segments_mut().insert(pos, seg);
            }
            let mut out = Vec::new();
            jpeg.encoder()
                .write_to(&mut out)
                .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
            Ok(out)
        }
        ContainerKind::Png => {
            let mut png = Png::from_bytes(Bytes::from(encoded))
                .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
            if let Some(p) = icc {
                png.set_icc_profile(Some(p));
            }
            if let Some(ex) = exif {
                png.set_exif(Some(ex));
            }
            if let Some(x) = xmp {
                let mut payload = Vec::with_capacity(XMP_PNG_PREFIX.len() + x.len());
                payload.extend_from_slice(XMP_PNG_PREFIX);
                payload.extend_from_slice(x);
                let chunk = PngChunk::new(*b"iTXt", Bytes::from(payload));
                let insert_at = png
                    .chunks()
                    .iter()
                    .position(|c| c.kind() == *b"IDAT")
                    .unwrap_or_else(|| png.chunks().len());
                png.chunks_mut().insert(insert_at, chunk);
            }
            let mut out = Vec::new();
            png.encoder()
                .write_to(&mut out)
                .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
            Ok(out)
        }
        ContainerKind::Webp => {
            let mut webp = WebP::from_bytes(Bytes::from(encoded))
                .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
            if let Some(p) = icc {
                webp.set_icc_profile(Some(p));
            }
            if let Some(ex) = exif {
                webp.set_exif(Some(ex));
            }
            if let Some(x) = xmp {
                webp.remove_chunks_by_id(*b"XMP ");
                webp.chunks_mut()
                    .push(RiffChunk::new(*b"XMP ", RiffContent::Data(Bytes::copy_from_slice(x))));
            }
            let mut out = Vec::new();
            webp.encoder()
                .write_to(&mut out)
                .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
            Ok(out)
        }
        ContainerKind::Other => Ok(encoded),
    }
}
