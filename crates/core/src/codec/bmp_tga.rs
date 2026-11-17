use crate::color::ColorSpace;
use crate::error::{EngineError, EngineResult};
use crate::pixel::{PackedImage, PixelFormat};

pub fn decode_bmp(bytes: &[u8]) -> EngineResult<PackedImage> {
    if bytes.len() < 54 || &bytes[0..2] != b"BM" {
        return Err(EngineError::DecodeFailed("invalid BMP header".to_string()));
    }
    let pixel_offset = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
    let dib_size = u32::from_le_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]) as usize;
    if dib_size < 40 || bytes.len() < 14 + dib_size {
        return Err(EngineError::DecodeFailed("unsupported BMP DIB header".to_string()));
    }
    let width_i = i32::from_le_bytes([bytes[18], bytes[19], bytes[20], bytes[21]]);
    let height_i = i32::from_le_bytes([bytes[22], bytes[23], bytes[24], bytes[25]]);
    if width_i <= 0 || height_i == 0 {
        return Err(EngineError::DecodeFailed("invalid BMP dimensions".to_string()));
    }
    let width = width_i as u32;
    let top_down = height_i < 0;
    let height = height_i.unsigned_abs();
    let planes = u16::from_le_bytes([bytes[26], bytes[27]]);
    let bpp = u16::from_le_bytes([bytes[28], bytes[29]]);
    let compression = u32::from_le_bytes([bytes[30], bytes[31], bytes[32], bytes[33]]);
    if planes != 1 {
        return Err(EngineError::DecodeFailed("invalid BMP plane count".to_string()));
    }
    if pixel_offset >= bytes.len() {
        return Err(EngineError::DecodeFailed("BMP pixel offset out of range".to_string()));
    }
    let px_data = &bytes[pixel_offset..];
    let w = width as usize;
    let h = height as usize;
    if w > 65536 || h > 65536 {
        return Err(EngineError::InvalidDimensions { width, height });
    }

    match (bpp, compression) {
        (24, 0) => {
            let row_stride = (w * 3).div_ceil(4) * 4;
            let needed = row_stride
                .checked_mul(h)
                .ok_or(EngineError::InvalidDimensions { width, height })?;
            if px_data.len() < needed {
                return Err(EngineError::DecodeFailed("truncated 24-bit BMP".to_string()));
            }
            let mut out = vec![0u8; w * h * 3];
            for row in 0..h {
                let src_row = if top_down { row } else { h - 1 - row };
                let s = &px_data[src_row * row_stride..src_row * row_stride + w * 3];
                let d = &mut out[row * w * 3..(row + 1) * w * 3];
                for x in 0..w {
                    d[x * 3] = s[x * 3 + 2];
                    d[x * 3 + 1] = s[x * 3 + 1];
                    d[x * 3 + 2] = s[x * 3];
                }
            }
            PackedImage::new(width, height, PixelFormat::Rgb8, ColorSpace::Srgb, out)
        }
        (32, 0 | 3) => {
            let row_stride = w * 4;
            let needed = row_stride
                .checked_mul(h)
                .ok_or(EngineError::InvalidDimensions { width, height })?;
            if px_data.len() < needed {
                return Err(EngineError::DecodeFailed("truncated 32-bit BMP".to_string()));
            }
            let mut r_shift = 16u32;
            let mut g_shift = 8u32;
            let mut b_shift = 0u32;
            let mut a_shift = 24u32;
            let mut has_alpha_mask = dib_size >= 56;
            if compression == 3 && bytes.len() >= 14 + 52 {
                let rm = u32::from_le_bytes([bytes[54], bytes[55], bytes[56], bytes[57]]);
                let gm = u32::from_le_bytes([bytes[58], bytes[59], bytes[60], bytes[61]]);
                let bm = u32::from_le_bytes([bytes[62], bytes[63], bytes[64], bytes[65]]);
                if rm != 0 && gm != 0 && bm != 0 {
                    r_shift = rm.trailing_zeros();
                    g_shift = gm.trailing_zeros();
                    b_shift = bm.trailing_zeros();
                }
                if bytes.len() >= 14 + 56 && dib_size >= 56 {
                    let am = u32::from_le_bytes([bytes[66], bytes[67], bytes[68], bytes[69]]);
                    if am != 0 {
                        a_shift = am.trailing_zeros();
                        has_alpha_mask = true;
                    } else {
                        has_alpha_mask = false;
                    }
                }
            } else if compression == 0 {
                has_alpha_mask = true;
            }
            let mut out = vec![0u8; w * h * 4];
            let mut all_zero_alpha = true;
            for row in 0..h {
                let src_row = if top_down { row } else { h - 1 - row };
                let s = &px_data[src_row * row_stride..src_row * row_stride + row_stride];
                let d = &mut out[row * row_stride..(row + 1) * row_stride];
                for x in 0..w {
                    let word = u32::from_le_bytes([
                        s[x * 4],
                        s[x * 4 + 1],
                        s[x * 4 + 2],
                        s[x * 4 + 3],
                    ]);
                    d[x * 4] = ((word >> r_shift) & 0xFF) as u8;
                    d[x * 4 + 1] = ((word >> g_shift) & 0xFF) as u8;
                    d[x * 4 + 2] = ((word >> b_shift) & 0xFF) as u8;
                    let a = if has_alpha_mask {
                        ((word >> a_shift) & 0xFF) as u8
                    } else {
                        255
                    };
                    if a != 0 {
                        all_zero_alpha = false;
                    }
                    d[x * 4 + 3] = a;
                }
            }
            if all_zero_alpha {
                for px in out.as_chunks_mut::<4>().0 {
                    px[3] = 255;
                }
            }
            PackedImage::new(width, height, PixelFormat::Rgba8, ColorSpace::Srgb, out)
        }
        (8, 0) => {
            let palette_offset = 14 + dib_size;
            let colors_used = u32::from_le_bytes([bytes[46], bytes[47], bytes[48], bytes[49]]);
            let palette_len = if colors_used == 0 { 256 } else { (colors_used as usize).min(256) };
            if pixel_offset < palette_offset + palette_len * 4 {
                return Err(EngineError::DecodeFailed("truncated BMP palette".to_string()));
            }
            let palette = &bytes[palette_offset..palette_offset + palette_len * 4];
            let row_stride = w.div_ceil(4) * 4;
            let needed = row_stride
                .checked_mul(h)
                .ok_or(EngineError::InvalidDimensions { width, height })?;
            if px_data.len() < needed {
                return Err(EngineError::DecodeFailed("truncated 8-bit BMP".to_string()));
            }
            let mut out = vec![0u8; w * h * 3];
            for row in 0..h {
                let src_row = if top_down { row } else { h - 1 - row };
                let s = &px_data[src_row * row_stride..src_row * row_stride + w];
                let d = &mut out[row * w * 3..(row + 1) * w * 3];
                for x in 0..w {
                    let idx = (s[x] as usize).min(palette_len - 1) * 4;
                    d[x * 3] = palette[idx + 2];
                    d[x * 3 + 1] = palette[idx + 1];
                    d[x * 3 + 2] = palette[idx];
                }
            }
            PackedImage::new(width, height, PixelFormat::Rgb8, ColorSpace::Srgb, out)
        }
        _ => Err(EngineError::UnsupportedFormat(format!(
            "BMP bpp={bpp} compression={compression}"
        ))),
    }
}

pub fn encode_bmp(img: &PackedImage) -> EngineResult<Vec<u8>> {
    let w = img.width as usize;
    let h = img.height as usize;
    let (bpp, row_stride, has_alpha) = match img.format {
        PixelFormat::Rgba8 | PixelFormat::GrayAlpha8 | PixelFormat::Rgba16 | PixelFormat::GrayAlpha16 | PixelFormat::RgbaF32 => {
            (32u16, w * 4, true)
        }
        _ => (24u16, (w * 3).div_ceil(4) * 4, false),
    };
    let pixel_size = row_stride
        .checked_mul(h)
        .ok_or(EngineError::InvalidDimensions {
            width: img.width,
            height: img.height,
        })?;
    let header_size = 54u32;
    let file_size = (header_size as usize)
        .checked_add(pixel_size)
        .ok_or(EngineError::InvalidDimensions {
            width: img.width,
            height: img.height,
        })?;
    let mut out = vec![0u8; file_size];
    out[0..2].copy_from_slice(b"BM");
    out[2..6].copy_from_slice(&(file_size as u32).to_le_bytes());
    out[10..14].copy_from_slice(&header_size.to_le_bytes());
    out[14..18].copy_from_slice(&40u32.to_le_bytes());
    out[18..22].copy_from_slice(&(img.width as i32).to_le_bytes());
    out[22..26].copy_from_slice(&(img.height as i32).to_le_bytes());
    out[26..28].copy_from_slice(&1u16.to_le_bytes());
    out[28..30].copy_from_slice(&bpp.to_le_bytes());
    out[34..38].copy_from_slice(&(pixel_size as u32).to_le_bytes());

    let px_dst = &mut out[header_size as usize..];
    for row in 0..h {
        let dst_row = h - 1 - row;
        let d = &mut px_dst[dst_row * row_stride..(dst_row + 1) * row_stride];
        for x in 0..w {
            let (r, g, b, a) = read_u8_rgba(img, row * w + x);
            if has_alpha {
                d[x * 4] = b;
                d[x * 4 + 1] = g;
                d[x * 4 + 2] = r;
                d[x * 4 + 3] = a;
            } else {
                d[x * 3] = b;
                d[x * 3 + 1] = g;
                d[x * 3 + 2] = r;
            }
        }
    }
    Ok(out)
}

pub fn decode_tga(bytes: &[u8]) -> EngineResult<PackedImage> {
    if bytes.len() < 18 {
        return Err(EngineError::DecodeFailed("truncated TGA header".to_string()));
    }
    let id_len = bytes[0] as usize;
    let color_map_type = bytes[1];
    let image_type = bytes[2];
    if color_map_type != 0 {
        return Err(EngineError::UnsupportedFormat("color-mapped TGA".to_string()));
    }
    let width = u16::from_le_bytes([bytes[12], bytes[13]]) as u32;
    let height = u16::from_le_bytes([bytes[14], bytes[15]]) as u32;
    let bpp = bytes[16];
    let descriptor = bytes[17];
    if width == 0 || height == 0 {
        return Err(EngineError::InvalidDimensions { width, height });
    }
    let top_down = (descriptor & 0x20) != 0;
    let right_to_left = (descriptor & 0x10) != 0;
    let cmap_bytes = if color_map_type != 0 {
        let len = u16::from_le_bytes([bytes[5], bytes[6]]) as usize;
        let entry_bits = bytes[7] as usize;
        len * entry_bits.div_ceil(8)
    } else {
        0
    };
    let data_offset = 18 + id_len + cmap_bytes;
    if data_offset > bytes.len() {
        return Err(EngineError::DecodeFailed("truncated TGA payload".to_string()));
    }
    let src = &bytes[data_offset..];
    let w = width as usize;
    let h = height as usize;
    let px_count = w * h;
    let channels = match bpp {
        8 => 1usize,
        24 => 3usize,
        32 => 4usize,
        _ => return Err(EngineError::UnsupportedFormat(format!("TGA {bpp}-bit"))),
    };

    let raw = match image_type {
        2 | 3 => {
            let needed = px_count * channels;
            if src.len() < needed {
                return Err(EngineError::DecodeFailed("truncated uncompressed TGA".to_string()));
            }
            src[..needed].to_vec()
        }
        10 | 11 => {
            let mut decoded = Vec::with_capacity(px_count * channels);
            let mut pos = 0usize;
            while decoded.len() < px_count * channels {
                if pos >= src.len() {
                    return Err(EngineError::DecodeFailed("truncated RLE TGA".to_string()));
                }
                let packet = src[pos];
                pos += 1;
                let count = ((packet & 0x7F) as usize) + 1;
                if (packet & 0x80) != 0 {
                    if pos + channels > src.len() {
                        return Err(EngineError::DecodeFailed("truncated RLE TGA run".to_string()));
                    }
                    let px = &src[pos..pos + channels];
                    pos += channels;
                    for _ in 0..count {
                        decoded.extend_from_slice(px);
                    }
                } else {
                    let raw_len = count * channels;
                    if pos + raw_len > src.len() {
                        return Err(EngineError::DecodeFailed("truncated RLE TGA raw".to_string()));
                    }
                    decoded.extend_from_slice(&src[pos..pos + raw_len]);
                    pos += raw_len;
                }
            }
            decoded.truncate(px_count * channels);
            decoded
        }
        _ => {
            return Err(EngineError::UnsupportedFormat(format!(
                "TGA image type {image_type}"
            )));
        }
    };

    let format = match channels {
        1 => PixelFormat::Gray8,
        3 => PixelFormat::Rgb8,
        4 => PixelFormat::Rgba8,
        _ => unreachable!(),
    };
    let color_space = if channels == 1 { ColorSpace::Gray } else { ColorSpace::Srgb };
    let mut out = vec![0u8; px_count * channels];
    for y in 0..h {
        let src_y = if top_down { y } else { h - 1 - y };
        for x in 0..w {
            let src_x = if right_to_left { w - 1 - x } else { x };
            let si = (src_y * w + src_x) * channels;
            let di = (y * w + x) * channels;
            match channels {
                1 => out[di] = raw[si],
                3 => {
                    out[di] = raw[si + 2];
                    out[di + 1] = raw[si + 1];
                    out[di + 2] = raw[si];
                }
                4 => {
                    out[di] = raw[si + 2];
                    out[di + 1] = raw[si + 1];
                    out[di + 2] = raw[si];
                    out[di + 3] = raw[si + 3];
                }
                _ => {}
            }
        }
    }
    PackedImage::new(width, height, format, color_space, out)
}

pub fn encode_tga(img: &PackedImage) -> EngineResult<Vec<u8>> {
    let w = img.width as usize;
    let h = img.height as usize;
    let (image_type, bpp, descriptor, channels) = match img.format {
        PixelFormat::Gray8 | PixelFormat::Gray16 => (3u8, 8u8, 0x20u8, 1usize),
        PixelFormat::Rgba8 | PixelFormat::GrayAlpha8 | PixelFormat::Rgba16 | PixelFormat::GrayAlpha16 | PixelFormat::RgbaF32 => {
            (2u8, 32u8, 0x28u8, 4usize)
        }
        _ => (2u8, 24u8, 0x20u8, 3usize),
    };
    let px_count = w * h;
    let body_len = px_count * channels;
    let footer_len = 26usize;
    let mut out = vec![0u8; 18 + body_len + footer_len];
    out[2] = image_type;
    out[12..14].copy_from_slice(&(img.width as u16).to_le_bytes());
    out[14..16].copy_from_slice(&(img.height as u16).to_le_bytes());
    out[16] = bpp;
    out[17] = descriptor;
    out[18 + body_len + 8..].copy_from_slice(b"TRUEVISION-XFILE.\0");

    let body = &mut out[18..18 + body_len];
    for i in 0..px_count {
        let (r, g, b, a) = read_u8_rgba(img, i);
        match channels {
            1 => body[i] = (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32).round() as u8,
            3 => {
                body[i * 3] = b;
                body[i * 3 + 1] = g;
                body[i * 3 + 2] = r;
            }
            4 => {
                body[i * 4] = b;
                body[i * 4 + 1] = g;
                body[i * 4 + 2] = r;
                body[i * 4 + 3] = a;
            }
            _ => {}
        }
    }
    Ok(out)
}

#[inline]
fn read_u8_rgba(img: &PackedImage, idx: usize) -> (u8, u8, u8, u8) {
    match img.format {
        PixelFormat::Gray8 => {
            let v = img.data[idx];
            (v, v, v, 255)
        }
        PixelFormat::GrayAlpha8 => {
            let v = img.data[idx * 2];
            (v, v, v, img.data[idx * 2 + 1])
        }
        PixelFormat::Rgb8 => (
            img.data[idx * 3],
            img.data[idx * 3 + 1],
            img.data[idx * 3 + 2],
            255,
        ),
        PixelFormat::Rgba8 => (
            img.data[idx * 4],
            img.data[idx * 4 + 1],
            img.data[idx * 4 + 2],
            img.data[idx * 4 + 3],
        ),
        PixelFormat::Gray16 => {
            let v = (u16::from_ne_bytes([img.data[idx * 2], img.data[idx * 2 + 1]]) >> 8) as u8;
            (v, v, v, 255)
        }
        PixelFormat::GrayAlpha16 => {
            let v = (u16::from_ne_bytes([img.data[idx * 4], img.data[idx * 4 + 1]]) >> 8) as u8;
            let a = (u16::from_ne_bytes([img.data[idx * 4 + 2], img.data[idx * 4 + 3]]) >> 8) as u8;
            (v, v, v, a)
        }
        PixelFormat::Rgb16 => {
            let base = idx * 6;
            let r = (u16::from_ne_bytes([img.data[base], img.data[base + 1]]) >> 8) as u8;
            let g = (u16::from_ne_bytes([img.data[base + 2], img.data[base + 3]]) >> 8) as u8;
            let b = (u16::from_ne_bytes([img.data[base + 4], img.data[base + 5]]) >> 8) as u8;
            (r, g, b, 255)
        }
        PixelFormat::Rgba16 => {
            let base = idx * 8;
            let r = (u16::from_ne_bytes([img.data[base], img.data[base + 1]]) >> 8) as u8;
            let g = (u16::from_ne_bytes([img.data[base + 2], img.data[base + 3]]) >> 8) as u8;
            let b = (u16::from_ne_bytes([img.data[base + 4], img.data[base + 5]]) >> 8) as u8;
            let a = (u16::from_ne_bytes([img.data[base + 6], img.data[base + 7]]) >> 8) as u8;
            (r, g, b, a)
        }
        PixelFormat::RgbF32 => {
            let base = idx * 12;
            let r = (f32::from_ne_bytes([img.data[base], img.data[base + 1], img.data[base + 2], img.data[base + 3]]).clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (f32::from_ne_bytes([img.data[base + 4], img.data[base + 5], img.data[base + 6], img.data[base + 7]]).clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (f32::from_ne_bytes([img.data[base + 8], img.data[base + 9], img.data[base + 10], img.data[base + 11]]).clamp(0.0, 1.0) * 255.0).round() as u8;
            (r, g, b, 255)
        }
        PixelFormat::RgbaF32 => {
            let base = idx * 16;
            let r = (f32::from_ne_bytes([img.data[base], img.data[base + 1], img.data[base + 2], img.data[base + 3]]).clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (f32::from_ne_bytes([img.data[base + 4], img.data[base + 5], img.data[base + 6], img.data[base + 7]]).clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (f32::from_ne_bytes([img.data[base + 8], img.data[base + 9], img.data[base + 10], img.data[base + 11]]).clamp(0.0, 1.0) * 255.0).round() as u8;
            let a = (f32::from_ne_bytes([img.data[base + 12], img.data[base + 13], img.data[base + 14], img.data[base + 15]]).clamp(0.0, 1.0) * 255.0).round() as u8;
            (r, g, b, a)
        }
    }
}
