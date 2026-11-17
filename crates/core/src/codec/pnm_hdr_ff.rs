use crate::color::ColorSpace;
use crate::error::{EngineError, EngineResult};
use crate::pixel::{PackedImage, PixelFormat};

pub fn decode_farbfeld(bytes: &[u8]) -> EngineResult<PackedImage> {
    if bytes.len() < 16 || &bytes[0..8] != b"farbfeld" {
        return Err(EngineError::DecodeFailed("invalid farbfeld header".to_string()));
    }
    let width = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let height = u32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    if width == 0 || height == 0 {
        return Err(EngineError::InvalidDimensions { width, height });
    }
    let count = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(8))
        .ok_or(EngineError::InvalidDimensions { width, height })?;
    if bytes.len() < 16 + count {
        return Err(EngineError::DecodeFailed("truncated farbfeld payload".to_string()));
    }
    let mut out = vec![0u8; count];
    for (dst, src) in out
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(bytes[16..16 + count].as_chunks::<2>().0)
    {
        let v = u16::from_be_bytes(*src);
        *dst = v.to_ne_bytes();
    }
    PackedImage::new(width, height, PixelFormat::Rgba16, ColorSpace::Srgb, out)
}

pub fn encode_farbfeld(img: &PackedImage) -> EngineResult<Vec<u8>> {
    let px = img.pixel_count();
    let mut out = Vec::with_capacity(16 + px * 8);
    out.extend_from_slice(b"farbfeld");
    out.extend_from_slice(&img.width.to_be_bytes());
    out.extend_from_slice(&img.height.to_be_bytes());
    for i in 0..px {
        let (r, g, b, a) = read_u16_rgba(img, i);
        out.extend_from_slice(&r.to_be_bytes());
        out.extend_from_slice(&g.to_be_bytes());
        out.extend_from_slice(&b.to_be_bytes());
        out.extend_from_slice(&a.to_be_bytes());
    }
    Ok(out)
}

pub fn decode_pnm(bytes: &[u8]) -> EngineResult<PackedImage> {
    if bytes.len() < 3 || bytes[0] != b'P' {
        return Err(EngineError::DecodeFailed("invalid PNM magic".to_string()));
    }
    let kind = bytes[1];
    let mut pos = 2usize;
    if kind == b'7' {
        return decode_pam(bytes, pos);
    }
    let width: u32 = next_pnm_token(bytes, &mut pos)?
        .parse()
        .map_err(|_| EngineError::DecodeFailed("invalid PNM width".to_string()))?;
    let height: u32 = next_pnm_token(bytes, &mut pos)?
        .parse()
        .map_err(|_| EngineError::DecodeFailed("invalid PNM height".to_string()))?;
    if width == 0 || height == 0 {
        return Err(EngineError::InvalidDimensions { width, height });
    }
    let w = width as usize;
    let h = height as usize;

    match kind {
        b'1' => {
            let mut out = vec![0u8; w * h];
            for slot in &mut out {
                let tok = next_pnm_token(bytes, &mut pos)?;
                *slot = if tok == "1" { 0 } else { 255 };
            }
            PackedImage::new(width, height, PixelFormat::Gray8, ColorSpace::Gray, out)
        }
        b'4' => {
            if pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            let row_bytes = w.div_ceil(8);
            if bytes.len() < pos + row_bytes * h {
                return Err(EngineError::DecodeFailed("truncated P4 PBM".to_string()));
            }
            let mut out = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    let byte = bytes[pos + y * row_bytes + (x / 8)];
                    let bit = (byte >> (7 - (x % 8))) & 1;
                    out[y * w + x] = if bit == 1 { 0 } else { 255 };
                }
            }
            PackedImage::new(width, height, PixelFormat::Gray8, ColorSpace::Gray, out)
        }
        b'2' | b'3' | b'5' | b'6' => {
            let maxval: u32 = next_pnm_token(bytes, &mut pos)?
                .parse()
                .map_err(|_| EngineError::DecodeFailed("invalid PNM maxval".to_string()))?;
            if maxval == 0 || maxval > 65535 {
                return Err(EngineError::DecodeFailed("unsupported PNM maxval".to_string()));
            }
            let channels = if kind == b'2' || kind == b'5' { 1 } else { 3 };
            let samples = w * h * channels;
            let is_16 = maxval > 255;
            if kind == b'2' || kind == b'3' {
                if is_16 {
                    let mut out = vec![0u8; samples * 2];
                    for i in 0..samples {
                        let v: u32 = next_pnm_token(bytes, &mut pos)?
                            .parse()
                            .map_err(|_| EngineError::DecodeFailed("invalid ASCII PNM sample".to_string()))?;
                        let scaled = ((v.min(maxval) * 65535) / maxval) as u16;
                        out[i * 2..i * 2 + 2].copy_from_slice(&scaled.to_ne_bytes());
                    }
                    let fmt = if channels == 1 { PixelFormat::Gray16 } else { PixelFormat::Rgb16 };
                    let cs = if channels == 1 { ColorSpace::Gray } else { ColorSpace::Srgb };
                    PackedImage::new(width, height, fmt, cs, out)
                } else {
                    let mut out = vec![0u8; samples];
                    for slot in &mut out {
                        let v: u32 = next_pnm_token(bytes, &mut pos)?
                            .parse()
                            .map_err(|_| EngineError::DecodeFailed("invalid ASCII PNM sample".to_string()))?;
                        *slot = ((v.min(maxval) * 255) / maxval) as u8;
                    }
                    let fmt = if channels == 1 { PixelFormat::Gray8 } else { PixelFormat::Rgb8 };
                    let cs = if channels == 1 { ColorSpace::Gray } else { ColorSpace::Srgb };
                    PackedImage::new(width, height, fmt, cs, out)
                }
            } else {
                if pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                    pos += 1;
                }
                if is_16 {
                    if bytes.len() < pos + samples * 2 {
                        return Err(EngineError::DecodeFailed("truncated 16-bit binary PNM".to_string()));
                    }
                    let mut out = vec![0u8; samples * 2];
                    for (i, chunk) in bytes[pos..pos + samples * 2].as_chunks::<2>().0.iter().enumerate() {
                        let raw = u16::from_be_bytes(*chunk) as u32;
                        let scaled = ((raw.min(maxval) * 65535) / maxval) as u16;
                        out[i * 2..i * 2 + 2].copy_from_slice(&scaled.to_ne_bytes());
                    }
                    let fmt = if channels == 1 { PixelFormat::Gray16 } else { PixelFormat::Rgb16 };
                    let cs = if channels == 1 { ColorSpace::Gray } else { ColorSpace::Srgb };
                    PackedImage::new(width, height, fmt, cs, out)
                } else {
                    if bytes.len() < pos + samples {
                        return Err(EngineError::DecodeFailed("truncated 8-bit binary PNM".to_string()));
                    }
                    let mut out = vec![0u8; samples];
                    for (dst, &v) in out.iter_mut().zip(&bytes[pos..pos + samples]) {
                        *dst = (((v as u32).min(maxval) * 255) / maxval) as u8;
                    }
                    let fmt = if channels == 1 { PixelFormat::Gray8 } else { PixelFormat::Rgb8 };
                    let cs = if channels == 1 { ColorSpace::Gray } else { ColorSpace::Srgb };
                    PackedImage::new(width, height, fmt, cs, out)
                }
            }
        }
        _ => Err(EngineError::UnsupportedFormat(format!("PNM P{}", kind as char))),
    }
}

fn decode_pam(bytes: &[u8], mut pos: usize) -> EngineResult<PackedImage> {
    let mut width = 0u32;
    let mut height = 0u32;
    let mut depth = 0u32;
    let mut maxval = 0u32;
    let mut found_end = false;
    while pos < bytes.len() {
        let line_end = bytes[pos..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|p| pos + p)
            .unwrap_or(bytes.len());
        let line = std::str::from_utf8(&bytes[pos..line_end])
            .map_err(|_| EngineError::DecodeFailed("non-UTF8 PAM header".to_string()))?
            .trim();
        pos = (line_end + 1).min(bytes.len());
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "ENDHDR" {
            found_end = true;
            break;
        }
        let mut parts = line.split_whitespace();
        match (parts.next(), parts.next()) {
            (Some("WIDTH"), Some(v)) => width = v.parse().unwrap_or(0),
            (Some("HEIGHT"), Some(v)) => height = v.parse().unwrap_or(0),
            (Some("DEPTH"), Some(v)) => depth = v.parse().unwrap_or(0),
            (Some("MAXVAL"), Some(v)) => maxval = v.parse().unwrap_or(0),
            _ => {}
        }
    }
    if !found_end || width == 0 || height == 0 || maxval == 0 || maxval > 65535 {
        return Err(EngineError::DecodeFailed("invalid PAM header".to_string()));
    }
    let format = match (depth, maxval > 255) {
        (1, false) => PixelFormat::Gray8,
        (2, false) => PixelFormat::GrayAlpha8,
        (3, false) => PixelFormat::Rgb8,
        (4, false) => PixelFormat::Rgba8,
        (1, true) => PixelFormat::Gray16,
        (2, true) => PixelFormat::GrayAlpha16,
        (3, true) => PixelFormat::Rgb16,
        (4, true) => PixelFormat::Rgba16,
        _ => return Err(EngineError::UnsupportedFormat(format!("PAM depth {depth}"))),
    };
    let samples = (width as usize) * (height as usize) * (depth as usize);
    let cs = if depth <= 2 { ColorSpace::Gray } else { ColorSpace::Srgb };
    if maxval > 255 {
        if bytes.len() < pos + samples * 2 {
            return Err(EngineError::DecodeFailed("truncated 16-bit PAM".to_string()));
        }
        let mut out = vec![0u8; samples * 2];
        for (i, c) in bytes[pos..pos + samples * 2].as_chunks::<2>().0.iter().enumerate() {
            let v = ((u16::from_be_bytes(*c) as u32).min(maxval) * 65535) / maxval;
            out[i * 2..i * 2 + 2].copy_from_slice(&(v as u16).to_ne_bytes());
        }
        PackedImage::new(width, height, format, cs, out)
    } else {
        if bytes.len() < pos + samples {
            return Err(EngineError::DecodeFailed("truncated 8-bit PAM".to_string()));
        }
        let mut out = vec![0u8; samples];
        for (dst, &b) in out.iter_mut().zip(&bytes[pos..pos + samples]) {
            *dst = (((b as u32).min(maxval) * 255) / maxval) as u8;
        }
        PackedImage::new(width, height, format, cs, out)
    }
}

pub fn encode_pnm(img: &PackedImage) -> EngineResult<Vec<u8>> {
    let px = img.pixel_count();
    match img.format {
        PixelFormat::Gray8 => {
            let mut out = format!("P5\n{} {}\n255\n", img.width, img.height).into_bytes();
            out.extend_from_slice(&img.data);
            Ok(out)
        }
        PixelFormat::Rgb8 => {
            let mut out = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
            out.extend_from_slice(&img.data);
            Ok(out)
        }
        PixelFormat::Gray16 => {
            let mut out = format!("P5\n{} {}\n65535\n", img.width, img.height).into_bytes();
            for c in img.data.as_chunks::<2>().0 {
                let v = u16::from_ne_bytes(*c);
                out.extend_from_slice(&v.to_be_bytes());
            }
            Ok(out)
        }
        PixelFormat::Rgb16 => {
            let mut out = format!("P6\n{} {}\n65535\n", img.width, img.height).into_bytes();
            for c in img.data.as_chunks::<2>().0 {
                let v = u16::from_ne_bytes(*c);
                out.extend_from_slice(&v.to_be_bytes());
            }
            Ok(out)
        }
        PixelFormat::GrayAlpha8 | PixelFormat::Rgba8 => {
            let depth = if img.format == PixelFormat::GrayAlpha8 { 2 } else { 4 };
            let tuple = if depth == 2 { "GRAYSCALE_ALPHA" } else { "RGB_ALPHA" };
            let mut out = format!(
                "P7\nWIDTH {}\nHEIGHT {}\nDEPTH {}\nMAXVAL 255\nTUPLTYPE {}\nENDHDR\n",
                img.width, img.height, depth, tuple
            )
            .into_bytes();
            out.extend_from_slice(&img.data);
            Ok(out)
        }
        PixelFormat::GrayAlpha16 | PixelFormat::Rgba16 => {
            let depth = if img.format == PixelFormat::GrayAlpha16 { 2 } else { 4 };
            let tuple = if depth == 2 { "GRAYSCALE_ALPHA" } else { "RGB_ALPHA" };
            let mut out = format!(
                "P7\nWIDTH {}\nHEIGHT {}\nDEPTH {}\nMAXVAL 65535\nTUPLTYPE {}\nENDHDR\n",
                img.width, img.height, depth, tuple
            )
            .into_bytes();
            for c in img.data.as_chunks::<2>().0 {
                let v = u16::from_ne_bytes(*c);
                out.extend_from_slice(&v.to_be_bytes());
            }
            Ok(out)
        }
        PixelFormat::RgbF32 | PixelFormat::RgbaF32 => {
            let mut out = format!(
                "P7\nWIDTH {}\nHEIGHT {}\nDEPTH 4\nMAXVAL 65535\nTUPLTYPE RGB_ALPHA\nENDHDR\n",
                img.width, img.height
            )
            .into_bytes();
            for i in 0..px {
                let (r, g, b, a) = read_u16_rgba(img, i);
                out.extend_from_slice(&r.to_be_bytes());
                out.extend_from_slice(&g.to_be_bytes());
                out.extend_from_slice(&b.to_be_bytes());
                out.extend_from_slice(&a.to_be_bytes());
            }
            Ok(out)
        }
    }
}

pub fn decode_radiance_hdr(bytes: &[u8]) -> EngineResult<PackedImage> {
    if !bytes.starts_with(b"#?RADIANCE") && !bytes.starts_with(b"#?RGBE") {
        return Err(EngineError::DecodeFailed("invalid Radiance HDR magic".to_string()));
    }
    let mut pos = 0usize;
    let mut found_blank = false;
    while pos < bytes.len() {
        let end = bytes[pos..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|p| pos + p)
            .unwrap_or(bytes.len());
        let line = std::str::from_utf8(&bytes[pos..end]).unwrap_or("").trim();
        pos = (end + 1).min(bytes.len());
        if line.is_empty() {
            found_blank = true;
            break;
        }
    }
    if !found_blank || pos >= bytes.len() {
        return Err(EngineError::DecodeFailed("truncated HDR header".to_string()));
    }
    let res_end = bytes[pos..]
        .iter()
        .position(|&b| b == b'\n')
        .map(|p| pos + p)
        .unwrap_or(bytes.len());
    let res_line = std::str::from_utf8(&bytes[pos..res_end])
        .map_err(|_| EngineError::DecodeFailed("invalid HDR resolution line".to_string()))?
        .trim();
    pos = (res_end + 1).min(bytes.len());
    let tokens: Vec<&str> = res_line.split_whitespace().collect();
    if tokens.len() != 4 || tokens[0] != "-Y" || tokens[2] != "+X" {
        return Err(EngineError::UnsupportedFormat(format!(
            "HDR orientation: {res_line}"
        )));
    }
    let height: u32 = tokens[1]
        .parse()
        .map_err(|_| EngineError::DecodeFailed("invalid HDR height".to_string()))?;
    let width: u32 = tokens[3]
        .parse()
        .map_err(|_| EngineError::DecodeFailed("invalid HDR width".to_string()))?;
    if width == 0 || height == 0 {
        return Err(EngineError::InvalidDimensions { width, height });
    }
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u8; w * h * 12];
    let mut scanline = vec![0u8; w * 4];

    for y in 0..h {
        if pos + 4 > bytes.len() {
            return Err(EngineError::DecodeFailed("truncated HDR scanline".to_string()));
        }
        let b0 = bytes[pos];
        let b1 = bytes[pos + 1];
        let b2 = bytes[pos + 2];
        let b3 = bytes[pos + 3];
        if b0 == 2 && b1 == 2 && (((b2 as usize) << 8) | (b3 as usize)) == w && (8..=32767).contains(&w) {
            pos += 4;
            for ch in 0..4 {
                let mut x = 0usize;
                while x < w {
                    if pos >= bytes.len() {
                        return Err(EngineError::DecodeFailed("truncated HDR RLE".to_string()));
                    }
                    let count = bytes[pos];
                    pos += 1;
                    if count > 128 {
                        let run = (count - 128) as usize;
                        if pos >= bytes.len() || x + run > w {
                            return Err(EngineError::DecodeFailed("corrupt HDR RLE run".to_string()));
                        }
                        let val = bytes[pos];
                        pos += 1;
                        for k in 0..run {
                            scanline[(x + k) * 4 + ch] = val;
                        }
                        x += run;
                    } else {
                        let run = count as usize;
                        if run == 0 || pos + run > bytes.len() || x + run > w {
                            return Err(EngineError::DecodeFailed("corrupt HDR RLE literal".to_string()));
                        }
                        for k in 0..run {
                            scanline[(x + k) * 4 + ch] = bytes[pos + k];
                        }
                        pos += run;
                        x += run;
                    }
                }
            }
        } else {
            let row_len = w * 4;
            if pos + row_len > bytes.len() {
                return Err(EngineError::DecodeFailed("truncated raw HDR row".to_string()));
            }
            scanline.copy_from_slice(&bytes[pos..pos + row_len]);
            pos += row_len;
        }
        for x in 0..w {
            let r8 = scanline[x * 4];
            let g8 = scanline[x * 4 + 1];
            let b8 = scanline[x * 4 + 2];
            let e8 = scanline[x * 4 + 3];
            let (rf, gf, bf) = if e8 == 0 {
                (0.0f32, 0.0f32, 0.0f32)
            } else {
                let scale = 2.0f32.powi((e8 as i32) - 136);
                (
                    (r8 as f32 + 0.5) * scale,
                    (g8 as f32 + 0.5) * scale,
                    (b8 as f32 + 0.5) * scale,
                )
            };
            let base = (y * w + x) * 12;
            out[base..base + 4].copy_from_slice(&rf.to_ne_bytes());
            out[base + 4..base + 8].copy_from_slice(&gf.to_ne_bytes());
            out[base + 8..base + 12].copy_from_slice(&bf.to_ne_bytes());
        }
    }
    PackedImage::new(width, height, PixelFormat::RgbF32, ColorSpace::LinearSrgb, out)
}

pub fn encode_radiance_hdr(img: &PackedImage) -> EngineResult<Vec<u8>> {
    let w = img.width as usize;
    let h = img.height as usize;
    let mut out = format!(
        "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {} +X {}\n",
        img.height, img.width
    )
    .into_bytes();
    for y in 0..h {
        for x in 0..w {
            let (rf, gf, bf) = read_linear_f32_rgb(img, y * w + x);
            let peak = rf.max(gf).max(bf);
            if peak < 1e-32 {
                out.extend_from_slice(&[0, 0, 0, 0]);
            } else {
                let exp = peak.log2().floor() as i32 + 1;
                let scale = 2.0f32.powi(8 - exp);
                let r8 = (rf * scale).clamp(0.0, 255.0) as u8;
                let g8 = (gf * scale).clamp(0.0, 255.0) as u8;
                let b8 = (bf * scale).clamp(0.0, 255.0) as u8;
                let e8 = (exp + 128).clamp(0, 255) as u8;
                out.extend_from_slice(&[r8, g8, b8, e8]);
            }
        }
    }
    Ok(out)
}

fn next_pnm_token<'a>(bytes: &'a [u8], pos: &mut usize) -> EngineResult<&'a str> {
    while *pos < bytes.len() {
        let b = bytes[*pos];
        if b.is_ascii_whitespace() {
            *pos += 1;
        } else if b == b'#' {
            while *pos < bytes.len() && bytes[*pos] != b'\n' {
                *pos += 1;
            }
        } else {
            break;
        }
    }
    let start = *pos;
    while *pos < bytes.len() && !bytes[*pos].is_ascii_whitespace() && bytes[*pos] != b'#' {
        *pos += 1;
    }
    if start == *pos {
        return Err(EngineError::DecodeFailed("unexpected EOF in PNM header".to_string()));
    }
    std::str::from_utf8(&bytes[start..*pos])
        .map_err(|_| EngineError::DecodeFailed("non-UTF8 PNM token".to_string()))
}

#[inline]
fn read_u16_rgba(img: &PackedImage, idx: usize) -> (u16, u16, u16, u16) {
    match img.format {
        PixelFormat::Gray8 => {
            let v = (img.data[idx] as u16) * 257;
            (v, v, v, 65535)
        }
        PixelFormat::GrayAlpha8 => {
            let v = (img.data[idx * 2] as u16) * 257;
            let a = (img.data[idx * 2 + 1] as u16) * 257;
            (v, v, v, a)
        }
        PixelFormat::Rgb8 => (
            (img.data[idx * 3] as u16) * 257,
            (img.data[idx * 3 + 1] as u16) * 257,
            (img.data[idx * 3 + 2] as u16) * 257,
            65535,
        ),
        PixelFormat::Rgba8 => (
            (img.data[idx * 4] as u16) * 257,
            (img.data[idx * 4 + 1] as u16) * 257,
            (img.data[idx * 4 + 2] as u16) * 257,
            (img.data[idx * 4 + 3] as u16) * 257,
        ),
        PixelFormat::Gray16 => {
            let v = u16::from_ne_bytes([img.data[idx * 2], img.data[idx * 2 + 1]]);
            (v, v, v, 65535)
        }
        PixelFormat::GrayAlpha16 => {
            let v = u16::from_ne_bytes([img.data[idx * 4], img.data[idx * 4 + 1]]);
            let a = u16::from_ne_bytes([img.data[idx * 4 + 2], img.data[idx * 4 + 3]]);
            (v, v, v, a)
        }
        PixelFormat::Rgb16 => {
            let b = idx * 6;
            (
                u16::from_ne_bytes([img.data[b], img.data[b + 1]]),
                u16::from_ne_bytes([img.data[b + 2], img.data[b + 3]]),
                u16::from_ne_bytes([img.data[b + 4], img.data[b + 5]]),
                65535,
            )
        }
        PixelFormat::Rgba16 => {
            let b = idx * 8;
            (
                u16::from_ne_bytes([img.data[b], img.data[b + 1]]),
                u16::from_ne_bytes([img.data[b + 2], img.data[b + 3]]),
                u16::from_ne_bytes([img.data[b + 4], img.data[b + 5]]),
                u16::from_ne_bytes([img.data[b + 6], img.data[b + 7]]),
            )
        }
        PixelFormat::RgbF32 => {
            let base = idx * 12;
            let r = (f32::from_ne_bytes([img.data[base], img.data[base + 1], img.data[base + 2], img.data[base + 3]]).clamp(0.0, 1.0) * 65535.0).round() as u16;
            let g = (f32::from_ne_bytes([img.data[base + 4], img.data[base + 5], img.data[base + 6], img.data[base + 7]]).clamp(0.0, 1.0) * 65535.0).round() as u16;
            let b = (f32::from_ne_bytes([img.data[base + 8], img.data[base + 9], img.data[base + 10], img.data[base + 11]]).clamp(0.0, 1.0) * 65535.0).round() as u16;
            (r, g, b, 65535)
        }
        PixelFormat::RgbaF32 => {
            let base = idx * 16;
            let r = (f32::from_ne_bytes([img.data[base], img.data[base + 1], img.data[base + 2], img.data[base + 3]]).clamp(0.0, 1.0) * 65535.0).round() as u16;
            let g = (f32::from_ne_bytes([img.data[base + 4], img.data[base + 5], img.data[base + 6], img.data[base + 7]]).clamp(0.0, 1.0) * 65535.0).round() as u16;
            let b = (f32::from_ne_bytes([img.data[base + 8], img.data[base + 9], img.data[base + 10], img.data[base + 11]]).clamp(0.0, 1.0) * 65535.0).round() as u16;
            let a = (f32::from_ne_bytes([img.data[base + 12], img.data[base + 13], img.data[base + 14], img.data[base + 15]]).clamp(0.0, 1.0) * 65535.0).round() as u16;
            (r, g, b, a)
        }
    }
}

#[inline]
fn read_linear_f32_rgb(img: &PackedImage, idx: usize) -> (f32, f32, f32) {
    match img.format {
        PixelFormat::RgbF32 => {
            let b = idx * 12;
            (
                f32::from_ne_bytes([img.data[b], img.data[b + 1], img.data[b + 2], img.data[b + 3]]),
                f32::from_ne_bytes([img.data[b + 4], img.data[b + 5], img.data[b + 6], img.data[b + 7]]),
                f32::from_ne_bytes([img.data[b + 8], img.data[b + 9], img.data[b + 10], img.data[b + 11]]),
            )
        }
        PixelFormat::RgbaF32 => {
            let b = idx * 16;
            (
                f32::from_ne_bytes([img.data[b], img.data[b + 1], img.data[b + 2], img.data[b + 3]]),
                f32::from_ne_bytes([img.data[b + 4], img.data[b + 5], img.data[b + 6], img.data[b + 7]]),
                f32::from_ne_bytes([img.data[b + 8], img.data[b + 9], img.data[b + 10], img.data[b + 11]]),
            )
        }
        _ => {
            let (r, g, b, _) = read_u16_rgba(img, idx);
            (
                crate::color::srgb_to_linear_f32(r as f32 / 65535.0),
                crate::color::srgb_to_linear_f32(g as f32 / 65535.0),
                crate::color::srgb_to_linear_f32(b as f32 / 65535.0),
            )
        }
    }
}
