use crate::codec::ImageFormat;
use crate::color::ColorSpace;
use crate::error::{EngineError, EngineResult};
use crate::pixel::{Frame, FrameMeta, FrameSequence, PackedImage, PixelFormat};
use std::io::Cursor;

pub fn decode_animation(bytes: &[u8], format: ImageFormat, max_pixels: u64) -> EngineResult<FrameSequence> {
    match format {
        ImageFormat::Gif => decode_gif_frames(bytes, max_pixels),
        ImageFormat::Png => decode_apng_frames(bytes, max_pixels),
        ImageFormat::Webp => decode_webp_frames(bytes, max_pixels),
        _ => Err(EngineError::UnsupportedFormat(
            "animation decoding supported for GIF, APNG, and animated WebP".to_string(),
        )),
    }
}

pub fn encode_animation(seq: &FrameSequence, format: ImageFormat) -> EngineResult<Vec<u8>> {
    if seq.frames.is_empty() {
        return Err(EngineError::EncodeFailed("cannot encode empty frame sequence".to_string()));
    }
    match format {
        ImageFormat::Gif => encode_gif_frames(seq),
        ImageFormat::Png => encode_apng_frames(seq),
        _ => Err(EngineError::UnsupportedFormat(
            "animation encoding supported for GIF and APNG".to_string(),
        )),
    }
}

fn decode_gif_frames(bytes: &[u8], max_pixels: u64) -> EngineResult<FrameSequence> {
    let mut gopts = gif::DecodeOptions::new();
    gopts.set_color_output(gif::ColorOutput::RGBA);
    let mut dec = gopts
        .read_info(Cursor::new(bytes))
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
    let cw = dec.width() as u32;
    let ch = dec.height() as u32;
    if cw == 0 || ch == 0 || (max_pixels > 0 && (cw as u64) * (ch as u64) > max_pixels) {
        return Err(EngineError::InvalidDimensions {
            width: cw,
            height: ch,
        });
    }

    let canvas_len = (cw as usize) * (ch as usize) * 4;
    let mut canvas = vec![0u8; canvas_len];
    let mut prev_canvas = vec![0u8; canvas_len];
    let mut frames = Vec::new();

    while let Some(raw_frame) = dec
        .read_next_frame()
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?
    {
        prev_canvas.copy_from_slice(&canvas);
        let fw = raw_frame.width as usize;
        let fh = raw_frame.height as usize;
        let fx = raw_frame.left as usize;
        let fy = raw_frame.top as usize;

        for row in 0..fh {
            let dy = fy + row;
            if dy >= ch as usize {
                break;
            }
            for col in 0..fw {
                let dx = fx + col;
                if dx >= cw as usize {
                    break;
                }
                let si = (row * fw + col) * 4;
                let di = (dy * cw as usize + dx) * 4;
                let sa = raw_frame.buffer[si + 3];
                if sa > 0 {
                    canvas[di..di + 4].copy_from_slice(&raw_frame.buffer[si..si + 4]);
                }
            }
        }

        let img = PackedImage::new(cw, ch, PixelFormat::Rgba8, ColorSpace::Srgb, canvas.clone())?;
        frames.push(Frame {
            image: img,
            meta: FrameMeta {
                duration_ms: (raw_frame.delay as u32).saturating_mul(10).max(20),
                left: 0,
                top: 0,
            },
        });

        match raw_frame.dispose {
            gif::DisposalMethod::Background => {
                for row in 0..fh {
                    let dy = fy + row;
                    if dy >= ch as usize {
                        break;
                    }
                    for col in 0..fw {
                        let dx = fx + col;
                        if dx >= cw as usize {
                            break;
                        }
                        let di = (dy * cw as usize + dx) * 4;
                        canvas[di..di + 4].copy_from_slice(&[0, 0, 0, 0]);
                    }
                }
            }
            gif::DisposalMethod::Previous => {
                canvas.copy_from_slice(&prev_canvas);
            }
            _ => {}
        }
    }

    Ok(FrameSequence {
        canvas_width: cw,
        canvas_height: ch,
        loop_count: 0,
        frames,
    })
}

fn decode_apng_frames(bytes: &[u8], max_pixels: u64) -> EngineResult<FrameSequence> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder
        .read_info()
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
    let cw = reader.info().width;
    let ch = reader.info().height;
    if cw == 0 || ch == 0 || (max_pixels > 0 && (cw as u64) * (ch as u64) > max_pixels) {
        return Err(EngineError::InvalidDimensions {
            width: cw,
            height: ch,
        });
    }
    let loop_count = reader
        .info()
        .animation_control()
        .map(|a| a.num_plays)
        .unwrap_or(0);
    let total_frames = reader
        .info()
        .animation_control()
        .map(|a| a.num_frames)
        .unwrap_or(1);

    let canvas_len = (cw as usize) * (ch as usize) * 4;
    let mut canvas = vec![0u8; canvas_len];
    let mut prev_canvas = vec![0u8; canvas_len];
    let mut frames = Vec::with_capacity(total_frames as usize);

    for idx in 0..total_frames {
        let fctl = reader.info().frame_control().copied();
        let buf_len = reader
            .output_buffer_size()
            .ok_or_else(|| EngineError::DecodeFailed("APNG frame buffer overflow".to_string()))?;
        let mut buf = vec![0u8; buf_len];
        let info = match reader.next_frame(&mut buf) {
            Ok(i) => i,
            Err(_) if idx > 0 => break,
            Err(e) => return Err(EngineError::DecodeFailed(e.to_string())),
        };
        buf.truncate(info.buffer_size());
        let rgba_sub = to_rgba8_frame(&buf, info.color_type, info.bit_depth);

        let (ox, oy, fw, fh, delay_ms, dispose, blend) = if let Some(fc) = fctl {
            let den = if fc.delay_den == 0 { 100 } else { fc.delay_den as u32 };
            let ms = ((fc.delay_num as u32) * 1000 / den).max(10);
            (
                fc.x_offset as usize,
                fc.y_offset as usize,
                fc.width as usize,
                fc.height as usize,
                ms,
                fc.dispose_op,
                fc.blend_op,
            )
        } else {
            (
                0,
                0,
                info.width as usize,
                info.height as usize,
                100,
                png::DisposeOp::None,
                png::BlendOp::Source,
            )
        };

        prev_canvas.copy_from_slice(&canvas);
        for row in 0..fh.min(info.height as usize) {
            let dy = oy + row;
            if dy >= ch as usize {
                break;
            }
            for col in 0..fw.min(info.width as usize) {
                let dx = ox + col;
                if dx >= cw as usize {
                    break;
                }
                let si = (row * info.width as usize + col) * 4;
                let di = (dy * cw as usize + dx) * 4;
                let src_px = &rgba_sub[si..si + 4];
                if matches!(blend, png::BlendOp::Source) || src_px[3] == 255 {
                    canvas[di..di + 4].copy_from_slice(src_px);
                } else if src_px[3] > 0 {
                    let sa = src_px[3] as f32 / 255.0;
                    let da = canvas[di + 3] as f32 / 255.0;
                    let out_a = sa + da * (1.0 - sa);
                    if out_a > 1e-4 {
                        for c in 0..3 {
                            let val = (src_px[c] as f32 * sa + canvas[di + c] as f32 * da * (1.0 - sa)) / out_a;
                            canvas[di + c] = val.round().clamp(0.0, 255.0) as u8;
                        }
                        canvas[di + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }

        let img = PackedImage::new(cw, ch, PixelFormat::Rgba8, ColorSpace::Srgb, canvas.clone())?;
        frames.push(Frame {
            image: img,
            meta: FrameMeta {
                duration_ms: delay_ms,
                left: 0,
                top: 0,
            },
        });

        match dispose {
            png::DisposeOp::Background => {
                for row in 0..fh {
                    let dy = oy + row;
                    if dy >= ch as usize {
                        break;
                    }
                    for col in 0..fw {
                        let dx = ox + col;
                        if dx >= cw as usize {
                            break;
                        }
                        let di = (dy * cw as usize + dx) * 4;
                        canvas[di..di + 4].copy_from_slice(&[0, 0, 0, 0]);
                    }
                }
            }
            png::DisposeOp::Previous => {
                canvas.copy_from_slice(&prev_canvas);
            }
            png::DisposeOp::None => {}
        }
    }

    Ok(FrameSequence {
        canvas_width: cw,
        canvas_height: ch,
        loop_count,
        frames,
    })
}

fn decode_webp_frames(bytes: &[u8], max_pixels: u64) -> EngineResult<FrameSequence> {
    let mut dec = image_webp::WebPDecoder::new(Cursor::new(bytes))
        .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
    let (cw, ch) = dec.dimensions();
    if cw == 0 || ch == 0 || (max_pixels > 0 && (cw as u64) * (ch as u64) > max_pixels) {
        return Err(EngineError::InvalidDimensions {
            width: cw,
            height: ch,
        });
    }
    let count = dec.num_frames().max(1);
    let bpp = if dec.has_alpha() { 4 } else { 3 };
    let frame_len = (cw as usize) * (ch as usize) * bpp;
    let mut buf = vec![0u8; frame_len];
    let mut frames = Vec::with_capacity(count as usize);

    for _ in 0..count {
        let duration_ms = dec
            .read_frame(&mut buf)
            .map_err(|e| EngineError::DecodeFailed(e.to_string()))?;
        let fmt = if bpp == 4 { PixelFormat::Rgba8 } else { PixelFormat::Rgb8 };
        let img = PackedImage::new(cw, ch, fmt, ColorSpace::Srgb, buf.clone())?;
        frames.push(Frame {
            image: img,
            meta: FrameMeta {
                duration_ms: duration_ms.max(10),
                left: 0,
                top: 0,
            },
        });
    }
    let loops = match dec.loop_count() {
        image_webp::LoopCount::Forever => 0,
        image_webp::LoopCount::Times(n) => n.get() as u32,
    };

    Ok(FrameSequence {
        canvas_width: cw,
        canvas_height: ch,
        loop_count: loops,
        frames,
    })
}

fn encode_gif_frames(seq: &FrameSequence) -> EngineResult<Vec<u8>> {
    let w = seq.canvas_width.min(u16::MAX as u32) as u16;
    let h = seq.canvas_height.min(u16::MAX as u32) as u16;
    let mut out = Vec::new();
    {
        let mut enc = gif::Encoder::new(&mut out, w, h, &[])
            .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
        let rep = if seq.loop_count == 0 {
            gif::Repeat::Infinite
        } else {
            gif::Repeat::Finite(seq.loop_count.min(u16::MAX as u32) as u16)
        };
        enc.set_repeat(rep)
            .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
        for f in &seq.frames {
            let mut rgba = crate::codec::to_rgba8_bytes(&f.image);
            let mut gframe = gif::Frame::from_rgba_speed(w, h, &mut rgba, 10);
            gframe.delay = (f.meta.duration_ms.div_ceil(10)).clamp(1, u16::MAX as u32) as u16;
            gframe.dispose = gif::DisposalMethod::Background;
            enc.write_frame(&gframe)
                .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
        }
    }
    Ok(out)
}

fn encode_apng_frames(seq: &FrameSequence) -> EngineResult<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, seq.canvas_width, seq.canvas_height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_animated(seq.frames.len() as u32, seq.loop_count)
            .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
        let mut writer = enc
            .write_header()
            .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
        for f in &seq.frames {
            let delay_ms = f.meta.duration_ms.min(u16::MAX as u32) as u16;
            writer
                .set_frame_delay(delay_ms.max(10), 1000)
                .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
            writer
                .set_dispose_op(png::DisposeOp::Background)
                .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
            let rgba = crate::codec::to_rgba8_bytes(&f.image);
            writer
                .write_image_data(&rgba)
                .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
        }
    }
    Ok(out)
}

fn to_rgba8_frame(buf: &[u8], ct: png::ColorType, bd: png::BitDepth) -> Vec<u8> {
    let step16 = matches!(bd, png::BitDepth::Sixteen);
    match (ct, step16) {
        (png::ColorType::Rgba, false) => buf.to_vec(),
        (png::ColorType::Rgb, false) => {
            let mut out = Vec::with_capacity((buf.len() / 3) * 4);
            for c in buf.as_chunks::<3>().0 {
                out.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
            out
        }
        (png::ColorType::GrayscaleAlpha, false) => {
            let mut out = Vec::with_capacity((buf.len() / 2) * 4);
            for c in buf.as_chunks::<2>().0 {
                out.extend_from_slice(&[c[0], c[0], c[0], c[1]]);
            }
            out
        }
        (png::ColorType::Grayscale, false) => {
            let mut out = Vec::with_capacity(buf.len() * 4);
            for &g in buf {
                out.extend_from_slice(&[g, g, g, 255]);
            }
            out
        }
        (png::ColorType::Rgba, true) => {
            let mut out = Vec::with_capacity(buf.len() / 2);
            for c in buf.as_chunks::<2>().0 {
                out.push(c[0]);
            }
            out
        }
        _ => buf.to_vec(),
    }
}
