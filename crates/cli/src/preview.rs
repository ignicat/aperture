use crate::args::PreviewProto;
use engine::arena::PlanCache;
use engine::codec::{ImageFormat, decode_image, encode_image, EncodeOptions};
use engine::color::{WorkingSpace, global_luts};
use engine::pixel::{PixelFormat, PlanarF32};
use engine::pool::ThreadPool;
use engine::resample::kernels::ResampleFilter;
use engine::resample::{ResampleConfig, resample_planar};
use std::env;
use std::io::{self, Write};

pub fn render_terminal_preview(
    bytes: &[u8],
    ext_hint: Option<&str>,
    proto: PreviewProto,
    cache: &PlanCache,
    pool: &ThreadPool,
) -> Result<(), String> {
    let chosen = resolve_preview_proto(proto);
    let dec = decode_image(bytes, ext_hint, &Default::default()).map_err(|e| e.to_string())?;
    let src = PlanarF32::from_packed(&dec.image, WorkingSpace::LinearSrgb, global_luts())
        .map_err(|e| e.to_string())?;

    let (cols, rows) = terminal_cells();
    let max_cols = cols.clamp(24, 96);
    let max_rows = rows.clamp(12, 32);

    let mut err = io::stderr().lock();
    match chosen {
        PreviewProto::Kitty => {
            let max_px_w = (max_cols * 10) as u32;
            let max_px_h = (max_rows * 20) as u32;
            let scaled = fit_for_preview(&src, max_px_w, max_px_h, cache, pool)?;
            let packed = scaled
                .to_packed(PixelFormat::Rgba8, global_luts())
                .map_err(|e| e.to_string())?;
            let png = encode_image(
                &packed,
                ImageFormat::Png,
                &EncodeOptions {
                    optimize: false,
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| e.to_string())?;
            let b64 = encode_base64(&png);
            write_kitty_png(&mut err, &b64).map_err(|e| e.to_string())?;
        }
        PreviewProto::Sixel => {
            let max_px_w = (max_cols * 8) as u32;
            let max_px_h = (max_rows * 16) as u32;
            let scaled = fit_for_preview(&src, max_px_w, max_px_h, cache, pool)?;
            let packed = scaled
                .to_packed(PixelFormat::Rgb8, global_luts())
                .map_err(|e| e.to_string())?;
            write_sixel(&mut err, packed.width as usize, packed.height as usize, &packed.data)
                .map_err(|e| e.to_string())?;
        }
        PreviewProto::Sextant => {
            let max_px_w = (max_cols * 2) as u32;
            let max_px_h = (max_rows * 3) as u32;
            let scaled = fit_for_preview(&src, max_px_w, max_px_h, cache, pool)?;
            let packed = scaled
                .to_packed(PixelFormat::Rgb8, global_luts())
                .map_err(|e| e.to_string())?;
            write_sextants(&mut err, packed.width as usize, packed.height as usize, &packed.data)
                .map_err(|e| e.to_string())?;
        }
        PreviewProto::HalfBlock | PreviewProto::Auto => {
            let max_px_w = max_cols as u32;
            let max_px_h = (max_rows * 2) as u32;
            let scaled = fit_for_preview(&src, max_px_w, max_px_h, cache, pool)?;
            let packed = scaled
                .to_packed(PixelFormat::Rgb8, global_luts())
                .map_err(|e| e.to_string())?;
            write_half_blocks(&mut err, packed.width as usize, packed.height as usize, &packed.data)
                .map_err(|e| e.to_string())?;
        }
    }
    let _ = writeln!(err);
    Ok(())
}

pub fn resolve_preview_proto(req: PreviewProto) -> PreviewProto {
    if req != PreviewProto::Auto {
        return req;
    }
    let term = env::var("TERM").unwrap_or_default().to_ascii_lowercase();
    let term_prog = env::var("TERM_PROGRAM")
        .unwrap_or_default()
        .to_ascii_lowercase();
    if env::var("KITTY_WINDOW_ID").is_ok()
        || env::var("GHOSTTY_RESOURCES_DIR").is_ok()
        || term.contains("kitty")
        || term_prog.contains("wezterm")
        || term_prog.contains("ghostty")
    {
        return PreviewProto::Kitty;
    }
    if term.contains("foot") || term.contains("mlterm") || term.contains("xterm") {
        return PreviewProto::Sixel;
    }
    PreviewProto::Sextant
}

fn terminal_cells() -> (usize, usize) {
    if let Ok(ws) = rustix::termios::tcgetwinsize(io::stderr())
        && ws.ws_col > 0
        && ws.ws_row > 0
    {
        return (ws.ws_col as usize, ws.ws_row as usize);
    }
    (80, 24)
}

fn fit_for_preview(
    src: &PlanarF32,
    max_w: u32,
    max_h: u32,
    cache: &PlanCache,
    pool: &ThreadPool,
) -> Result<PlanarF32, String> {
    let scale = (max_w as f64 / src.width as f64)
        .min(max_h as f64 / src.height as f64)
        .min(1.0);
    let dw = ((src.width as f64 * scale).round() as u32).max(2);
    let dh = ((src.height as f64 * scale).round() as u32).max(2);
    let cfg = ResampleConfig {
        filter: ResampleFilter::Mitchell,
        ..Default::default()
    };
    resample_planar(src, dw, dh, &cfg, cache, pool).map_err(|e| e.to_string())
}

fn write_kitty_png<W: Write>(w: &mut W, b64: &[u8]) -> io::Result<()> {
    let chunk_size = 4096usize;
    let total = b64.len();
    let mut off = 0usize;
    let mut first = true;
    while off < total {
        let end = (off + chunk_size).min(total);
        let more = u8::from(end < total);
        let chunk = std::str::from_utf8(&b64[off..end]).unwrap_or("");
        if first {
            write!(w, "\x1b_Ga=T,f=100,m={more};{chunk}\x1b\\")?;
            first = false;
        } else {
            write!(w, "\x1b_Gm={more};{chunk}\x1b\\")?;
        }
        off = end;
    }
    w.flush()
}

fn write_sixel<W: Write>(w: &mut W, width: usize, height: usize, rgb: &[u8]) -> io::Result<()> {
    const BAYER4: [i16; 16] = [
        -8, 0, -6, 2,
        4, -4, 6, -2,
        -5, 3, -7, 1,
        7, -1, 5, -3,
    ];
    write!(w, "\x1bPq\"1;1;{width};{height}")?;
    for r in 0..6u8 {
        for g in 0..6u8 {
            for b in 0..6u8 {
                let idx = (r as usize) * 36 + (g as usize) * 6 + (b as usize);
                write!(
                    w,
                    "#{idx};2;{};{};{}",
                    r as u32 * 100 / 5,
                    g as u32 * 100 / 5,
                    b as u32 * 100 / 5
                )?;
            }
        }
    }

    let mut band_indices = vec![0u8; width * 6];
    let mut y = 0usize;
    while y < height {
        let band_h = (height - y).min(6);
        let mut used = [false; 216];
        for dy in 0..band_h {
            let py = y + dy;
            for x in 0..width {
                let idx_px = (py * width + x) * 3;
                let dither = BAYER4[((py & 3) << 2) | (x & 3)] * 3;
                let quant = |v: u8| -> usize {
                    let shifted = (v as i16 + dither).clamp(0, 255);
                    ((shifted as usize * 5 + 127) / 255).min(5)
                };
                let ci = (quant(rgb[idx_px]) * 36
                    + quant(rgb[idx_px + 1]) * 6
                    + quant(rgb[idx_px + 2])) as u8;
                used[ci as usize] = true;
                band_indices[dy * width + x] = ci;
            }
        }

        let mut first_color = true;
        for (ci, &is_used) in used.iter().enumerate() {
            if !is_used {
                continue;
            }
            if !first_color {
                write!(w, "$")?;
            }
            first_color = false;
            write!(w, "#{ci}")?;
            let mut x = 0usize;
            while x < width {
                let mut bits = 0u8;
                for dy in 0..band_h {
                    if band_indices[dy * width + x] as usize == ci {
                        bits |= 1u8 << dy;
                    }
                }
                let ch = (bits + 63) as char;
                let mut run = 1usize;
                while x + run < width {
                    let mut next_bits = 0u8;
                    for dy in 0..band_h {
                        if band_indices[dy * width + x + run] as usize == ci {
                            next_bits |= 1u8 << dy;
                        }
                    }
                    if (next_bits + 63) as char == ch {
                        run += 1;
                    } else {
                        break;
                    }
                }
                if run >= 4 {
                    write!(w, "!{run}{ch}")?;
                } else {
                    for _ in 0..run {
                        write!(w, "{ch}")?;
                    }
                }
                x += run;
            }
        }
        if y + 6 < height {
            write!(w, "-")?;
        }
        y += 6;
    }
    write!(w, "\x1b\\")?;
    w.flush()
}

fn write_sextants<W: Write>(w: &mut W, width: usize, height: usize, rgb: &[u8]) -> io::Result<()> {
    let sample = |x: usize, y: usize| -> [u8; 3] {
        let sx = x.min(width - 1);
        let sy = y.min(height - 1);
        let i = (sy * width + sx) * 3;
        [rgb[i], rgb[i + 1], rgb[i + 2]]
    };

    let mut cy = 0usize;
    while cy < height {
        let mut cx = 0usize;
        while cx < width {
            let pts = [
                sample(cx, cy),
                sample(cx + 1, cy),
                sample(cx, cy + 1),
                sample(cx + 1, cy + 1),
                sample(cx, cy + 2),
                sample(cx + 1, cy + 2),
            ];
            let lums: [u16; 6] = pts.map(|p| p[0] as u16 * 54 + p[1] as u16 * 183 + p[2] as u16 * 19);
            let min_l = *lums.iter().min().unwrap_or(&0);
            let max_l = *lums.iter().max().unwrap_or(&0);
            let thresh = (min_l + max_l) / 2;

            let mut mask = 0u8;
            let mut fg_sum = [0u32; 3];
            let mut fg_n = 0u32;
            let mut bg_sum = [0u32; 3];
            let mut bg_n = 0u32;

            for i in 0..6 {
                if lums[i] > thresh {
                    mask |= 1 << i;
                    fg_sum[0] += pts[i][0] as u32;
                    fg_sum[1] += pts[i][1] as u32;
                    fg_sum[2] += pts[i][2] as u32;
                    fg_n += 1;
                } else {
                    bg_sum[0] += pts[i][0] as u32;
                    bg_sum[1] += pts[i][1] as u32;
                    bg_sum[2] += pts[i][2] as u32;
                    bg_n += 1;
                }
            }

            let avg_rgb = |sum: [u32; 3], n: u32, fallback: [u8; 3]| -> [u8; 3] {
                match (sum[0].checked_div(n), sum[1].checked_div(n), sum[2].checked_div(n)) {
                    (Some(r), Some(g), Some(b)) => [r as u8, g as u8, b as u8],
                    _ => fallback,
                }
            };
            let fg = avg_rgb(fg_sum, fg_n, pts[0]);
            let bg = avg_rgb(bg_sum, bg_n, pts[0]);

            let ch = sextant_char(mask);
            write!(
                w,
                "\x1b[38;2;{};{};{};48;2;{};{};{}m{ch}",
                fg[0], fg[1], fg[2], bg[0], bg[1], bg[2]
            )?;
            cx += 2;
        }
        writeln!(w, "\x1b[0m")?;
        cy += 3;
    }
    w.flush()
}

fn sextant_char(mask: u8) -> char {
    match mask & 0x3F {
        0 => ' ',
        0x3F => '\u{2588}',
        0x15 => '\u{258C}',
        0x2A => '\u{2590}',
        m => {
            let skip = u32::from(m > 0x15) + u32::from(m > 0x2A);
            char::from_u32(0x1FB00 + (m as u32) - 1 - skip).unwrap_or('\u{2588}')
        }
    }
}

fn write_half_blocks<W: Write>(w: &mut W, width: usize, height: usize, rgb: &[u8]) -> io::Result<()> {
    let mut y = 0usize;
    while y < height {
        let y2 = (y + 1).min(height - 1);
        for x in 0..width {
            let t = (y * width + x) * 3;
            let b = (y2 * width + x) * 3;
            write!(
                w,
                "\x1b[38;2;{};{};{};48;2;{};{};{}m\u{2580}",
                rgb[t], rgb[t + 1], rgb[t + 2], rgb[b], rgb[b + 1], rgb[b + 2]
            )?;
        }
        writeln!(w, "\x1b[0m")?;
        y += 2;
    }
    w.flush()
}

fn encode_base64(input: &[u8]) -> Vec<u8> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let val = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((val >> 18) & 63) as usize]);
        out.push(TABLE[((val >> 12) & 63) as usize]);
        out.push(if chunk.len() > 1 { TABLE[((val >> 6) & 63) as usize] } else { b'=' });
        out.push(if chunk.len() > 2 { TABLE[(val & 63) as usize] } else { b'=' });
    }
    out
}
