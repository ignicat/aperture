use crate::arena::PlanCache;
use crate::color::{
    WorkingSpace, linear_rgb_to_oklab, linear_to_srgb_f32, oklab_to_linear_rgb_gamut_clipped,
    srgb_to_linear_f32,
};
use crate::error::{EngineError, EngineResult};
use crate::geometry::saliency::{SafeZone, choose_smart_crop};
use crate::geometry::seam::seam_carve_resize;
use crate::geometry::trim::{CropRect, crop_planar};
use crate::pixel::PlanarF32;
use crate::pool::ThreadPool;
use crate::resample::kernels::ResampleFilter;
use crate::resample::{ResampleConfig, resample_planar};
use crate::restore::guided::box_mean_plane;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitMode {
    Auto,
    SmartCrop,
    SeamCarve,
    Ambient,
    Extend,
    Mirror,
    Gradient,
    Contain,
    Stretch,
    Tile,
    Split,
}

impl FitMode {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "smart" | "smart-crop" | "smartcrop" | "fill" | "cover" => Some(Self::SmartCrop),
            "seam" | "seam-carve" | "seamcarve" | "liquid" => Some(Self::SeamCarve),
            "ambient" | "pad-blur" | "blur" => Some(Self::Ambient),
            "extend" | "pad-extend" => Some(Self::Extend),
            "mirror" | "pad-mirror" => Some(Self::Mirror),
            "gradient" | "pad-gradient" | "palette" => Some(Self::Gradient),
            "contain" | "pad" | "letterbox" => Some(Self::Contain),
            "stretch" | "exact" => Some(Self::Stretch),
            "tile" | "repeat" => Some(Self::Tile),
            "split" | "span" => Some(Self::Split),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gravity {
    Auto,
    Center,
    North,
    South,
    East,
    West,
    NorthEast,
    NorthWest,
    SouthEast,
    SouthWest,
    Point(f32, f32),
}

impl Gravity {
    pub fn parse(raw: &str) -> Option<Self> {
        let lower = raw.trim().to_ascii_lowercase();
        match lower.as_str() {
            "auto" => Some(Self::Auto),
            "c" | "center" | "centre" => Some(Self::Center),
            "n" | "north" | "top" => Some(Self::North),
            "s" | "south" | "bottom" => Some(Self::South),
            "e" | "east" | "right" => Some(Self::East),
            "w" | "west" | "left" => Some(Self::West),
            "ne" | "northeast" | "top-right" => Some(Self::NorthEast),
            "nw" | "northwest" | "top-left" => Some(Self::NorthWest),
            "se" | "southeast" | "bottom-right" => Some(Self::SouthEast),
            "sw" | "southwest" | "bottom-left" => Some(Self::SouthWest),
            _ => {
                let (xs, ys) = lower.split_once(',')?;
                let xf: f32 = xs.trim().trim_end_matches('%').parse().ok()?;
                let yf: f32 = ys.trim().trim_end_matches('%').parse().ok()?;
                let nx = if xs.contains('%') || xf > 1.0 { xf / 100.0 } else { xf };
                let ny = if ys.contains('%') || yf > 1.0 { yf / 100.0 } else { yf };
                Some(Self::Point(nx.clamp(0.0, 1.0), ny.clamp(0.0, 1.0)))
            }
        }
    }

    pub fn normalized_xy(self) -> Option<(f32, f32)> {
        match self {
            Self::Auto => None,
            Self::Center => Some((0.5, 0.5)),
            Self::North => Some((0.5, 0.0)),
            Self::South => Some((0.5, 1.0)),
            Self::East => Some((1.0, 0.5)),
            Self::West => Some((0.0, 0.5)),
            Self::NorthEast => Some((1.0, 0.0)),
            Self::NorthWest => Some((0.0, 0.0)),
            Self::SouthEast => Some((1.0, 1.0)),
            Self::SouthWest => Some((0.0, 1.0)),
            Self::Point(x, y) => Some((x, y)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonitorRegion {
    pub name: String,
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct FitConfig {
    pub mode: FitMode,
    pub gravity: Gravity,
    pub safe_zone: SafeZone,
    pub bg_color: [f32; 4],
    pub bg_blur_radius: usize,
    pub bg_dim: f32,
    pub shadow: f32,
    pub corner_radius: f32,
    pub foreground_dim: f32,
    pub vignette: f32,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            mode: FitMode::Auto,
            gravity: Gravity::Auto,
            safe_zone: SafeZone::default(),
            bg_color: [0.0, 0.0, 0.0, 1.0],
            bg_blur_radius: 28,
            bg_dim: 0.45,
            shadow: 0.35,
            corner_radius: 0.0,
            foreground_dim: 0.0,
            vignette: 0.0,
        }
    }
}

pub fn parse_hex_color(raw: &str) -> Option<[f32; 4]> {
    let trimmed = raw.trim().to_ascii_lowercase();
    match trimmed.as_str() {
        "oled" | "black" => return Some([0.0, 0.0, 0.0, 1.0]),
        "white" => return Some([1.0, 1.0, 1.0, 1.0]),
        "transparent" | "none" => return Some([0.0, 0.0, 0.0, 0.0]),
        _ => {}
    }
    let hex = trimmed.strip_prefix('#').unwrap_or(&trimmed);
    let parse_byte = |s: &str| u8::from_str_radix(s, 16).ok();
    match hex.len() {
        6 => {
            let r = parse_byte(&hex[0..2])? as f32 / 255.0;
            let g = parse_byte(&hex[2..4])? as f32 / 255.0;
            let b = parse_byte(&hex[4..6])? as f32 / 255.0;
            Some([
                srgb_to_linear_f32(r),
                srgb_to_linear_f32(g),
                srgb_to_linear_f32(b),
                1.0,
            ])
        }
        8 => {
            let r = parse_byte(&hex[0..2])? as f32 / 255.0;
            let g = parse_byte(&hex[2..4])? as f32 / 255.0;
            let b = parse_byte(&hex[4..6])? as f32 / 255.0;
            let a = parse_byte(&hex[6..8])? as f32 / 255.0;
            Some([
                srgb_to_linear_f32(r),
                srgb_to_linear_f32(g),
                srgb_to_linear_f32(b),
                a,
            ])
        }
        _ => None,
    }
}

pub fn extract_oklab_palette(src: &PlanarF32, count: usize) -> Vec<[u8; 3]> {
    let k = count.clamp(1, 8);
    let mut work = src.clone();
    work.convert_working_space(WorkingSpace::LinearSrgb);

    let side = 48usize;
    let sw = work.width as usize;
    let sh = work.height as usize;
    let mut pts = Vec::with_capacity(side * side);
    for gy in 0..side {
        let sy = (gy * sh / side).min(sh - 1);
        for gx in 0..side {
            let sx = (gx * sw / side).min(sw - 1);
            let idx = sy * sw + sx;
            let (l, a, b) = linear_rgb_to_oklab(work.r[idx], work.g[idx], work.b[idx]);
            pts.push([l, a, b]);
        }
    }
    if pts.is_empty() {
        return vec![[0, 0, 0]];
    }

    let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(k);
    centroids.push(pts[pts.len() / 2]);
    while centroids.len() < k {
        let mut best_pt = pts[0];
        let mut best_dist = -1.0f32;
        for &p in &pts {
            let mut min_c = f32::MAX;
            for &c in &centroids {
                let d = (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2);
                min_c = min_c.min(d);
            }
            if min_c > best_dist {
                best_dist = min_c;
                best_pt = p;
            }
        }
        centroids.push(best_pt);
    }

    let mut counts = vec![0usize; k];
    for _ in 0..8 {
        let mut sums = vec![[0.0f64; 3]; k];
        counts.fill(0);
        for &p in &pts {
            let mut best_c = 0usize;
            let mut best_d = f32::MAX;
            for (ci, &c) in centroids.iter().enumerate() {
                let d = (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2);
                if d < best_d {
                    best_d = d;
                    best_c = ci;
                }
            }
            sums[best_c][0] += p[0] as f64;
            sums[best_c][1] += p[1] as f64;
            sums[best_c][2] += p[2] as f64;
            counts[best_c] += 1;
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                let n = counts[ci] as f64;
                centroids[ci] = [
                    (sums[ci][0] / n) as f32,
                    (sums[ci][1] / n) as f32,
                    (sums[ci][2] / n) as f32,
                ];
            }
        }
    }

    let mut paired: Vec<(usize, [f32; 3])> = counts.into_iter().zip(centroids).collect();
    paired.sort_by_key(|a| std::cmp::Reverse(a.0));

    paired
        .into_iter()
        .map(|(_, c)| {
            let (lr, lg, lb) = oklab_to_linear_rgb_gamut_clipped(c[0], c[1], c[2]);
            [
                (linear_to_srgb_f32(lr).clamp(0.0, 1.0) * 255.0).round() as u8,
                (linear_to_srgb_f32(lg).clamp(0.0, 1.0) * 255.0).round() as u8,
                (linear_to_srgb_f32(lb).clamp(0.0, 1.0) * 255.0).round() as u8,
            ]
        })
        .collect()
}

pub fn fit_planar(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    fit_cfg: &FitConfig,
    res_cfg: &ResampleConfig,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    if dst_w == 0 || dst_h == 0 {
        return Err(EngineError::InvalidDimensions {
            width: dst_w,
            height: dst_h,
        });
    }

    let src_ar = src.width as f64 / src.height as f64;
    let dst_ar = dst_w as f64 / dst_h as f64;
    let ar_ratio = (src_ar / dst_ar).max(dst_ar / src_ar);

    let effective_mode = match fit_cfg.mode {
        FitMode::Auto => {
            if ar_ratio <= 1.015 {
                FitMode::Stretch
            } else {
                let decision = choose_smart_crop(src, dst_w, dst_h, fit_cfg.safe_zone);
                if decision.border_loss_ratio <= 0.14 && ar_ratio <= 1.55 {
                    FitMode::SmartCrop
                } else {
                    FitMode::Ambient
                }
            }
        }
        other => other,
    };

    let mut out = match effective_mode {
        FitMode::Stretch | FitMode::Auto | FitMode::Split => {
            resample_planar(src, dst_w, dst_h, res_cfg, plan_cache, pool)?
        }
        FitMode::SmartCrop => {
            let crop_rect = if let Some((gx, gy)) = fit_cfg.gravity.normalized_xy() {
                let (cw, ch) = if src_ar > dst_ar {
                    (((src.height as f64 * dst_ar).round() as u32).clamp(1, src.width), src.height)
                } else {
                    (src.width, ((src.width as f64 / dst_ar).round() as u32).clamp(1, src.height))
                };
                CropRect {
                    x: ((src.width - cw) as f32 * gx).round() as u32,
                    y: ((src.height - ch) as f32 * gy).round() as u32,
                    width: cw,
                    height: ch,
                }
            } else {
                choose_smart_crop(src, dst_w, dst_h, fit_cfg.safe_zone).rect
            };
            let cropped = crop_planar(src, crop_rect)?;
            resample_planar(&cropped, dst_w, dst_h, res_cfg, plan_cache, pool)?
        }
        FitMode::SeamCarve => {
            let scale_w = dst_w as f64 / src.width as f64;
            let scale_h = dst_h as f64 / src.height as f64;
            let iso_scale = (scale_w * scale_h).sqrt();
            let pre = if (iso_scale - 1.0).abs() > 0.2 {
                let mid_w = ((src.width as f64 * iso_scale).round() as u32).max(16);
                let mid_h = ((src.height as f64 * iso_scale).round() as u32).max(16);
                resample_planar(src, mid_w, mid_h, res_cfg, plan_cache, pool)?
            } else {
                src.clone()
            };
            seam_carve_resize(&pre, dst_w, dst_h)?
        }
        FitMode::Ambient => {
            let fg = contain_scale(src, dst_w, dst_h, res_cfg, plan_cache, pool)?;
            let mut bg = cover_scale_for_bg(src, dst_w, dst_h, plan_cache, pool)?;
            let radius = fit_cfg.bg_blur_radius.max(12).min((dst_w.min(dst_h) / 4) as usize);
            bg.r = box_mean_plane(&bg.r, dst_w as usize, dst_h as usize, radius);
            bg.g = box_mean_plane(&bg.g, dst_w as usize, dst_h as usize, radius);
            bg.b = box_mean_plane(&bg.b, dst_w as usize, dst_h as usize, radius);
            let dim_factor = (1.0 - fit_cfg.bg_dim.clamp(0.0, 0.95)).max(0.02);
            for ((r, g), b) in bg.r.iter_mut().zip(&mut bg.g).zip(&mut bg.b) {
                *r *= dim_factor;
                *g *= dim_factor;
                *b *= dim_factor;
            }
            composite_foreground(&mut bg, &fg, fit_cfg)
        }
        FitMode::Extend | FitMode::Mirror => {
            let fg = contain_scale(src, dst_w, dst_h, res_cfg, plan_cache, pool)?;
            let mirror = effective_mode == FitMode::Mirror;
            let mut bg = fill_extended_background(&fg, dst_w, dst_h, mirror, fit_cfg);
            composite_foreground(&mut bg, &fg, fit_cfg)
        }
        FitMode::Gradient => {
            let fg = contain_scale(src, dst_w, dst_h, res_cfg, plan_cache, pool)?;
            let pal = extract_oklab_palette(src, 3);
            let mut bg = build_gradient_canvas(dst_w, dst_h, &pal, fg.has_alpha);
            composite_foreground(&mut bg, &fg, fit_cfg)
        }
        FitMode::Contain => {
            let fg = contain_scale(src, dst_w, dst_h, res_cfg, plan_cache, pool)?;
            let has_a = fg.has_alpha || fit_cfg.bg_color[3] < 0.999;
            let mut bg = PlanarF32::new(dst_w, dst_h, has_a, WorkingSpace::LinearSrgb)?;
            for i in 0..(dst_w as usize * dst_h as usize) {
                bg.r[i] = fit_cfg.bg_color[0];
                bg.g[i] = fit_cfg.bg_color[1];
                bg.b[i] = fit_cfg.bg_color[2];
                if has_a {
                    bg.a[i] = fit_cfg.bg_color[3];
                }
            }
            composite_foreground(&mut bg, &fg, fit_cfg)
        }
        FitMode::Tile => build_tiled_canvas(src, dst_w, dst_h)?,
    };

    apply_wallpaper_polish(&mut out, fit_cfg.foreground_dim, fit_cfg.vignette);
    Ok(out)
}

pub fn split_multi_monitor(
    src: &PlanarF32,
    monitors: &[MonitorRegion],
    bezel_px: u32,
    fit_cfg: &FitConfig,
    res_cfg: &ResampleConfig,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<Vec<(String, PlanarF32)>> {
    if monitors.is_empty() {
        return Err(EngineError::InvalidSpec("no monitors provided for split".to_string()));
    }
    let min_x = monitors.iter().map(|m| m.x).min().unwrap_or(0);
    let min_y = monitors.iter().map(|m| m.y).min().unwrap_or(0);
    let max_x = monitors
        .iter()
        .map(|m| m.x.saturating_add(m.width as i32))
        .max()
        .unwrap_or(1);
    let max_y = monitors
        .iter()
        .map(|m| m.y.saturating_add(m.height as i32))
        .max()
        .unwrap_or(1);
    let gap = if monitors.len() > 1 { bezel_px } else { 0 };
    let total_w = ((max_x - min_x).max(1) as u32).saturating_add(gap);
    let total_h = ((max_y - min_y).max(1) as u32).saturating_add(gap);

    let mut master_cfg = fit_cfg.clone();
    if master_cfg.mode == FitMode::Split {
        master_cfg.mode = FitMode::SmartCrop;
    }
    let canvas = fit_planar(src, total_w, total_h, &master_cfg, res_cfg, plan_cache, pool)?;

    let mut results = Vec::with_capacity(monitors.len());
    for m in monitors {
        let rx = ((m.x - min_x).max(0) as u32).min(canvas.width.saturating_sub(m.width));
        let ry = ((m.y - min_y).max(0) as u32).min(canvas.height.saturating_sub(m.height));
        let rw = m.width.min(canvas.width - rx);
        let rh = m.height.min(canvas.height - ry);
        let cropped = crop_planar(
            &canvas,
            CropRect {
                x: rx,
                y: ry,
                width: rw,
                height: rh,
            },
        )?;
        results.push((m.name.clone(), cropped));
    }
    Ok(results)
}

fn contain_scale(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    res_cfg: &ResampleConfig,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    let scale = (dst_w as f64 / src.width as f64).min(dst_h as f64 / src.height as f64);
    let fw = ((src.width as f64 * scale).round() as u32).clamp(1, dst_w);
    let fh = ((src.height as f64 * scale).round() as u32).clamp(1, dst_h);
    resample_planar(src, fw, fh, res_cfg, plan_cache, pool)
}

fn cover_scale_for_bg(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    let scale = (dst_w as f64 / src.width as f64).max(dst_h as f64 / src.height as f64);
    let cw = ((src.width as f64 * scale).round() as u32).max(dst_w);
    let ch = ((src.height as f64 * scale).round() as u32).max(dst_h);
    let bg_cfg = ResampleConfig {
        filter: ResampleFilter::Mitchell,
        antiring: 0.0,
        ..ResampleConfig::default()
    };
    let scaled = resample_planar(src, cw, ch, &bg_cfg, plan_cache, pool)?;
    crop_planar(
        &scaled,
        CropRect {
            x: (cw - dst_w) / 2,
            y: (ch - dst_h) / 2,
            width: dst_w,
            height: dst_h,
        },
    )
}

fn placement_offsets(fg: &PlanarF32, dst_w: u32, dst_h: u32, fit_cfg: &FitConfig) -> (usize, usize) {
    let (gx, gy) = fit_cfg.gravity.normalized_xy().unwrap_or((0.5, 0.5));
    let avail_w = dst_w.saturating_sub(fg.width) as f32;
    let avail_h = dst_h.saturating_sub(fg.height) as f32;
    let shift_x = (fit_cfg.safe_zone.left - fit_cfg.safe_zone.right) * 0.5;
    let shift_y = (fit_cfg.safe_zone.top - fit_cfg.safe_zone.bottom) * 0.5;
    let ox = (avail_w * (gx + shift_x).clamp(0.0, 1.0)).round() as usize;
    let oy = (avail_h * (gy + shift_y).clamp(0.0, 1.0)).round() as usize;
    (ox, oy)
}

fn composite_foreground(bg: &mut PlanarF32, fg: &PlanarF32, fit_cfg: &FitConfig) -> PlanarF32 {
    let bw = bg.width as usize;
    let bh = bg.height as usize;
    let fw = fg.width as usize;
    let fh = fg.height as usize;
    let (ox, oy) = placement_offsets(fg, bg.width, bg.height, fit_cfg);

    if fit_cfg.shadow > 0.01 && (fw < bw || fh < bh) {
        let pad = 16usize;
        for y in oy.saturating_sub(pad)..(oy + fh + pad).min(bh) {
            for x in ox.saturating_sub(pad)..(ox + fw + pad).min(bw) {
                let dx = if x < ox {
                    (ox - x) as f32
                } else if x >= ox + fw {
                    (x - (ox + fw - 1)) as f32
                } else {
                    0.0
                };
                let dy = if y < oy {
                    (oy - y) as f32
                } else if y >= oy + fh {
                    (y - (oy + fh - 1)) as f32
                } else {
                    0.0
                };
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > 0.0 && dist < pad as f32 {
                    let s_alpha = (1.0 - dist / pad as f32).powi(2) * fit_cfg.shadow.clamp(0.0, 0.85);
                    let bi = y * bw + x;
                    let keep = 1.0 - s_alpha;
                    bg.r[bi] *= keep;
                    bg.g[bi] *= keep;
                    bg.b[bi] *= keep;
                }
            }
        }
    }

    let r_rad = fit_cfg.corner_radius.max(0.0);
    for fy in 0..fh {
        let dy = oy + fy;
        if dy >= bh {
            break;
        }
        for fx in 0..fw {
            let dx = ox + fx;
            if dx >= bw {
                break;
            }
            let fi = fy * fw + fx;
            let bi = dy * bw + dx;

            let mut mask = if fg.has_alpha { fg.a[fi].clamp(0.0, 1.0) } else { 1.0 };
            if r_rad > 0.5 {
                let cx = if (fx as f32) < r_rad {
                    r_rad - fx as f32
                } else if (fx as f32) >= fw as f32 - r_rad {
                    fx as f32 - (fw as f32 - r_rad)
                } else {
                    0.0
                };
                let cy = if (fy as f32) < r_rad {
                    r_rad - fy as f32
                } else if (fy as f32) >= fh as f32 - r_rad {
                    fy as f32 - (fh as f32 - r_rad)
                } else {
                    0.0
                };
                if cx > 0.0 && cy > 0.0 {
                    let d = (cx * cx + cy * cy).sqrt() - r_rad;
                    mask *= (0.5 - d).clamp(0.0, 1.0);
                }
            }

            if mask >= 0.9999 {
                bg.r[bi] = fg.r[fi];
                bg.g[bi] = fg.g[fi];
                bg.b[bi] = fg.b[fi];
                if bg.has_alpha {
                    bg.a[bi] = 1.0;
                }
            } else if mask > 0.0 {
                let inv = 1.0 - mask;
                bg.r[bi] = fg.r[fi] * mask + bg.r[bi] * inv;
                bg.g[bi] = fg.g[fi] * mask + bg.g[bi] * inv;
                bg.b[bi] = fg.b[fi] * mask + bg.b[bi] * inv;
                if bg.has_alpha {
                    bg.a[bi] = (mask + bg.a[bi] * inv).clamp(0.0, 1.0);
                }
            }
        }
    }
    bg.clone()
}

fn fill_extended_background(
    fg: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    mirror: bool,
    fit_cfg: &FitConfig,
) -> PlanarF32 {
    let bw = dst_w as usize;
    let bh = dst_h as usize;
    let fw = fg.width as usize;
    let fh = fg.height as usize;
    let (ox, oy) = placement_offsets(fg, dst_w, dst_h, fit_cfg);

    let mut bg = PlanarF32::new(dst_w, dst_h, fg.has_alpha, WorkingSpace::LinearSrgb)
        .unwrap_or_else(|_| fg.clone());
    for y in 0..bh {
        let sy = if mirror {
            let rel = y as isize - oy as isize;
            let period = (fh * 2).max(2) as isize;
            let m = rel.rem_euclid(period) as usize;
            if m < fh { m } else { period as usize - 1 - m }
        } else {
            y.saturating_sub(oy).min(fh - 1)
        };
        for x in 0..bw {
            let sx = if mirror {
                let rel = x as isize - ox as isize;
                let period = (fw * 2).max(2) as isize;
                let m = rel.rem_euclid(period) as usize;
                if m < fw { m } else { period as usize - 1 - m }
            } else {
                x.saturating_sub(ox).min(fw - 1)
            };
            let fi = sy * fw + sx;
            let bi = y * bw + x;
            let dist_x = if x < ox { ox - x } else if x >= ox + fw { x - (ox + fw - 1) } else { 0 };
            let dist_y = if y < oy { oy - y } else if y >= oy + fh { y - (oy + fh - 1) } else { 0 };
            let dist = ((dist_x * dist_x + dist_y * dist_y) as f32).sqrt();
            let fade = if fit_cfg.bg_dim > 0.01 {
                1.0 - (dist / (bw.max(bh) as f32 * 0.25).max(1.0)).clamp(0.0, 1.0) * fit_cfg.bg_dim
            } else {
                1.0
            };
            bg.r[bi] = fg.r[fi] * fade;
            bg.g[bi] = fg.g[fi] * fade;
            bg.b[bi] = fg.b[fi] * fade;
            if bg.has_alpha {
                bg.a[bi] = 1.0;
            }
        }
    }
    bg
}

fn build_gradient_canvas(dst_w: u32, dst_h: u32, palette: &[[u8; 3]], has_alpha: bool) -> PlanarF32 {
    let c0 = palette.first().copied().unwrap_or([24, 24, 37]);
    let c1 = palette.get(1).copied().unwrap_or(c0);
    let c2 = palette.get(2).copied().unwrap_or(c1);

    let to_ok = |rgb: [u8; 3]| -> (f32, f32, f32) {
        linear_rgb_to_oklab(
            srgb_to_linear_f32(rgb[0] as f32 / 255.0),
            srgb_to_linear_f32(rgb[1] as f32 / 255.0),
            srgb_to_linear_f32(rgb[2] as f32 / 255.0),
        )
    };
    let o0 = to_ok(c0);
    let o1 = to_ok(c1);
    let o2 = to_ok(c2);

    let w = dst_w as usize;
    let h = dst_h as usize;
    let mut bg = PlanarF32 {
        width: dst_w,
        height: dst_h,
        has_alpha,
        premultiplied: false,
        working_space: WorkingSpace::LinearSrgb,
        r: vec![0.0; w * h],
        g: vec![0.0; w * h],
        b: vec![0.0; w * h],
        a: if has_alpha { vec![1.0; w * h] } else { Vec::new() },
    };

    for y in 0..h {
        let ny = y as f32 / h.max(1) as f32;
        for x in 0..w {
            let nx = x as f32 / w.max(1) as f32;
            let t = (0.6 * nx + 0.4 * ny).clamp(0.0, 1.0);
            let (l, a, b) = if t < 0.5 {
                let u = t * 2.0;
                (
                    o0.0 + (o1.0 - o0.0) * u,
                    o0.1 + (o1.1 - o0.1) * u,
                    o0.2 + (o1.2 - o0.2) * u,
                )
            } else {
                let u = (t - 0.5) * 2.0;
                (
                    o1.0 + (o2.0 - o1.0) * u,
                    o1.1 + (o2.1 - o1.1) * u,
                    o1.2 + (o2.2 - o1.2) * u,
                )
            };
            let (lr, lg, lb) = oklab_to_linear_rgb_gamut_clipped(l * 0.72, a * 0.85, b * 0.85);
            let i = y * w + x;
            bg.r[i] = lr;
            bg.g[i] = lg;
            bg.b[i] = lb;
        }
    }
    bg
}

fn build_tiled_canvas(src: &PlanarF32, dst_w: u32, dst_h: u32) -> EngineResult<PlanarF32> {
    let mut dst = PlanarF32::new(dst_w, dst_h, src.has_alpha, src.working_space)?;
    dst.premultiplied = src.premultiplied;
    let sw = src.width as usize;
    let sh = src.height as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;
    for y in 0..dh {
        let sy = y % sh;
        for x in 0..dw {
            let sx = x % sw;
            let si = sy * sw + sx;
            let di = y * dw + x;
            dst.r[di] = src.r[si];
            dst.g[di] = src.g[si];
            dst.b[di] = src.b[si];
            if src.has_alpha {
                dst.a[di] = src.a[si];
            }
        }
    }
    Ok(dst)
}

fn apply_wallpaper_polish(img: &mut PlanarF32, dim: f32, vignette: f32) {
    let d = (1.0 - dim.clamp(0.0, 0.95)).max(0.02);
    let vig = vignette.clamp(0.0, 1.0);
    if (d - 1.0).abs() < 1e-4 && vig < 1e-4 {
        return;
    }
    let w = img.width as usize;
    let h = img.height as usize;
    let cx = w as f32 * 0.5;
    let cy = h as f32 * 0.5;
    let max_r2 = (cx * cx + cy * cy).max(1.0);

    for y in 0..h {
        let dy = y as f32 - cy;
        for x in 0..w {
            let dx = x as f32 - cx;
            let norm_r2 = (dx * dx + dy * dy) / max_r2;
            let v_mul = 1.0 - vig * 0.65 * norm_r2.powf(1.3);
            let scale = (d * v_mul).clamp(0.0, 1.0);
            let i = y * w + x;
            img.r[i] *= scale;
            img.g[i] *= scale;
            img.b[i] *= scale;
        }
    }
}
