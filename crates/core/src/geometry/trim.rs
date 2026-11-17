use crate::error::{EngineError, EngineResult};
use crate::pixel::PlanarF32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub fn detect_solid_borders(img: &PlanarF32, tolerance: f32) -> CropRect {
    let w = img.width as usize;
    let h = img.height as usize;
    if w < 16 || h < 16 {
        return CropRect {
            x: 0,
            y: 0,
            width: img.width,
            height: img.height,
        };
    }

    let corners = [0, w - 1, (h - 1) * w, (h - 1) * w + w - 1];
    let mut ref_r = 0.0f32;
    let mut ref_g = 0.0f32;
    let mut ref_b = 0.0f32;
    for &c in &corners {
        ref_r += img.r[c];
        ref_g += img.g[c];
        ref_b += img.b[c];
    }
    ref_r *= 0.25;
    ref_g *= 0.25;
    ref_b *= 0.25;

    let tol = tolerance.max(0.005);
    let is_border_px = |idx: usize| -> bool {
        (img.r[idx] - ref_r).abs() <= tol
            && (img.g[idx] - ref_g).abs() <= tol
            && (img.b[idx] - ref_b).abs() <= tol
    };

    let row_is_border = |y: usize| -> bool {
        let mut diff_count = 0usize;
        let max_outliers = (w / 40).max(1);
        for x in 0..w {
            if !is_border_px(y * w + x) {
                diff_count += 1;
                if diff_count > max_outliers {
                    return false;
                }
            }
        }
        true
    };

    let mut top = 0usize;
    while top < h / 2 && row_is_border(top) {
        top += 1;
    }
    let mut bottom = h;
    while bottom > top + h / 4 && row_is_border(bottom - 1) {
        bottom -= 1;
    }

    let col_is_border = |x: usize| -> bool {
        let mut diff_count = 0usize;
        let span = bottom - top;
        let max_outliers = (span / 40).max(1);
        for y in top..bottom {
            if !is_border_px(y * w + x) {
                diff_count += 1;
                if diff_count > max_outliers {
                    return false;
                }
            }
        }
        true
    };

    let mut left = 0usize;
    while left < w / 2 && col_is_border(left) {
        left += 1;
    }
    let mut right = w;
    while right > left + w / 4 && col_is_border(right - 1) {
        right -= 1;
    }

    let cw = (right - left) as u32;
    let ch = (bottom - top) as u32;
    if cw < 16 || ch < 16 || (cw as u64 * ch as u64) * 4 < (img.width as u64 * img.height as u64) {
        return CropRect {
            x: 0,
            y: 0,
            width: img.width,
            height: img.height,
        };
    }

    CropRect {
        x: left as u32,
        y: top as u32,
        width: cw,
        height: ch,
    }
}

pub fn crop_planar(src: &PlanarF32, rect: CropRect) -> EngineResult<PlanarF32> {
    if rect.width == 0
        || rect.height == 0
        || rect.x.saturating_add(rect.width) > src.width
        || rect.y.saturating_add(rect.height) > src.height
    {
        return Err(EngineError::InvalidDimensions {
            width: rect.width,
            height: rect.height,
        });
    }
    if rect.x == 0 && rect.y == 0 && rect.width == src.width && rect.height == src.height {
        return Ok(src.clone());
    }

    let sw = src.width as usize;
    let cw = rect.width as usize;
    let ch = rect.height as usize;
    let ox = rect.x as usize;
    let oy = rect.y as usize;
    let mut dst = PlanarF32::new(rect.width, rect.height, src.has_alpha, src.working_space)?;
    dst.premultiplied = src.premultiplied;

    for row in 0..ch {
        let s_off = (oy + row) * sw + ox;
        let d_off = row * cw;
        dst.r[d_off..d_off + cw].copy_from_slice(&src.r[s_off..s_off + cw]);
        dst.g[d_off..d_off + cw].copy_from_slice(&src.g[s_off..s_off + cw]);
        dst.b[d_off..d_off + cw].copy_from_slice(&src.b[s_off..s_off + cw]);
        if src.has_alpha {
            dst.a[d_off..d_off + cw].copy_from_slice(&src.a[s_off..s_off + cw]);
        }
    }
    Ok(dst)
}
