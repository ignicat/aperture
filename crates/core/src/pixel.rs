use crate::color::{
    ColorLuts, ColorSpace, WorkingSpace, linear_rgb_to_oklab, linear_to_srgb_f32,
    oklab_to_linear_rgb_gamut_clipped, sigmoidal_forward, sigmoidal_inverse, srgb_to_linear_f32,
};
use crate::error::{EngineError, EngineResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    Gray8,
    GrayAlpha8,
    Rgb8,
    Rgba8,
    Gray16,
    GrayAlpha16,
    Rgb16,
    Rgba16,
    RgbF32,
    RgbaF32,
}

impl PixelFormat {
    pub fn channels(self) -> usize {
        match self {
            Self::Gray8 | Self::Gray16 => 1,
            Self::GrayAlpha8 | Self::GrayAlpha16 => 2,
            Self::Rgb8 | Self::Rgb16 | Self::RgbF32 => 3,
            Self::Rgba8 | Self::Rgba16 | Self::RgbaF32 => 4,
        }
    }

    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::Gray8 | Self::GrayAlpha8 | Self::Rgb8 | Self::Rgba8 => 1,
            Self::Gray16 | Self::GrayAlpha16 | Self::Rgb16 | Self::Rgba16 => 2,
            Self::RgbF32 | Self::RgbaF32 => 4,
        }
    }

    pub fn bytes_per_pixel(self) -> usize {
        self.channels() * self.bytes_per_sample()
    }

    pub fn has_alpha(self) -> bool {
        matches!(
            self,
            Self::GrayAlpha8 | Self::Rgba8 | Self::GrayAlpha16 | Self::Rgba16 | Self::RgbaF32
        )
    }

    pub fn is_gray(self) -> bool {
        matches!(
            self,
            Self::Gray8 | Self::GrayAlpha8 | Self::Gray16 | Self::GrayAlpha16
        )
    }

    pub fn is_16bit(self) -> bool {
        matches!(
            self,
            Self::Gray16 | Self::GrayAlpha16 | Self::Rgb16 | Self::Rgba16
        )
    }

    pub fn is_f32(self) -> bool {
        matches!(self, Self::RgbF32 | Self::RgbaF32)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub color_space: ColorSpace,
    pub icc_profile: Option<Vec<u8>>,
    pub is_premultiplied: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameMeta {
    pub duration_ms: u32,
    pub left: u32,
    pub top: u32,
}

#[derive(Debug, Clone)]
pub struct PackedImage {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub color_space: ColorSpace,
    pub icc_profile: Option<Vec<u8>>,
    pub data: Vec<u8>,
}

impl PackedImage {
    pub fn new(
        width: u32,
        height: u32,
        format: PixelFormat,
        color_space: ColorSpace,
        data: Vec<u8>,
    ) -> EngineResult<Self> {
        if width == 0 || height == 0 {
            return Err(EngineError::InvalidDimensions { width, height });
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|px| px.checked_mul(format.bytes_per_pixel()))
            .ok_or(EngineError::InvalidDimensions { width, height })?;
        if data.len() != expected {
            return Err(EngineError::InvalidDimensions { width, height });
        }
        Ok(Self {
            width,
            height,
            format,
            color_space,
            icc_profile: None,
            data,
        })
    }

    pub fn pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub image: PackedImage,
    pub meta: FrameMeta,
}

#[derive(Debug, Clone)]
pub struct FrameSequence {
    pub canvas_width: u32,
    pub canvas_height: u32,
    pub loop_count: u32,
    pub frames: Vec<Frame>,
}

pub trait FrameSource {
    fn canvas_dimensions(&self) -> (u32, u32);
    fn loop_count(&self) -> u32;
    fn next_frame(&mut self) -> EngineResult<Option<Frame>>;
}

pub trait FrameSink {
    fn begin(&mut self, canvas_width: u32, canvas_height: u32, loop_count: u32) -> EngineResult<()>;
    fn push_frame(&mut self, frame: &Frame) -> EngineResult<()>;
    fn finish(self) -> EngineResult<()>;
}

#[derive(Debug, Clone)]
pub struct PlanarF32 {
    pub width: u32,
    pub height: u32,
    pub has_alpha: bool,
    pub premultiplied: bool,
    pub working_space: WorkingSpace,
    pub r: Vec<f32>,
    pub g: Vec<f32>,
    pub b: Vec<f32>,
    pub a: Vec<f32>,
}

impl PlanarF32 {
    pub fn new(
        width: u32,
        height: u32,
        has_alpha: bool,
        working_space: WorkingSpace,
    ) -> EngineResult<Self> {
        if width == 0 || height == 0 {
            return Err(EngineError::InvalidDimensions { width, height });
        }
        let len = (width as usize)
            .checked_mul(height as usize)
            .ok_or(EngineError::InvalidDimensions { width, height })?;
        Ok(Self {
            width,
            height,
            has_alpha,
            premultiplied: false,
            working_space,
            r: vec![0.0; len],
            g: vec![0.0; len],
            b: vec![0.0; len],
            a: if has_alpha { vec![1.0; len] } else { Vec::new() },
        })
    }

    pub fn from_packed(
        img: &PackedImage,
        target_space: WorkingSpace,
        luts: &ColorLuts,
    ) -> EngineResult<Self> {
        let len = img.pixel_count();
        let has_alpha = img.format.has_alpha();
        let mut out = Self::new(img.width, img.height, has_alpha, WorkingSpace::LinearSrgb)?;
        let is_source_linear = img.color_space == ColorSpace::LinearSrgb;

        match img.format {
            PixelFormat::Gray8 => {
                for (i, &v) in img.data.iter().enumerate() {
                    let lin = if is_source_linear {
                        v as f32 / 255.0
                    } else {
                        luts.u8_to_linear[v as usize]
                    };
                    out.r[i] = lin;
                    out.g[i] = lin;
                    out.b[i] = lin;
                }
            }
            PixelFormat::GrayAlpha8 => {
                for (i, chunk) in img.data.as_chunks::<2>().0.iter().enumerate() {
                    let lin = if is_source_linear {
                        chunk[0] as f32 / 255.0
                    } else {
                        luts.u8_to_linear[chunk[0] as usize]
                    };
                    out.r[i] = lin;
                    out.g[i] = lin;
                    out.b[i] = lin;
                    out.a[i] = chunk[1] as f32 / 255.0;
                }
            }
            PixelFormat::Rgb8 => {
                for (i, chunk) in img.data.as_chunks::<3>().0.iter().enumerate() {
                    if is_source_linear {
                        out.r[i] = chunk[0] as f32 / 255.0;
                        out.g[i] = chunk[1] as f32 / 255.0;
                        out.b[i] = chunk[2] as f32 / 255.0;
                    } else {
                        out.r[i] = luts.u8_to_linear[chunk[0] as usize];
                        out.g[i] = luts.u8_to_linear[chunk[1] as usize];
                        out.b[i] = luts.u8_to_linear[chunk[2] as usize];
                    }
                }
            }
            PixelFormat::Rgba8 => {
                for (i, chunk) in img.data.as_chunks::<4>().0.iter().enumerate() {
                    if is_source_linear {
                        out.r[i] = chunk[0] as f32 / 255.0;
                        out.g[i] = chunk[1] as f32 / 255.0;
                        out.b[i] = chunk[2] as f32 / 255.0;
                    } else {
                        out.r[i] = luts.u8_to_linear[chunk[0] as usize];
                        out.g[i] = luts.u8_to_linear[chunk[1] as usize];
                        out.b[i] = luts.u8_to_linear[chunk[2] as usize];
                    }
                    out.a[i] = chunk[3] as f32 / 255.0;
                }
            }
            PixelFormat::Gray16 => {
                for (i, chunk) in img.data.as_chunks::<2>().0.iter().enumerate() {
                    let norm = u16::from_ne_bytes([chunk[0], chunk[1]]) as f32 / 65535.0;
                    let lin = if is_source_linear { norm } else { srgb_to_linear_f32(norm) };
                    out.r[i] = lin;
                    out.g[i] = lin;
                    out.b[i] = lin;
                }
            }
            PixelFormat::GrayAlpha16 => {
                for (i, chunk) in img.data.as_chunks::<4>().0.iter().enumerate() {
                    let norm = u16::from_ne_bytes([chunk[0], chunk[1]]) as f32 / 65535.0;
                    let alpha = u16::from_ne_bytes([chunk[2], chunk[3]]) as f32 / 65535.0;
                    let lin = if is_source_linear { norm } else { srgb_to_linear_f32(norm) };
                    out.r[i] = lin;
                    out.g[i] = lin;
                    out.b[i] = lin;
                    out.a[i] = alpha;
                }
            }
            PixelFormat::Rgb16 => {
                for (i, chunk) in img.data.as_chunks::<6>().0.iter().enumerate() {
                    let nr = u16::from_ne_bytes([chunk[0], chunk[1]]) as f32 / 65535.0;
                    let ng = u16::from_ne_bytes([chunk[2], chunk[3]]) as f32 / 65535.0;
                    let nb = u16::from_ne_bytes([chunk[4], chunk[5]]) as f32 / 65535.0;
                    if is_source_linear {
                        out.r[i] = nr;
                        out.g[i] = ng;
                        out.b[i] = nb;
                    } else {
                        out.r[i] = srgb_to_linear_f32(nr);
                        out.g[i] = srgb_to_linear_f32(ng);
                        out.b[i] = srgb_to_linear_f32(nb);
                    }
                }
            }
            PixelFormat::Rgba16 => {
                for (i, chunk) in img.data.as_chunks::<8>().0.iter().enumerate() {
                    let nr = u16::from_ne_bytes([chunk[0], chunk[1]]) as f32 / 65535.0;
                    let ng = u16::from_ne_bytes([chunk[2], chunk[3]]) as f32 / 65535.0;
                    let nb = u16::from_ne_bytes([chunk[4], chunk[5]]) as f32 / 65535.0;
                    let na = u16::from_ne_bytes([chunk[6], chunk[7]]) as f32 / 65535.0;
                    if is_source_linear {
                        out.r[i] = nr;
                        out.g[i] = ng;
                        out.b[i] = nb;
                    } else {
                        out.r[i] = srgb_to_linear_f32(nr);
                        out.g[i] = srgb_to_linear_f32(ng);
                        out.b[i] = srgb_to_linear_f32(nb);
                    }
                    out.a[i] = na;
                }
            }
            PixelFormat::RgbF32 => {
                for (i, chunk) in img.data.as_chunks::<12>().0.iter().enumerate() {
                    let fr = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    let fg = f32::from_ne_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                    let fb = f32::from_ne_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
                    if is_source_linear {
                        out.r[i] = fr;
                        out.g[i] = fg;
                        out.b[i] = fb;
                    } else {
                        out.r[i] = srgb_to_linear_f32(fr);
                        out.g[i] = srgb_to_linear_f32(fg);
                        out.b[i] = srgb_to_linear_f32(fb);
                    }
                }
            }
            PixelFormat::RgbaF32 => {
                for (i, chunk) in img.data.as_chunks::<16>().0.iter().enumerate() {
                    let fr = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    let fg = f32::from_ne_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                    let fb = f32::from_ne_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
                    let fa = f32::from_ne_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]);
                    if is_source_linear {
                        out.r[i] = fr;
                        out.g[i] = fg;
                        out.b[i] = fb;
                    } else {
                        out.r[i] = srgb_to_linear_f32(fr);
                        out.g[i] = srgb_to_linear_f32(fg);
                        out.b[i] = srgb_to_linear_f32(fb);
                    }
                    out.a[i] = fa;
                }
            }
        }

        if len > 0 {
            out.convert_working_space(target_space);
        }
        Ok(out)
    }

    pub fn convert_working_space(&mut self, target: WorkingSpace) {
        if self.working_space == target {
            return;
        }
        self.convert_to_linear_srgb();
        match target {
            WorkingSpace::LinearSrgb => {}
            WorkingSpace::PerceptualSrgb => {
                for ((r, g), b) in self.r.iter_mut().zip(&mut self.g).zip(&mut self.b) {
                    *r = linear_to_srgb_f32(*r);
                    *g = linear_to_srgb_f32(*g);
                    *b = linear_to_srgb_f32(*b);
                }
            }
            WorkingSpace::Sigmoidal(k) => {
                for ((r, g), b) in self.r.iter_mut().zip(&mut self.g).zip(&mut self.b) {
                    *r = sigmoidal_forward(*r, k);
                    *g = sigmoidal_forward(*g, k);
                    *b = sigmoidal_forward(*b, k);
                }
            }
            WorkingSpace::Oklab => {
                for ((r, g), b) in self.r.iter_mut().zip(&mut self.g).zip(&mut self.b) {
                    let (l, oa, ob) = linear_rgb_to_oklab(*r, *g, *b);
                    *r = l;
                    *g = oa;
                    *b = ob;
                }
            }
        }
        self.working_space = target;
    }

    fn convert_to_linear_srgb(&mut self) {
        match self.working_space {
            WorkingSpace::LinearSrgb => {}
            WorkingSpace::PerceptualSrgb => {
                for ((r, g), b) in self.r.iter_mut().zip(&mut self.g).zip(&mut self.b) {
                    *r = srgb_to_linear_f32(*r);
                    *g = srgb_to_linear_f32(*g);
                    *b = srgb_to_linear_f32(*b);
                }
            }
            WorkingSpace::Sigmoidal(k) => {
                for ((r, g), b) in self.r.iter_mut().zip(&mut self.g).zip(&mut self.b) {
                    *r = sigmoidal_inverse(*r, k);
                    *g = sigmoidal_inverse(*g, k);
                    *b = sigmoidal_inverse(*b, k);
                }
            }
            WorkingSpace::Oklab => {
                for ((r, g), b) in self.r.iter_mut().zip(&mut self.g).zip(&mut self.b) {
                    let (lr, lg, lb) = oklab_to_linear_rgb_gamut_clipped(*r, *g, *b);
                    *r = lr;
                    *g = lg;
                    *b = lb;
                }
            }
        }
        self.working_space = WorkingSpace::LinearSrgb;
    }

    pub fn premultiply(&mut self) {
        if !self.has_alpha || self.premultiplied {
            return;
        }
        let was_space = self.working_space;
        if was_space != WorkingSpace::LinearSrgb {
            self.convert_to_linear_srgb();
        }
        for (((r, g), b), &a) in self
            .r
            .iter_mut()
            .zip(&mut self.g)
            .zip(&mut self.b)
            .zip(&self.a)
        {
            let ac = a.clamp(0.0, 1.0);
            *r *= ac;
            *g *= ac;
            *b *= ac;
        }
        self.premultiplied = true;
        if was_space != WorkingSpace::LinearSrgb && was_space != WorkingSpace::Oklab {
            self.convert_working_space(was_space);
        }
    }

    pub fn unpremultiply(&mut self) {
        if !self.has_alpha || !self.premultiplied {
            return;
        }
        const EPS: f32 = 1e-6;
        for (((r, g), b), a) in self
            .r
            .iter_mut()
            .zip(&mut self.g)
            .zip(&mut self.b)
            .zip(&mut self.a)
        {
            let ac = a.clamp(0.0, 1.0);
            *a = ac;
            if ac > EPS {
                let inv = 1.0 / ac;
                *r = (*r * inv).clamp(0.0, 1.0);
                *g = (*g * inv).clamp(0.0, 1.0);
                *b = (*b * inv).clamp(0.0, 1.0);
            } else {
                *r = 0.0;
                *g = 0.0;
                *b = 0.0;
            }
        }
        self.premultiplied = false;
    }

    pub fn fix_transparent_bleeding(&mut self, alpha_threshold: f32) {
        if !self.has_alpha || self.premultiplied {
            return;
        }
        let w = self.width as usize;
        let h = self.height as usize;
        let len = w * h;
        if len == 0 {
            return;
        }
        let mut dist = vec![u32::MAX; len];
        let mut seed_idx = vec![0usize; len];
        let mut has_seed = false;
        let mut has_transparent = false;
        for i in 0..len {
            if self.a[i] > alpha_threshold {
                dist[i] = 0;
                seed_idx[i] = i;
                has_seed = true;
            } else {
                has_transparent = true;
            }
        }
        if !has_seed || !has_transparent {
            return;
        }
        let relax = |dst_i: usize, src_i: usize, step: u32, dist: &mut [u32], seed_idx: &mut [usize]| {
            let sd = dist[src_i];
            if sd < u32::MAX {
                let cand = sd.saturating_add(step);
                if cand < dist[dst_i] {
                    dist[dst_i] = cand;
                    seed_idx[dst_i] = seed_idx[src_i];
                }
            }
        };
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if dist[i] == 0 {
                    continue;
                }
                if x > 0 {
                    relax(i, i - 1, 3, &mut dist, &mut seed_idx);
                }
                if y > 0 {
                    relax(i, i - w, 3, &mut dist, &mut seed_idx);
                    if x > 0 {
                        relax(i, i - w - 1, 4, &mut dist, &mut seed_idx);
                    }
                    if x + 1 < w {
                        relax(i, i - w + 1, 4, &mut dist, &mut seed_idx);
                    }
                }
            }
        }
        for y in (0..h).rev() {
            for x in (0..w).rev() {
                let i = y * w + x;
                if dist[i] == 0 {
                    continue;
                }
                if x + 1 < w {
                    relax(i, i + 1, 3, &mut dist, &mut seed_idx);
                }
                if y + 1 < h {
                    relax(i, i + w, 3, &mut dist, &mut seed_idx);
                    if x + 1 < w {
                        relax(i, i + w + 1, 4, &mut dist, &mut seed_idx);
                    }
                    if x > 0 {
                        relax(i, i + w - 1, 4, &mut dist, &mut seed_idx);
                    }
                }
            }
        }
        let orig_r = self.r.clone();
        let orig_g = self.g.clone();
        let orig_b = self.b.clone();
        for i in 0..len {
            if self.a[i] <= alpha_threshold && dist[i] < u32::MAX {
                let src = seed_idx[i];
                self.r[i] = orig_r[src];
                self.g[i] = orig_g[src];
                self.b[i] = orig_b[src];
            }
        }
    }

    pub fn to_packed(&self, format: PixelFormat, luts: &ColorLuts) -> EngineResult<PackedImage> {
        let mut copy = self.clone();
        if copy.premultiplied {
            copy.unpremultiply();
        }
        if copy.working_space != WorkingSpace::LinearSrgb {
            copy.convert_to_linear_srgb();
        }
        let len = (copy.width as usize) * (copy.height as usize);
        let mut data = vec![0u8; len * format.bytes_per_pixel()];
        let out_color_space = match format {
            PixelFormat::Gray8 | PixelFormat::GrayAlpha8 | PixelFormat::Gray16 | PixelFormat::GrayAlpha16 => {
                ColorSpace::Gray
            }
            _ => ColorSpace::Srgb,
        };
        let to_u8 = |lin: f32| -> u8 {
            let idx = (lin.clamp(0.0, 1.0) * 65535.0).round() as usize;
            luts.u16_linear_to_u8[idx]
        };
        let to_u16 = |lin: f32| -> u16 {
            (linear_to_srgb_f32(lin).clamp(0.0, 1.0) * 65535.0).round() as u16
        };
        let alpha_at = |i: usize| -> f32 {
            if copy.has_alpha {
                copy.a[i].clamp(0.0, 1.0)
            } else {
                1.0
            }
        };

        match format {
            PixelFormat::Gray8 => {
                for (i, slot) in data.iter_mut().enumerate().take(len) {
                    let lum = 0.2126 * copy.r[i] + 0.7152 * copy.g[i] + 0.0722 * copy.b[i];
                    *slot = to_u8(lum);
                }
            }
            PixelFormat::GrayAlpha8 => {
                for i in 0..len {
                    let lum = 0.2126 * copy.r[i] + 0.7152 * copy.g[i] + 0.0722 * copy.b[i];
                    data[i * 2] = to_u8(lum);
                    data[i * 2 + 1] = (alpha_at(i) * 255.0).round() as u8;
                }
            }
            PixelFormat::Rgb8 => {
                for i in 0..len {
                    data[i * 3] = to_u8(copy.r[i]);
                    data[i * 3 + 1] = to_u8(copy.g[i]);
                    data[i * 3 + 2] = to_u8(copy.b[i]);
                }
            }
            PixelFormat::Rgba8 => {
                for i in 0..len {
                    data[i * 4] = to_u8(copy.r[i]);
                    data[i * 4 + 1] = to_u8(copy.g[i]);
                    data[i * 4 + 2] = to_u8(copy.b[i]);
                    data[i * 4 + 3] = (alpha_at(i) * 255.0).round() as u8;
                }
            }
            PixelFormat::Gray16 => {
                for i in 0..len {
                    let lum = 0.2126 * copy.r[i] + 0.7152 * copy.g[i] + 0.0722 * copy.b[i];
                    let bytes = to_u16(lum).to_ne_bytes();
                    data[i * 2..i * 2 + 2].copy_from_slice(&bytes);
                }
            }
            PixelFormat::GrayAlpha16 => {
                for i in 0..len {
                    let lum = 0.2126 * copy.r[i] + 0.7152 * copy.g[i] + 0.0722 * copy.b[i];
                    let lv = to_u16(lum).to_ne_bytes();
                    let av = (alpha_at(i) * 65535.0).round() as u16;
                    data[i * 4..i * 4 + 2].copy_from_slice(&lv);
                    data[i * 4 + 2..i * 4 + 4].copy_from_slice(&av.to_ne_bytes());
                }
            }
            PixelFormat::Rgb16 => {
                for i in 0..len {
                    let base = i * 6;
                    data[base..base + 2].copy_from_slice(&to_u16(copy.r[i]).to_ne_bytes());
                    data[base + 2..base + 4].copy_from_slice(&to_u16(copy.g[i]).to_ne_bytes());
                    data[base + 4..base + 6].copy_from_slice(&to_u16(copy.b[i]).to_ne_bytes());
                }
            }
            PixelFormat::Rgba16 => {
                for i in 0..len {
                    let base = i * 8;
                    let av = (alpha_at(i) * 65535.0).round() as u16;
                    data[base..base + 2].copy_from_slice(&to_u16(copy.r[i]).to_ne_bytes());
                    data[base + 2..base + 4].copy_from_slice(&to_u16(copy.g[i]).to_ne_bytes());
                    data[base + 4..base + 6].copy_from_slice(&to_u16(copy.b[i]).to_ne_bytes());
                    data[base + 6..base + 8].copy_from_slice(&av.to_ne_bytes());
                }
            }
            PixelFormat::RgbF32 => {
                for i in 0..len {
                    let base = i * 12;
                    data[base..base + 4].copy_from_slice(&linear_to_srgb_f32(copy.r[i]).to_ne_bytes());
                    data[base + 4..base + 8].copy_from_slice(&linear_to_srgb_f32(copy.g[i]).to_ne_bytes());
                    data[base + 8..base + 12].copy_from_slice(&linear_to_srgb_f32(copy.b[i]).to_ne_bytes());
                }
            }
            PixelFormat::RgbaF32 => {
                for i in 0..len {
                    let base = i * 16;
                    data[base..base + 4].copy_from_slice(&linear_to_srgb_f32(copy.r[i]).to_ne_bytes());
                    data[base + 4..base + 8].copy_from_slice(&linear_to_srgb_f32(copy.g[i]).to_ne_bytes());
                    data[base + 8..base + 12].copy_from_slice(&linear_to_srgb_f32(copy.b[i]).to_ne_bytes());
                    data[base + 12..base + 16].copy_from_slice(&alpha_at(i).to_ne_bytes());
                }
            }
        }
        PackedImage::new(copy.width, copy.height, format, out_color_space, data)
    }
}

#[derive(Debug, Clone)]
pub struct StripWindow {
    width: usize,
    total_height: usize,
    ring_rows: usize,
    has_alpha: bool,
    base_y: usize,
    filled_rows: usize,
    r: Vec<f32>,
    g: Vec<f32>,
    b: Vec<f32>,
    a: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct StripRow<'a> {
    pub r: &'a [f32],
    pub g: &'a [f32],
    pub b: &'a [f32],
    pub a: Option<&'a [f32]>,
}

impl StripWindow {
    pub fn new(width: usize, total_height: usize, ring_rows: usize, has_alpha: bool) -> EngineResult<Self> {
        if width == 0 || total_height == 0 || ring_rows == 0 {
            return Err(EngineError::InvalidDimensions {
                width: width as u32,
                height: total_height as u32,
            });
        }
        let buf_len = width
            .checked_mul(ring_rows)
            .ok_or(EngineError::InvalidDimensions {
                width: width as u32,
                height: total_height as u32,
            })?;
        Ok(Self {
            width,
            total_height,
            ring_rows,
            has_alpha,
            base_y: 0,
            filled_rows: 0,
            r: vec![0.0; buf_len],
            g: vec![0.0; buf_len],
            b: vec![0.0; buf_len],
            a: if has_alpha { vec![1.0; buf_len] } else { Vec::new() },
        })
    }

    pub fn push_row(&mut self, r_row: &[f32], g_row: &[f32], b_row: &[f32], a_row: Option<&[f32]>) -> EngineResult<()> {
        if r_row.len() != self.width || g_row.len() != self.width || b_row.len() != self.width {
            return Err(EngineError::InvalidDimensions {
                width: r_row.len() as u32,
                height: 1,
            });
        }
        let target_y = self.base_y + self.filled_rows;
        if target_y >= self.total_height {
            return Err(EngineError::InvalidDimensions {
                width: self.width as u32,
                height: self.total_height as u32,
            });
        }
        if self.filled_rows == self.ring_rows {
            self.base_y += 1;
            self.filled_rows -= 1;
        }
        let slot = target_y % self.ring_rows;
        let offset = slot * self.width;
        self.r[offset..offset + self.width].copy_from_slice(r_row);
        self.g[offset..offset + self.width].copy_from_slice(g_row);
        self.b[offset..offset + self.width].copy_from_slice(b_row);
        if self.has_alpha
            && let Some(ar) = a_row
            && ar.len() == self.width
        {
            self.a[offset..offset + self.width].copy_from_slice(ar);
        }
        self.filled_rows += 1;
        Ok(())
    }

    pub fn row_slices_clamped(&self, y: i64) -> Option<StripRow<'_>> {
        if self.filled_rows == 0 {
            return None;
        }
        let min_y = self.base_y as i64;
        let max_y = (self.base_y + self.filled_rows - 1) as i64;
        let clamped_y = y.clamp(min_y, max_y) as usize;
        let slot = clamped_y % self.ring_rows;
        let offset = slot * self.width;
        let end = offset + self.width;
        Some(StripRow {
            r: &self.r[offset..end],
            g: &self.g[offset..end],
            b: &self.b[offset..end],
            a: if self.has_alpha {
                Some(&self.a[offset..end])
            } else {
                None
            },
        })
    }
}
