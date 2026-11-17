use crate::error::{EngineError, EngineResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeConstraint {
    Any,
    ShrinkOnly,
    EnlargeOnly,
    CoverMinimum,
    ForceExact,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TargetGeometry {
    Exact {
        width: u32,
        height: u32,
        constraint: SizeConstraint,
    },
    WidthOnly {
        width: u32,
        constraint: SizeConstraint,
    },
    HeightOnly {
        height: u32,
        constraint: SizeConstraint,
    },
    ScaleFactor(f64),
    AspectRatio {
        num: u32,
        den: u32,
        expand: bool,
    },
    PixelBudget(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedDimensions {
    pub width: u32,
    pub height: u32,
    pub skipped: bool,
    pub exact_stretch: bool,
    pub cover_mode: bool,
}

impl TargetGeometry {
    pub fn parse(raw: &str) -> EngineResult<Self> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(EngineError::InvalidSpec("empty target specification".to_string()));
        }

        let (body, constraint) = match trimmed.chars().last() {
            Some('>') => (&trimmed[..trimmed.len() - 1], SizeConstraint::ShrinkOnly),
            Some('<') => (&trimmed[..trimmed.len() - 1], SizeConstraint::EnlargeOnly),
            Some('^') => (&trimmed[..trimmed.len() - 1], SizeConstraint::CoverMinimum),
            Some('!') => (&trimmed[..trimmed.len() - 1], SizeConstraint::ForceExact),
            _ => (trimmed, SizeConstraint::Any),
        };
        let lower = body.trim().to_ascii_lowercase();

        if let Some((w, h)) = preset_dimensions(&lower) {
            return Ok(Self::Exact {
                width: w,
                height: h,
                constraint,
            });
        }

        if let Some(rest) = lower.strip_prefix("ar:") {
            let expand = rest.ends_with('+');
            let clean = rest.trim_end_matches('+');
            let (a, b) = split_pair(clean).ok_or_else(|| {
                EngineError::InvalidSpec(format!("invalid aspect ratio target: {raw}"))
            })?;
            let num: u32 = a
                .parse()
                .map_err(|_| EngineError::InvalidSpec(format!("invalid aspect numerator: {raw}")))?;
            let den: u32 = b
                .parse()
                .map_err(|_| EngineError::InvalidSpec(format!("invalid aspect denominator: {raw}")))?;
            if num == 0 || den == 0 {
                return Err(EngineError::InvalidSpec(format!("zero in aspect ratio: {raw}")));
            }
            return Ok(Self::AspectRatio { num, den, expand });
        }

        if let Some(rest) = lower.strip_prefix("mp:") {
            let mp: f64 = rest
                .parse()
                .map_err(|_| EngineError::InvalidSpec(format!("invalid megapixel count: {raw}")))?;
            if !mp.is_finite() || mp <= 0.0 {
                return Err(EngineError::InvalidSpec(format!("invalid megapixel value: {raw}")));
            }
            return Ok(Self::PixelBudget((mp * 1_000_000.0).round().max(1.0) as u64));
        }

        if let Some(rest) = lower.strip_prefix('@') {
            let px: u64 = rest
                .parse()
                .map_err(|_| EngineError::InvalidSpec(format!("invalid pixel budget: {raw}")))?;
            if px == 0 {
                return Err(EngineError::InvalidSpec("pixel budget must be > 0".to_string()));
            }
            return Ok(Self::PixelBudget(px));
        }

        if let Some(rest) = lower.strip_suffix('%') {
            let pct: f64 = rest
                .parse()
                .map_err(|_| EngineError::InvalidSpec(format!("invalid percentage: {raw}")))?;
            if !pct.is_finite() || pct <= 0.0 {
                return Err(EngineError::InvalidSpec(format!("invalid percentage: {raw}")));
            }
            return Ok(Self::ScaleFactor(pct / 100.0));
        }

        if let Some(rest) = strip_x_suffix(&lower) {
            let factor: f64 = rest
                .parse()
                .map_err(|_| EngineError::InvalidSpec(format!("invalid scale factor: {raw}")))?;
            if !factor.is_finite() || factor <= 0.0 {
                return Err(EngineError::InvalidSpec(format!("invalid scale factor: {raw}")));
            }
            return Ok(Self::ScaleFactor(factor));
        }

        if let Some(rest) = lower.strip_suffix('w') {
            let width: u32 = rest
                .parse()
                .map_err(|_| EngineError::InvalidSpec(format!("invalid width spec: {raw}")))?;
            if width == 0 {
                return Err(EngineError::InvalidSpec("width must be > 0".to_string()));
            }
            return Ok(Self::WidthOnly { width, constraint });
        }

        if let Some(rest) = lower.strip_suffix('h') {
            let height: u32 = rest
                .parse()
                .map_err(|_| EngineError::InvalidSpec(format!("invalid height spec: {raw}")))?;
            if height == 0 {
                return Err(EngineError::InvalidSpec("height must be > 0".to_string()));
            }
            return Ok(Self::HeightOnly { height, constraint });
        }

        if let Some((left, right)) = split_pair_unicode(body.trim()) {
            let left_wild = left == "_" || left == "0" || left.is_empty();
            let right_wild = right == "_" || right == "0" || right.is_empty();
            match (left_wild, right_wild) {
                (false, true) => {
                    let width: u32 = left.parse().map_err(|_| {
                        EngineError::InvalidSpec(format!("invalid width in spec: {raw}"))
                    })?;
                    if width == 0 {
                        return Err(EngineError::InvalidSpec("width must be > 0".to_string()));
                    }
                    return Ok(Self::WidthOnly { width, constraint });
                }
                (true, false) => {
                    let height: u32 = right.parse().map_err(|_| {
                        EngineError::InvalidSpec(format!("invalid height in spec: {raw}"))
                    })?;
                    if height == 0 {
                        return Err(EngineError::InvalidSpec("height must be > 0".to_string()));
                    }
                    return Ok(Self::HeightOnly { height, constraint });
                }
                (false, false) => {
                    let width: u32 = left.parse().map_err(|_| {
                        EngineError::InvalidSpec(format!("invalid width in spec: {raw}"))
                    })?;
                    let height: u32 = right.parse().map_err(|_| {
                        EngineError::InvalidSpec(format!("invalid height in spec: {raw}"))
                    })?;
                    if width == 0 || height == 0 {
                        return Err(EngineError::InvalidSpec("dimensions must be > 0".to_string()));
                    }
                    return Ok(Self::Exact {
                        width,
                        height,
                        constraint,
                    });
                }
                (true, true) => {
                    return Err(EngineError::InvalidSpec(format!("both sides wildcard: {raw}")));
                }
            }
        }

        Err(EngineError::InvalidSpec(format!("unrecognized target spec: {raw}")))
    }

    pub fn parse_list(raw: &str) -> EngineResult<Vec<Self>> {
        let mut list = Vec::new();
        for part in raw.split(',') {
            let item = part.trim();
            if !item.is_empty() {
                list.push(Self::parse(item)?);
            }
        }
        if list.is_empty() {
            return Err(EngineError::InvalidSpec("empty target list".to_string()));
        }
        Ok(list)
    }

    pub fn resolve(&self, src_w: u32, src_h: u32) -> EngineResult<ResolvedDimensions> {
        if src_w == 0 || src_h == 0 {
            return Err(EngineError::InvalidDimensions {
                width: src_w,
                height: src_h,
            });
        }
        match *self {
            Self::Exact {
                width,
                height,
                constraint,
            } => {
                let skipped = match constraint {
                    SizeConstraint::ShrinkOnly => src_w <= width && src_h <= height,
                    SizeConstraint::EnlargeOnly => src_w >= width && src_h >= height,
                    _ => false,
                };
                Ok(ResolvedDimensions {
                    width: if skipped { src_w } else { width },
                    height: if skipped { src_h } else { height },
                    skipped,
                    exact_stretch: constraint == SizeConstraint::ForceExact,
                    cover_mode: constraint == SizeConstraint::CoverMinimum,
                })
            }
            Self::WidthOnly { width, constraint } => {
                let skipped = match constraint {
                    SizeConstraint::ShrinkOnly => src_w <= width,
                    SizeConstraint::EnlargeOnly => src_w >= width,
                    _ => false,
                };
                if skipped {
                    return Ok(ResolvedDimensions {
                        width: src_w,
                        height: src_h,
                        skipped: true,
                        exact_stretch: false,
                        cover_mode: false,
                    });
                }
                let scaled_h = ((width as f64) * (src_h as f64) / (src_w as f64))
                    .round()
                    .clamp(1.0, u32::MAX as f64) as u32;
                Ok(ResolvedDimensions {
                    width,
                    height: scaled_h,
                    skipped: false,
                    exact_stretch: false,
                    cover_mode: false,
                })
            }
            Self::HeightOnly { height, constraint } => {
                let skipped = match constraint {
                    SizeConstraint::ShrinkOnly => src_h <= height,
                    SizeConstraint::EnlargeOnly => src_h >= height,
                    _ => false,
                };
                if skipped {
                    return Ok(ResolvedDimensions {
                        width: src_w,
                        height: src_h,
                        skipped: true,
                        exact_stretch: false,
                        cover_mode: false,
                    });
                }
                let scaled_w = ((height as f64) * (src_w as f64) / (src_h as f64))
                    .round()
                    .clamp(1.0, u32::MAX as f64) as u32;
                Ok(ResolvedDimensions {
                    width: scaled_w,
                    height,
                    skipped: false,
                    exact_stretch: false,
                    cover_mode: false,
                })
            }
            Self::ScaleFactor(factor) => {
                let w = ((src_w as f64) * factor)
                    .round()
                    .clamp(1.0, u32::MAX as f64) as u32;
                let h = ((src_h as f64) * factor)
                    .round()
                    .clamp(1.0, u32::MAX as f64) as u32;
                Ok(ResolvedDimensions {
                    width: w,
                    height: h,
                    skipped: w == src_w && h == src_h,
                    exact_stretch: false,
                    cover_mode: false,
                })
            }
            Self::AspectRatio { num, den, expand } => {
                let target_ar = num as f64 / den as f64;
                let src_ar = src_w as f64 / src_h as f64;
                let (w, h) = if expand {
                    if src_ar < target_ar {
                        (((src_h as f64) * target_ar).round() as u32, src_h)
                    } else {
                        (src_w, ((src_w as f64) / target_ar).round() as u32)
                    }
                } else if src_ar > target_ar {
                    (((src_h as f64) * target_ar).round() as u32, src_h)
                } else {
                    (src_w, ((src_w as f64) / target_ar).round() as u32)
                };
                Ok(ResolvedDimensions {
                    width: w.max(1),
                    height: h.max(1),
                    skipped: w == src_w && h == src_h,
                    exact_stretch: false,
                    cover_mode: false,
                })
            }
            Self::PixelBudget(budget) => {
                let current = (src_w as f64) * (src_h as f64);
                let factor = (budget as f64 / current).sqrt();
                let w = ((src_w as f64) * factor)
                    .round()
                    .clamp(1.0, u32::MAX as f64) as u32;
                let h = ((src_h as f64) * factor)
                    .round()
                    .clamp(1.0, u32::MAX as f64) as u32;
                Ok(ResolvedDimensions {
                    width: w,
                    height: h,
                    skipped: false,
                    exact_stretch: false,
                    cover_mode: false,
                })
            }
        }
    }
}

fn strip_x_suffix(s: &str) -> Option<&str> {
    s.strip_suffix('x')
        .or_else(|| s.strip_suffix('х'))
        .or_else(|| s.strip_suffix('×'))
}

fn split_pair(s: &str) -> Option<(&str, &str)> {
    s.split_once(':')
        .or_else(|| s.split_once('/'))
        .or_else(|| s.split_once('x'))
}

fn split_pair_unicode(s: &str) -> Option<(&str, &str)> {
    for sep in ['x', 'X', 'х', 'Х', '×', '*', ':'] {
        if let Some((a, b)) = s.split_once(sep) {
            return Some((a.trim(), b.trim()));
        }
    }
    None
}

fn preset_dimensions(name: &str) -> Option<(u32, u32)> {
    match name {
        "vga" => Some((640, 480)),
        "hd" | "720p" => Some((1280, 720)),
        "fhd" | "1080p" => Some((1920, 1080)),
        "wuxga" => Some((1920, 1200)),
        "qhd" | "wqhd" | "2k" | "1440p" => Some((2560, 1440)),
        "wqxga" => Some((2560, 1600)),
        "uwfhd" => Some((2560, 1080)),
        "uwqhd" => Some((3440, 1440)),
        "4k" | "uhd" | "2160p" => Some((3840, 2160)),
        "dci4k" => Some((4096, 2160)),
        "5k" => Some((5120, 2880)),
        "dqhd" | "suwqhd" => Some((5120, 1440)),
        "6k" => Some((6016, 3384)),
        "8k" | "4320p" => Some((7680, 4320)),
        "fhd-portrait" => Some((1080, 1920)),
        "qhd-portrait" => Some((1440, 2560)),
        "og" | "og-image" => Some((1200, 630)),
        "a4-300" => Some((2480, 3508)),
        "a4-landscape" => Some((3508, 2480)),
        _ => None,
    }
}
