use crate::config::{load_presets, split_shell_tokens};
use engine::archive::{
    ArchiveErrorPolicy, ArchiveFormat, ArchiveMode, ArchiveProcessOptions, DedupMode,
    NonImagePolicy,
};
use engine::codec::metadata::MetadataPolicy;
use engine::codec::{ChromaSubsampling, ImageFormat};
use engine::color::WorkingSpace;
use engine::geometry::fit::{FitMode, Gravity, parse_hex_color};
use engine::geometry::saliency::SafeZone;
use engine::geometry::spec::TargetGeometry;
use engine::pipeline::PipelineConfig;
use engine::resample::kernels::ResampleFilter;
use engine::restore::dither::DitherMode;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Process,
    Fit,
    Archive,
    Inspect,
    Compare,
    Bench,
    Watch,
    Preset(PresetAction),
    Completions(ShellKind),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresetAction {
    List,
    Path,
    Show(String),
    Save { name: String, spec: String },
    Delete(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellKind {
    Bash,
    Zsh,
    Fish,
    Nushell,
}

impl ShellKind {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "bash" => Some(Self::Bash),
            "zsh" => Some(Self::Zsh),
            "fish" => Some(Self::Fish),
            "nu" | "nushell" => Some(Self::Nushell),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScreenTarget {
    None,
    Auto,
    Output(String),
    All,
    PerScreen,
    Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictPolicy {
    Skip,
    Overwrite,
    Rename,
    Smaller,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailLevel {
    Error,
    Truncated,
    Warn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewProto {
    Auto,
    Kitty,
    Sixel,
    Sextant,
    HalfBlock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaletteFormat {
    None,
    Json,
    Css,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrientationFilter {
    Landscape,
    Portrait,
    Square,
}

#[derive(Debug, Clone)]
pub struct SafeZoneRule {
    pub edge: SafeEdge,
    pub value: f32,
    pub is_percent: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeEdge {
    Top,
    Bottom,
    Left,
    Right,
    All,
}

#[derive(Debug, Clone)]
pub struct CliOptions {
    pub command: Command,
    pub inputs: Vec<String>,
    pub target_specs: Vec<String>,
    pub screen: ScreenTarget,
    pub recursive: bool,
    pub manifest: Option<String>,
    pub out_path: Option<String>,
    pub name_template: Option<String>,
    pub suffix: Option<String>,
    pub replace: bool,
    pub backup: bool,
    pub keep_dates: bool,
    pub verify: bool,
    pub stdout_mode: bool,
    pub on_conflict: ConflictPolicy,
    pub pipeline: PipelineConfig,
    pub archive: ArchiveProcessOptions,
    pub link_skipped: bool,
    pub copy_skipped: bool,
    pub include_globs: Vec<String>,
    pub exclude_globs: Vec<String>,
    pub only_orientation: Option<OrientationFilter>,
    pub min_res: Option<(u32, u32)>,
    pub max_res: Option<(u32, u32)>,
    pub fail_level: FailLevel,
    pub strict: bool,
    pub threads: Option<usize>,
    pub memory_limit_bytes: Option<usize>,
    pub dry_run: bool,
    pub resume_path: Option<String>,
    pub watch_interval_ms: u64,
    pub report: ReportFormat,
    pub json_output: bool,
    pub quiet: bool,
    pub verbose: bool,
    pub preview: bool,
    pub preview_only: bool,
    pub preview_proto: PreviewProto,
    pub palette_out: PaletteFormat,
    pub set_wallpaper: bool,
    pub bezel_px: u32,
    pub safe_zone_rules: Vec<SafeZoneRule>,
    pub bench_iters: usize,
    pub help_requested: bool,
    pub version_requested: bool,
}

impl Default for CliOptions {
    fn default() -> Self {
        Self {
            command: Command::Process,
            inputs: Vec::new(),
            target_specs: Vec::new(),
            screen: ScreenTarget::None,
            recursive: false,
            manifest: None,
            out_path: None,
            name_template: None,
            suffix: None,
            replace: false,
            backup: false,
            keep_dates: false,
            verify: false,
            stdout_mode: false,
            on_conflict: ConflictPolicy::Rename,
            pipeline: PipelineConfig::default(),
            archive: ArchiveProcessOptions::default(),
            link_skipped: false,
            copy_skipped: false,
            include_globs: Vec::new(),
            exclude_globs: Vec::new(),
            only_orientation: None,
            min_res: None,
            max_res: None,
            fail_level: FailLevel::Error,
            strict: false,
            threads: None,
            memory_limit_bytes: None,
            dry_run: false,
            resume_path: None,
            watch_interval_ms: 400,
            report: ReportFormat::Text,
            json_output: false,
            quiet: false,
            verbose: false,
            preview: false,
            preview_only: false,
            preview_proto: PreviewProto::Auto,
            palette_out: PaletteFormat::None,
            set_wallpaper: false,
            bezel_px: 0,
            safe_zone_rules: Vec::new(),
            bench_iters: 3,
            help_requested: false,
            version_requested: false,
        }
    }
}

pub fn parse_args(program: &str, raw_args: &[String]) -> Result<CliOptions, String> {
    let mut preset_names = Vec::new();
    let mut i = 0usize;
    while i < raw_args.len() {
        let arg = &raw_args[i];
        if arg == "--" {
            break;
        }
        if let Some(val) = arg.strip_prefix("--preset=") {
            preset_names.push(val.to_string());
        } else if (arg == "--preset" || arg == "-p") && i + 1 < raw_args.len() {
            preset_names.push(raw_args[i + 1].clone());
            i += 1;
        }
        i += 1;
    }

    let mut merged = Vec::new();
    if !preset_names.is_empty() {
        let presets = load_presets(program);
        for name in preset_names {
            let line = presets
                .get(&name)
                .ok_or_else(|| format!("unknown preset: {name}"))?;
            merged.extend(split_shell_tokens(line)?);
        }
    }
    merged.extend(raw_args.iter().cloned());

    parse_merged_tokens(&merged)
}

fn parse_merged_tokens(tokens: &[String]) -> Result<CliOptions, String> {
    let mut opts = CliOptions::default();
    if tokens.is_empty() {
        opts.help_requested = true;
        return Ok(opts);
    }

    let mut idx = 0usize;
    if let Some(first) = tokens.first()
        && !first.starts_with('-')
        && !Path::new(first).exists()
    {
        match first.to_ascii_lowercase().as_str() {
            "process" => {
                opts.command = Command::Process;
                idx = 1;
            }
            "fit" | "wp" => {
                opts.command = Command::Fit;
                idx = 1;
            }
            "archive" | "arc" => {
                opts.command = Command::Archive;
                idx = 1;
            }
            "inspect" | "info" => {
                opts.command = Command::Inspect;
                idx = 1;
            }
            "compare" | "diff" => {
                opts.command = Command::Compare;
                idx = 1;
            }
            "bench" => {
                opts.command = Command::Bench;
                idx = 1;
            }
            "watch" => {
                opts.command = Command::Watch;
                idx = 1;
            }
            "completions" => {
                let shell_str = tokens
                    .get(1)
                    .ok_or_else(|| "usage: completions <bash|zsh|fish|nushell>".to_string())?;
                let shell = ShellKind::parse(shell_str)
                    .ok_or_else(|| format!("unsupported shell: {shell_str}"))?;
                opts.command = Command::Completions(shell);
                return Ok(opts);
            }
            "preset" => {
                return parse_preset_subcommand(&tokens[1..]);
            }
            _ => {}
        }
    }

    let mut force_positional = false;
    while idx < tokens.len() {
        let tok = &tokens[idx];
        if tok == "--" {
            force_positional = true;
            idx += 1;
            continue;
        }

        if force_positional || tok == "-" || !tok.starts_with('-') {
            classify_positional(tok, &mut opts);
            idx += 1;
            continue;
        }

        let (flag, inline_val) = if let Some((f, v)) = tok.split_once('=') {
            (f, Some(v))
        } else {
            (tok.as_str(), None)
        };

        let next_val = |cur_idx: &mut usize| -> Result<String, String> {
            if let Some(v) = inline_val {
                return Ok(v.to_string());
            }
            *cur_idx += 1;
            tokens
                .get(*cur_idx)
                .cloned()
                .ok_or_else(|| format!("flag {flag} requires a value"))
        };

        let optional_val = |cur_idx: &mut usize| -> Option<String> {
            if let Some(v) = inline_val {
                return Some(v.to_string());
            }
            if let Some(nxt) = tokens.get(*cur_idx + 1)
                && !nxt.starts_with('-')
                && !Path::new(nxt).exists()
                && TargetGeometry::parse(nxt).is_err()
            {
                *cur_idx += 1;
                return Some(nxt.clone());
            }
            None
        };

        match flag {
            "-h" | "--help" => opts.help_requested = true,
            "-V" | "--version" => opts.version_requested = true,
            "-p" | "--preset" => {
                let _ = next_val(&mut idx)?;
            }
            "-t" | "--target" => {
                let v = next_val(&mut idx)?;
                for part in v.split(',') {
                    let trimmed = part.trim();
                    if !trimmed.is_empty() {
                        opts.target_specs.push(trimmed.to_string());
                    }
                }
            }
            "--screen" => {
                let val = optional_val(&mut idx).unwrap_or_else(|| "auto".to_string());
                opts.screen = match val.to_ascii_lowercase().as_str() {
                    "auto" => ScreenTarget::Auto,
                    "all" => ScreenTarget::All,
                    "per-screen" | "perscreen" => ScreenTarget::PerScreen,
                    "span" => ScreenTarget::Span,
                    _ => ScreenTarget::Output(val),
                };
            }
            "-a" | "--algo" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.resample.filter = ResampleFilter::parse(&v)
                    .ok_or_else(|| format!("unknown resampling algorithm: {v}"))?;
            }
            "--ibp-iters" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.resample.ibp_iters = v
                    .parse::<u8>()
                    .map_err(|_| format!("invalid --ibp-iters: {v}"))?
                    .clamp(0, 6);
            }
            "--antiring" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.resample.antiring = parse_unit_or_pct(&v, "--antiring")?;
            }
            "--sigmoidal" => {
                let v = optional_val(&mut idx)
                    .map(|s| s.parse::<f32>().map_err(|_| format!("invalid --sigmoidal: {s}")))
                    .transpose()?
                    .unwrap_or(6.5);
                opts.pipeline.resample.sigmoidal_strength = Some(v.clamp(1.0, 12.0));
            }
            "--linear" => {
                opts.pipeline.resample.working_space = WorkingSpace::LinearSrgb;
                opts.pipeline.resample.sigmoidal_strength = None;
            }
            "--oklab" => {
                opts.pipeline.resample.working_space = WorkingSpace::Oklab;
                opts.pipeline.resample.sigmoidal_strength = None;
            }
            "--premultiply" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.resample.premultiply = parse_bool_str(&v, "--premultiply")?;
            }
            "--alpha-fix" => opts.pipeline.resample.fix_alpha_bleed = true,
            "--no-alpha-fix" => opts.pipeline.resample.fix_alpha_bleed = false,
            "--chroma" => {
                let v = next_val(&mut idx)?;
                match v.to_ascii_lowercase().as_str() {
                    "krig" | "krigbilateral" => opts.pipeline.resample.use_krig_chroma = true,
                    "lanczos" | "default" => opts.pipeline.resample.use_krig_chroma = false,
                    _ => return Err(format!("invalid --chroma mode: {v}")),
                }
            }
            "--dejpeg" => {
                let v = optional_val(&mut idx)
                    .map(|s| parse_unit_or_pct(&s, "--dejpeg"))
                    .transpose()?
                    .unwrap_or(0.6);
                opts.pipeline.dejpeg_strength = v;
            }
            "--denoise" => {
                let v = optional_val(&mut idx)
                    .map(|s| parse_unit_or_pct(&s, "--denoise"))
                    .transpose()?
                    .unwrap_or(0.4);
                opts.pipeline.denoise_strength = v;
            }
            "--deband" => {
                let v = optional_val(&mut idx)
                    .map(|s| parse_unit_or_pct(&s, "--deband"))
                    .transpose()?
                    .unwrap_or(0.5);
                opts.pipeline.deband_strength = v;
            }
            "--sharpen" | "--rcas" => {
                let v = optional_val(&mut idx)
                    .map(|s| parse_unit_or_pct(&s, flag))
                    .transpose()?
                    .unwrap_or(0.5);
                opts.pipeline.resample.rcas_sharpness = v;
            }
            "--descaled" => opts.pipeline.auto_descale = true,
            "--dither" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.dither_mode = DitherMode::parse(&v)
                    .ok_or_else(|| format!("unknown --dither mode: {v}"))?;
            }
            "--grain" => {
                let v = optional_val(&mut idx)
                    .map(|s| parse_unit_or_pct(&s, "--grain"))
                    .transpose()?
                    .unwrap_or(0.35);
                opts.pipeline.grain_amount = v;
            }
            "-m" | "--fit" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.fit.mode = FitMode::parse(&v)
                    .ok_or_else(|| format!("unknown --fit mode: {v}"))?;
            }
            "--gravity" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.fit.gravity = Gravity::parse(&v)
                    .ok_or_else(|| format!("invalid --gravity value: {v}"))?;
            }
            "--safe-zone" => {
                let v = next_val(&mut idx)?;
                for part in v.split(',') {
                    opts.safe_zone_rules.push(parse_safe_zone_rule(part)?);
                }
            }
            "--trim-borders" => opts.pipeline.trim_borders = true,
            "--bg-color" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.fit.bg_color = parse_hex_color(&v)
                    .ok_or_else(|| format!("invalid --bg-color value: {v}"))?;
            }
            "--bg-blur" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.fit.bg_blur_radius = v
                    .trim_end_matches("px")
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --bg-blur: {v}"))?;
            }
            "--bg-dim" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.fit.bg_dim = parse_unit_or_pct(&v, "--bg-dim")?;
            }
            "--shadow" => {
                let v = optional_val(&mut idx)
                    .map(|s| parse_unit_or_pct(&s, "--shadow"))
                    .transpose()?
                    .unwrap_or(0.35);
                opts.pipeline.fit.shadow = v;
            }
            "--radius" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.fit.corner_radius = v
                    .trim_end_matches("px")
                    .parse::<f32>()
                    .map_err(|_| format!("invalid --radius: {v}"))?;
            }
            "--dim" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.fit.foreground_dim = parse_unit_or_pct(&v, "--dim")?;
            }
            "--vignette" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.fit.vignette = parse_unit_or_pct(&v, "--vignette")?;
            }
            "--bezel" => {
                let v = next_val(&mut idx)?;
                opts.bezel_px = v
                    .trim_end_matches("px")
                    .parse::<u32>()
                    .map_err(|_| format!("invalid --bezel: {v}"))?;
            }
            "--palette" => {
                let v = optional_val(&mut idx).unwrap_or_else(|| "json".to_string());
                opts.palette_out = match v.to_ascii_lowercase().as_str() {
                    "json" => PaletteFormat::Json,
                    "css" => PaletteFormat::Css,
                    _ => return Err(format!("invalid --palette format: {v}")),
                };
                opts.pipeline.extract_palette = 5;
            }
            "--set" => opts.set_wallpaper = true,
            "--alongside" => opts.replace = false,
            "-i" | "--replace" | "--in-place" => opts.replace = true,
            "--backup" => opts.backup = true,
            "-o" | "--out" => opts.out_path = Some(next_val(&mut idx)?),
            "--name" => opts.name_template = Some(next_val(&mut idx)?),
            "--suffix" => opts.suffix = Some(next_val(&mut idx)?),
            "--on-conflict" => {
                let v = next_val(&mut idx)?;
                opts.on_conflict = match v.to_ascii_lowercase().as_str() {
                    "skip" => ConflictPolicy::Skip,
                    "overwrite" => ConflictPolicy::Overwrite,
                    "rename" => ConflictPolicy::Rename,
                    "smaller" => ConflictPolicy::Smaller,
                    _ => return Err(format!("invalid --on-conflict policy: {v}")),
                };
            }
            "-f" | "--format" => {
                let v = next_val(&mut idx)?;
                if v.eq_ignore_ascii_case("auto") {
                    opts.pipeline.output_format = None;
                } else {
                    opts.pipeline.output_format = Some(
                        ImageFormat::from_extension(&v)
                            .ok_or_else(|| format!("unknown output format: {v}"))?,
                    );
                }
            }
            "--lossless" => opts.pipeline.encode.lossless = true,
            "--optimize" => opts.pipeline.encode.optimize = true,
            "--png-quant" => {
                let c = optional_val(&mut idx)
                    .map(|s| s.parse::<u16>().map_err(|_| format!("invalid --png-quant: {s}")))
                    .transpose()?
                    .unwrap_or(256)
                    .clamp(2, 256);
                opts.pipeline.encode.png_quant_colors = Some(c);
            }
            "-q" | "--quality" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.encode.quality = v
                    .parse::<u8>()
                    .map_err(|_| format!("invalid --quality: {v}"))?
                    .clamp(1, 100);
            }
            "--target-ssim2" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.target_ssim2 = Some(
                    v.parse::<f64>()
                        .map_err(|_| format!("invalid --target-ssim2: {v}"))?
                        .clamp(10.0, 99.9),
                );
            }
            "--max-bytes" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.encode.max_bytes = Some(parse_byte_size(&v)?);
            }
            "--chroma-subsampling" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.encode.chroma_subsampling = match v.to_ascii_lowercase().as_str() {
                    "auto" => ChromaSubsampling::Auto,
                    "444" | "4:4:4" => ChromaSubsampling::S444,
                    "422" | "4:2:2" => ChromaSubsampling::S422,
                    "420" | "4:2:0" => ChromaSubsampling::S420,
                    _ => return Err(format!("invalid --chroma-subsampling: {v}")),
                };
            }
            "--ico-sizes" => {
                let v = next_val(&mut idx)?;
                let mut sizes = Vec::new();
                for part in v.split(',') {
                    let s: u32 = part
                        .trim()
                        .parse()
                        .map_err(|_| format!("invalid --ico-sizes item: {part}"))?;
                    sizes.push(s.clamp(16, 256));
                }
                if !sizes.is_empty() {
                    opts.pipeline.encode.ico_sizes = sizes;
                }
            }
            "--meta" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.encode.meta_policy = match v.to_ascii_lowercase().as_str() {
                    "keep" | "all" => MetadataPolicy::Keep,
                    "strip-privacy" | "privacy" => MetadataPolicy::StripPrivacy,
                    "strip-all" | "strip" | "none" => MetadataPolicy::StripAll,
                    _ => return Err(format!("invalid --meta policy: {v}")),
                };
            }
            "--keep-dates" => opts.keep_dates = true,
            "--verify" => opts.verify = true,
            "--stdout" => opts.stdout_mode = true,
            "-r" | "--recursive" => opts.recursive = true,
            "--manifest" => opts.manifest = Some(next_val(&mut idx)?),
            "--archive-mode" => {
                let v = next_val(&mut idx)?;
                opts.archive.mode = match v.to_ascii_lowercase().as_str() {
                    "repack" => ArchiveMode::Repack,
                    "extract" => ArchiveMode::Extract,
                    "flatten" => ArchiveMode::Flatten,
                    _ => return Err(format!("invalid --archive-mode: {v}")),
                };
            }
            "--archive-format" => {
                let v = next_val(&mut idx)?;
                if v.eq_ignore_ascii_case("auto") {
                    opts.archive.output_format = None;
                } else {
                    opts.archive.output_format = Some(
                        ArchiveFormat::from_name(&v)
                            .ok_or_else(|| format!("unknown --archive-format: {v}"))?,
                    );
                }
            }
            "--archive-depth" => {
                let v = next_val(&mut idx)?;
                opts.archive.security.max_depth = v
                    .parse::<u8>()
                    .map_err(|_| format!("invalid --archive-depth: {v}"))?;
            }
            "--non-images" => {
                let v = next_val(&mut idx)?;
                opts.archive.non_images = match v.to_ascii_lowercase().as_str() {
                    "keep" => NonImagePolicy::Keep,
                    "drop" => NonImagePolicy::Drop,
                    "extract" => NonImagePolicy::Extract,
                    _ => return Err(format!("invalid --non-images policy: {v}")),
                };
            }
            "--dedup" => {
                let v = optional_val(&mut idx).unwrap_or_else(|| "perceptual".to_string());
                opts.archive.dedup = match v.to_ascii_lowercase().as_str() {
                    "none" | "off" => DedupMode::None,
                    "exact" => DedupMode::Exact,
                    "perceptual" | "dhash" => DedupMode::Perceptual(5),
                    other => {
                        let ham = other
                            .parse::<u32>()
                            .map_err(|_| format!("invalid --dedup value: {other}"))?;
                        DedupMode::Perceptual(ham)
                    }
                };
            }
            "--link-skipped" => opts.link_skipped = true,
            "--copy-skipped" => opts.copy_skipped = true,
            "--include" => opts.include_globs.push(next_val(&mut idx)?),
            "--exclude" => opts.exclude_globs.push(next_val(&mut idx)?),
            "--only-orientation" => {
                let v = next_val(&mut idx)?;
                opts.only_orientation = Some(match v.to_ascii_lowercase().as_str() {
                    "landscape" | "h" => OrientationFilter::Landscape,
                    "portrait" | "v" => OrientationFilter::Portrait,
                    "square" | "sq" => OrientationFilter::Square,
                    _ => return Err(format!("invalid --only-orientation: {v}")),
                });
            }
            "--min-res" => opts.min_res = Some(parse_wh_pair(&next_val(&mut idx)?)?),
            "--max-res" => opts.max_res = Some(parse_wh_pair(&next_val(&mut idx)?)?),
            "--skip-if-fits" => opts.pipeline.skip_if_fits = true,
            "--on-error" => {
                let v = next_val(&mut idx)?;
                opts.archive.on_error = match v.to_ascii_lowercase().as_str() {
                    "skip" => ArchiveErrorPolicy::Skip,
                    "copy-raw" | "copy" => ArchiveErrorPolicy::CopyRaw,
                    "fail" => ArchiveErrorPolicy::Fail,
                    _ => return Err(format!("invalid --on-error: {v}")),
                };
            }
            "--strict" => {
                opts.strict = true;
                opts.archive.on_error = ArchiveErrorPolicy::Fail;
            }
            "--fail-on" => {
                let v = next_val(&mut idx)?;
                opts.fail_level = match v.to_ascii_lowercase().as_str() {
                    "error" => FailLevel::Error,
                    "truncated" => FailLevel::Truncated,
                    "warn" | "warning" => FailLevel::Warn,
                    _ => return Err(format!("invalid --fail-on level: {v}")),
                };
            }
            "--max-entry-size" => {
                opts.archive.security.max_entry_bytes = parse_byte_size(&next_val(&mut idx)?)?;
            }
            "--max-total-size" => {
                opts.archive.security.max_total_bytes = parse_byte_size(&next_val(&mut idx)?)?;
            }
            "--max-pixels" => {
                let v = next_val(&mut idx)?;
                opts.pipeline.decode.max_pixels = v
                    .parse::<u64>()
                    .map_err(|_| format!("invalid --max-pixels: {v}"))?;
            }
            "--max-ratio" => {
                let v = next_val(&mut idx)?;
                opts.archive.security.max_ratio = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid --max-ratio: {v}"))?;
            }
            "-j" | "--threads" => {
                let v = next_val(&mut idx)?;
                opts.threads = Some(
                    v.parse::<usize>()
                        .map_err(|_| format!("invalid --threads: {v}"))?
                        .max(1),
                );
            }
            "--memory-limit" => {
                opts.memory_limit_bytes = Some(parse_byte_size(&next_val(&mut idx)?)? as usize);
            }
            "-n" | "--dry-run" => opts.dry_run = true,
            "--resume" => {
                opts.resume_path =
                    Some(optional_val(&mut idx).unwrap_or_else(|| ".resume.journal".to_string()));
            }
            "--watch" => opts.command = Command::Watch,
            "--iters" => {
                let v = next_val(&mut idx)?;
                opts.bench_iters = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --iters: {v}"))?
                    .max(1);
            }
            "--report" => {
                let v = next_val(&mut idx)?;
                opts.report = match v.to_ascii_lowercase().as_str() {
                    "text" => ReportFormat::Text,
                    "json" => ReportFormat::Json,
                    _ => return Err(format!("invalid --report format: {v}")),
                };
            }
            "--preview" => opts.preview = true,
            "--preview-only" => {
                opts.preview = true;
                opts.preview_only = true;
            }
            "--preview-proto" => {
                let v = next_val(&mut idx)?;
                opts.preview_proto = match v.to_ascii_lowercase().as_str() {
                    "auto" => PreviewProto::Auto,
                    "kitty" => PreviewProto::Kitty,
                    "sixel" => PreviewProto::Sixel,
                    "sextant" | "unicode" => PreviewProto::Sextant,
                    "half" | "half-block" => PreviewProto::HalfBlock,
                    _ => return Err(format!("invalid --preview-proto: {v}")),
                };
            }
            "--quiet" => opts.quiet = true,
            "--verbose" => opts.verbose = true,
            "--json" => {
                opts.json_output = true;
                opts.report = ReportFormat::Json;
            }
            other if other.starts_with("-j") && other.len() > 2 => {
                let n = other[2..]
                    .parse::<usize>()
                    .map_err(|_| format!("invalid thread count in {other}"))?
                    .max(1);
                opts.threads = Some(n);
            }
            _ => return Err(format!("unrecognized flag: {tok}")),
        }

        idx += 1;
    }

    if opts.palette_out != PaletteFormat::None && opts.pipeline.extract_palette == 0 {
        opts.pipeline.extract_palette = 5;
    }

    for spec in &opts.target_specs {
        let parsed = TargetGeometry::parse(spec).map_err(|e| e.to_string())?;
        opts.pipeline.targets.push(parsed);
    }

    if !opts.safe_zone_rules.is_empty() {
        apply_safe_zone_rules(&mut opts);
    }

    Ok(opts)
}

fn classify_positional(tok: &str, opts: &mut CliOptions) {
    if tok == "-" {
        opts.inputs.push(tok.to_string());
        return;
    }
    if !Path::new(tok).exists() && !matches!(opts.command, Command::Inspect | Command::Compare) {
        let parts: Vec<&str> = tok
            .split(',')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .collect();
        if !parts.is_empty() && parts.iter().all(|p| TargetGeometry::parse(p).is_ok()) {
            for p in parts {
                opts.target_specs.push(p.to_string());
            }
            return;
        }
    }
    opts.inputs.push(tok.to_string());
}

fn parse_preset_subcommand(args: &[String]) -> Result<CliOptions, String> {
    let mut opts = CliOptions::default();
    let Some(action) = args.first().map(|s| s.to_ascii_lowercase()) else {
        opts.command = Command::Preset(PresetAction::List);
        return Ok(opts);
    };
    let parsed = match action.as_str() {
        "list" | "ls" => PresetAction::List,
        "path" => PresetAction::Path,
        "show" => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| "usage: preset show <name>".to_string())?;
            PresetAction::Show(name)
        }
        "delete" | "rm" => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| "usage: preset delete <name>".to_string())?;
            PresetAction::Delete(name)
        }
        "save" | "set" => {
            let name = args
                .get(1)
                .cloned()
                .ok_or_else(|| "usage: preset save <name> <flags...>".to_string())?;
            let spec = args[2..].join(" ");
            if spec.trim().is_empty() {
                return Err("usage: preset save <name> <flags...>".to_string());
            }
            PresetAction::Save { name, spec }
        }
        other => return Err(format!("unknown preset action: {other}")),
    };
    opts.command = Command::Preset(parsed);
    Ok(opts)
}

fn parse_unit_or_pct(raw: &str, flag: &str) -> Result<f32, String> {
    let t = raw.trim();
    if let Some(p) = t.strip_suffix('%') {
        let v: f32 = p.parse().map_err(|_| format!("invalid percentage for {flag}: {raw}"))?;
        return Ok((v / 100.0).clamp(0.0, 1.0));
    }
    let v: f32 = t.parse().map_err(|_| format!("invalid value for {flag}: {raw}"))?;
    if v > 1.0 {
        Ok((v / 100.0).clamp(0.0, 1.0))
    } else {
        Ok(v.clamp(0.0, 1.0))
    }
}

fn parse_bool_str(raw: &str, flag: &str) -> Result<bool, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Ok(true),
        "0" | "false" | "off" | "no" => Ok(false),
        _ => Err(format!("invalid boolean for {flag}: {raw}")),
    }
}

fn parse_wh_pair(raw: &str) -> Result<(u32, u32), String> {
    let norm = raw.replace(['х', '×', '*'], "x");
    let (ws, hs) = norm
        .split_once('x')
        .ok_or_else(|| format!("expected WxH dimension pair, got: {raw}"))?;
    let w: u32 = ws
        .trim()
        .parse()
        .map_err(|_| format!("invalid width in {raw}"))?;
    let h: u32 = hs
        .trim()
        .parse()
        .map_err(|_| format!("invalid height in {raw}"))?;
    Ok((w, h))
}

fn parse_safe_zone_rule(raw: &str) -> Result<SafeZoneRule, String> {
    let (edge_s, val_s) = raw
        .trim()
        .split_once(':')
        .ok_or_else(|| format!("invalid --safe-zone item (expected edge:val): {raw}"))?;
    let edge = match edge_s.trim().to_ascii_lowercase().as_str() {
        "top" | "n" => SafeEdge::Top,
        "bottom" | "bot" | "s" => SafeEdge::Bottom,
        "left" | "w" => SafeEdge::Left,
        "right" | "e" => SafeEdge::Right,
        "all" => SafeEdge::All,
        _ => return Err(format!("unknown safe-zone edge: {edge_s}")),
    };
    let clean = val_s.trim();
    if let Some(pct) = clean.strip_suffix('%') {
        let v: f32 = pct
            .parse()
            .map_err(|_| format!("invalid percentage in safe-zone: {raw}"))?;
        Ok(SafeZoneRule {
            edge,
            value: (v / 100.0).clamp(0.0, 0.45),
            is_percent: true,
        })
    } else {
        let v: f32 = clean
            .trim_end_matches("px")
            .parse()
            .map_err(|_| format!("invalid pixel value in safe-zone: {raw}"))?;
        Ok(SafeZoneRule {
            edge,
            value: v.max(0.0),
            is_percent: false,
        })
    }
}

fn apply_safe_zone_rules(opts: &mut CliOptions) {
    let mut sz = SafeZone::default();
    for r in &opts.safe_zone_rules {
        let norm = if r.is_percent {
            r.value
        } else {
            (r.value / 1080.0).clamp(0.0, 0.4)
        };
        match r.edge {
            SafeEdge::Top => sz.top = sz.top.max(norm),
            SafeEdge::Bottom => sz.bottom = sz.bottom.max(norm),
            SafeEdge::Left => sz.left = sz.left.max(norm),
            SafeEdge::Right => sz.right = sz.right.max(norm),
            SafeEdge::All => {
                sz.top = sz.top.max(norm);
                sz.bottom = sz.bottom.max(norm);
                sz.left = sz.left.max(norm);
                sz.right = sz.right.max(norm);
            }
        }
    }
    opts.pipeline.fit.safe_zone = sz;
}

pub fn parse_byte_size(raw: &str) -> Result<u64, String> {
    let t = raw.trim().to_ascii_lowercase();
    let split_idx = t
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(t.len());
    let (num_str, unit) = t.split_at(split_idx);
    let num: f64 = num_str
        .parse()
        .map_err(|_| format!("invalid byte size: {raw}"))?;
    let mul: f64 = match unit.trim() {
        "" | "b" => 1.0,
        "k" | "kb" => 1_000.0,
        "ki" | "kib" => 1024.0,
        "m" | "mb" => 1_000_000.0,
        "mi" | "mib" => 1024.0 * 1024.0,
        "g" | "gb" => 1_000_000_000.0,
        "gi" | "gib" => 1024.0 * 1024.0 * 1024.0,
        other => return Err(format!("unknown byte unit '{other}' in {raw}")),
    };
    Ok((num * mul).round().max(1.0) as u64)
}

pub fn format_help(program: &str) -> String {
    format!(
        "{program} - pure-Rust lossless-grade image scaler, wallpaper fitter, forensics & archive processor\n\n\
         USAGE:\n\
         \x20 {program} <inputs...> [target] [flags]\n\
         \x20 {program} process|fit|archive|inspect|compare|bench|watch|preset|completions ...\n\n\
         TARGET DSL:\n\
         \x20 1920x1080, 1920х1080, 1920*1080, 1920:1080, 4k, uwqhd, fhd, qhd, 8k\n\
         \x20 1920w, 1080h, 2x, 150%, ar:21:9, mp:12, >, <, ^, !, comma-lists, --screen [auto|DP-1|span]\n\n\
         SCALING & RESTORATION:\n\
         \x20 -a, --algo <auto|adaptive|fsr|ibp|ewa-lanczos|ewa-lanczossharp|ewa-robidoux|lanczos3|\n\
         \x20             lanczos4|ssim-down|ssim-up|icbi|nedi|lineart|mmpx|xbrz|scalefx|mitchell|catmull|box|nearest>\n\
         \x20 --linear | --oklab | --sigmoidal [k] | --chroma <krig|lanczos> | --antiring <0..1>\n\
         \x20 --dejpeg [pct] | --denoise [pct] | --deband [pct] | --sharpen [pct] | --descaled\n\
         \x20 --dither <none|bayer|floyd|blue-noise> | --grain [pct]\n\n\
         FITTING & DISPLAY:\n\
         \x20 -m, --fit <auto|smart-crop|seam-carve|ambient|extend|mirror|gradient|contain|stretch|tile|split>\n\
         \x20 --gravity <auto|c|n|s|e|w|ne|nw|se|sw|x%,y%> | --safe-zone <edge:px|%> | --trim-borders\n\
         \x20 --bg-color <#hex|oled> | --bg-blur <px> | --bg-dim <%> | --shadow | --radius <px>\n\
         \x20 --dim <%> | --vignette <%> | --bezel <px> | --palette [json|css] | --set\n\n\
         OUTPUT, ARCHIVES & AUTOMATION:\n\
         \x20 --alongside (default) | -i, --replace [--backup] | -o, --out <path> | --name <tpl>\n\
         \x20 -f, --format <auto|png|jpg|webp|qoi|tiff|bmp|tga|pnm|ico|ff|hdr|exr> | --lossless | --optimize\n\
         \x20 --png-quant [N] | -q, --quality <1..100> | --target-ssim2 <score> | --max-bytes <size>\n\
         \x20 --meta <keep|strip-privacy|strip-all> | --keep-dates | --verify | --stdout\n\
         \x20 -r | --manifest <file> | --archive-mode <repack|extract|flatten> | --archive-format <fmt>\n\
         \x20 --non-images <keep|drop|extract> | --dedup [exact|perceptual] | --link-skipped | --copy-skipped\n\
         \x20 --include <glob> | --exclude <glob> | --skip-if-fits | --resume | --watch | --preview\n\
         \x20 -j <threads> | --memory-limit <size> | -p, --preset <name> | --json | --dry-run"
    )
}

pub fn generate_completions(shell: ShellKind, program: &str) -> String {
    let flags = "--algo --fit --target --screen --out --name --suffix --replace --alongside \
                 --backup --format --lossless --optimize --png-quant --quality --target-ssim2 \
                 --max-bytes --meta --keep-dates --verify --stdout --recursive --manifest \
                 --archive-mode --archive-format --non-images --dedup --include --exclude \
                 --skip-if-fits --dejpeg --denoise --deband --sharpen --descaled --dither \
                 --grain --gravity --safe-zone --trim-borders --bg-color --bg-blur --bg-dim \
                 --shadow --radius --dim --vignette --bezel --palette --set --threads \
                 --memory-limit --dry-run --resume --watch --preview --preview-only --preset \
                 --json --quiet --verbose --help --version";
    match shell {
        ShellKind::Bash => format!(
            "_{program}_complete() {{\n\
             \x20 local cur=\"${{COMP_WORDS[COMP_CWORD]}}\"\n\
             \x20 if [[ \"$cur\" == -* ]]; then\n\
             \x20   COMPREPLY=( $(compgen -W \"{flags}\" -- \"$cur\") )\n\
             \x20 else\n\
             \x20   COMPREPLY=( $(compgen -f -- \"$cur\") )\n\
             \x20 fi\n\
             }}\n\
             complete -F _{program}_complete {program}"
        ),
        ShellKind::Zsh => format!(
            "#compdef {program}\n\
             local -a opts\n\
             opts=({})\n\
             _arguments '*:files:_files' $opts",
            flags
                .split_whitespace()
                .map(|f| format!("'{f}'"))
                .collect::<Vec<_>>()
                .join(" ")
        ),
        ShellKind::Fish => flags
            .split_whitespace()
            .filter_map(|f| f.strip_prefix("--"))
            .map(|f| format!("complete -c {program} -l {f}"))
            .collect::<Vec<_>>()
            .join("\n"),
        ShellKind::Nushell => format!(
            "export extern \"{program}\" [\n\
             \x20 ...args: string\n\
             \x20 --algo(-a): string\n\
             \x20 --fit(-m): string\n\
             \x20 --target(-t): string\n\
             \x20 --format(-f): string\n\
             \x20 --out(-o): path\n\
             \x20 --replace(-i)\n\
             \x20 --lossless\n\
             \x20 --optimize\n\
             \x20 --set\n\
             ]"
        ),
    }
}
