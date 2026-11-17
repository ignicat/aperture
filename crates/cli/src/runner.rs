use crate::args::{
    CliOptions, Command, OrientationFilter, PaletteFormat, PresetAction, ReportFormat,
    ScreenTarget, format_help, generate_completions, parse_args,
};
use crate::config::{delete_preset, load_presets, preset_file_path, save_preset};
use crate::discover::{DiscoveredInput, DiscoveryConfig, discover_inputs, is_image_candidate};
use crate::journal::ResumeJournal;
use crate::linux::display::{ConnectedMonitor, detect_monitors};
use crate::linux::fs::{passthrough_file, replace_input_atomic, write_output_atomic};
use crate::linux::wallpaper::set_wallpaper_for_output;
use crate::preview::render_terminal_preview;
use crate::progress::{ItemSummary, ProgressTracker};
use crate::watch::DirectoryWatcher;
use engine::archive::{
    ArchiveMode, DedupMode, process_archive_file, process_archive_bytes,
};
use engine::arena::{MemoryBudget, PlanCache};
use engine::classify::{compute_dhash_64, dhash_distance, fnv1a_128};
use engine::codec::{DecodeOptions, decode_image, encode_image};
use engine::color::{WorkingSpace, global_luts};
use engine::geometry::fit::{FitMode, extract_oklab_palette, split_multi_monitor};
use engine::geometry::spec::TargetGeometry;
use engine::inspect::inspect_image_bytes;
use engine::inspect::metrics::{compare_images, export_palette_css, export_palette_json};
use engine::pipeline::{ProcessedImageOutput, process_image_bytes};
use engine::pixel::{PixelFormat, PlanarF32};
use engine::pool::{ThreadPool, effective_cpu_count};
use engine::resample::kernels::ResampleFilter;
use engine::resample::{ResampleConfig, resample_planar};
use std::collections::HashSet;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const EXIT_OK: u8 = 0;
pub const EXIT_FATAL: u8 = 1;
pub const EXIT_ARGS: u8 = 2;
pub const EXIT_PARTIAL: u8 = 3;
pub const EXIT_SECURITY: u8 = 4;

pub fn execute(program: &str, raw_args: &[String]) -> u8 {
    let mut opts = match parse_args(program, raw_args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}");
            return EXIT_ARGS;
        }
    };

    if opts.help_requested {
        println!("{}", format_help(program));
        return EXIT_OK;
    }
    if opts.version_requested {
        println!("{program} {}", engine::version());
        return EXIT_OK;
    }

    match opts.command.clone() {
        Command::Completions(shell) => {
            println!("{}", generate_completions(shell, program));
            EXIT_OK
        }
        Command::Preset(action) => run_preset_command(program, &action),
        Command::Inspect => run_inspect_command(&opts),
        Command::Compare => run_compare_command(&opts),
        Command::Bench => run_bench_command(&opts),
        Command::Watch => run_watch_command(&mut opts),
        Command::Process | Command::Fit | Command::Archive => run_batch_command(&mut opts),
    }
}

fn build_engine_context(opts: &CliOptions) -> (MemoryBudget, PlanCache, ThreadPool) {
    let threads = opts.threads.unwrap_or_else(effective_cpu_count);
    let budget = match opts.memory_limit_bytes {
        Some(limit) => MemoryBudget::new(limit),
        None => MemoryBudget::from_system_fraction(1, 2),
    };
    let cache = PlanCache::new(64);
    let pool = ThreadPool::new(threads);
    (budget, cache, pool)
}

fn run_preset_command(program: &str, action: &PresetAction) -> u8 {
    match action {
        PresetAction::List => {
            let map = load_presets(program);
            for (k, v) in map {
                println!("{k} = {v}");
            }
            EXIT_OK
        }
        PresetAction::Path => {
            match preset_file_path(program) {
                Some(p) => {
                    println!("{}", p.display());
                    EXIT_OK
                }
                None => {
                    eprintln!("error: HOME/XDG_CONFIG_HOME is unset");
                    EXIT_FATAL
                }
            }
        }
        PresetAction::Show(name) => {
            let map = load_presets(program);
            if let Some(v) = map.get(name) {
                println!("{name} = {v}");
                EXIT_OK
            } else {
                eprintln!("error: preset '{name}' not found");
                EXIT_FATAL
            }
        }
        PresetAction::Save { name, spec } => match save_preset(program, name, spec) {
            Ok(path) => {
                println!("saved preset '{name}' to {}", path.display());
                EXIT_OK
            }
            Err(e) => {
                eprintln!("error: {e}");
                EXIT_FATAL
            }
        },
        PresetAction::Delete(name) => match delete_preset(program, name) {
            Ok(true) => {
                println!("deleted preset '{name}'");
                EXIT_OK
            }
            Ok(false) => {
                eprintln!("error: preset '{name}' not found in user config");
                EXIT_FATAL
            }
            Err(e) => {
                eprintln!("error: {e}");
                EXIT_FATAL
            }
        },
    }
}

fn run_inspect_command(opts: &CliOptions) -> u8 {
    if opts.inputs.is_empty() && opts.manifest.is_none() {
        eprintln!("error: inspect requires at least one image path or '-'");
        return EXIT_ARGS;
    }
    let disc_cfg = DiscoveryConfig {
        recursive: opts.recursive,
        include_globs: opts.include_globs.clone(),
        exclude_globs: opts.exclude_globs.clone(),
    };
    let inputs = match discover_inputs(&opts.inputs, opts.manifest.as_deref(), &disc_cfg) {
        Ok(list) => list,
        Err(e) => {
            eprintln!("error: {e}");
            return EXIT_FATAL;
        }
    };
    if inputs.is_empty() {
        eprintln!("error: no matching images found");
        return EXIT_FATAL;
    }

    let (_, cache, pool) = build_engine_context(opts);
    let mut reports = Vec::new();
    let mut had_err = false;

    for inp in inputs {
        let (label, bytes, ext) = match read_input_bytes(&inp) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}");
                had_err = true;
                continue;
            }
        };
        match inspect_image_bytes(
            &bytes,
            ext.as_deref(),
            opts.pipeline.decode.max_pixels,
            &cache,
            &pool,
        ) {
            Ok(rep) => {
                if opts.preview || opts.preview_only {
                    let _ = render_terminal_preview(
                        &bytes,
                        ext.as_deref(),
                        opts.preview_proto,
                        &cache,
                        &pool,
                    );
                }
                reports.push((label, rep));
            }
            Err(e) => {
                eprintln!("error inspecting {label}: {e}");
                had_err = true;
            }
        }
    }

    if reports.is_empty() {
        return EXIT_FATAL;
    }

    if opts.report == ReportFormat::Json || opts.json_output {
        if reports.len() == 1 {
            println!("{}", reports[0].1.to_json());
        } else {
            let items: Vec<String> = reports
                .iter()
                .map(|(lbl, r)| format!("  \"{}\": {}", escape_json_basic(lbl), r.to_json()))
                .collect();
            println!("{{\n{}\n}}", items.join(",\n"));
        }
    } else {
        for (lbl, r) in &reports {
            println!("=== {lbl} ===");
            println!("{}", r.to_text());
            match opts.palette_out {
                PaletteFormat::Json => println!("{}", export_palette_json(&r.palette)),
                PaletteFormat::Css => println!("{}", export_palette_css(&r.palette)),
                PaletteFormat::None => {}
            }
        }
    }

    if had_err { EXIT_PARTIAL } else { EXIT_OK }
}

fn run_compare_command(opts: &CliOptions) -> u8 {
    if opts.inputs.len() != 2 {
        eprintln!("error: compare requires exactly 2 image files: <reference> <candidate>");
        return EXIT_ARGS;
    }
    let (_, cache, pool) = build_engine_context(opts);
    let read_planar = |p: &str| -> Result<PlanarF32, String> {
        let bytes = fs::read(p).map_err(|e| format!("failed to read {p}: {e}"))?;
        let ext = Path::new(p).extension().and_then(|s| s.to_str());
        let dec = decode_image(&bytes, ext, &opts.pipeline.decode).map_err(|e| e.to_string())?;
        PlanarF32::from_packed(&dec.image, WorkingSpace::LinearSrgb, global_luts())
            .map_err(|e| e.to_string())
    };

    let ref_img = match read_planar(&opts.inputs[0]) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return EXIT_FATAL;
        }
    };
    let mut cmp_img = match read_planar(&opts.inputs[1]) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return EXIT_FATAL;
        }
    };

    let same_dims = ref_img.width == cmp_img.width && ref_img.height == cmp_img.height;
    if !same_dims {
        let rcfg = ResampleConfig {
            filter: opts.pipeline.resample.filter,
            ..Default::default()
        };
        match resample_planar(&cmp_img, ref_img.width, ref_img.height, &rcfg, &cache, &pool) {
            Ok(resized) => cmp_img = resized,
            Err(e) => {
                eprintln!("error aligning candidate resolution: {e}");
                return EXIT_FATAL;
            }
        }
    }

    let metrics = match compare_images(&ref_img, &cmp_img) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error comparing images: {e}");
            return EXIT_FATAL;
        }
    };

    if opts.report == ReportFormat::Json || opts.json_output {
        println!(
            "{{\n  \"width\": {},\n  \"height\": {},\n  \"aligned\": {},\n  \"mse\": {:.8},\n  \"psnr_db\": {:.3},\n  \"ssim\": {:.6},\n  \"ssim2_est\": {:.3},\n  \"butteraugli_est\": {:.4},\n  \"dssim_est\": {:.6}\n}}",
            metrics.width,
            metrics.height,
            !same_dims,
            metrics.mse,
            metrics.psnr_db,
            metrics.ssim,
            metrics.ssim2_est,
            metrics.butteraugli_est.unwrap_or(0.0),
            metrics.dssim_est.unwrap_or(0.0),
        );
    } else {
        println!(
            "compare {}x{} (aligned={}):\n  psnr: {:.3} dB | ssim: {:.5} | ssim2_est: {:.2}\n  butteraugli_est: {:.3} | dssim_est: {:.5}",
            metrics.width,
            metrics.height,
            !same_dims,
            metrics.psnr_db,
            metrics.ssim,
            metrics.ssim2_est,
            metrics.butteraugli_est.unwrap_or(0.0),
            metrics.dssim_est.unwrap_or(0.0),
        );
    }
    EXIT_OK
}

fn run_bench_command(opts: &CliOptions) -> u8 {
    let Some(first) = opts.inputs.first() else {
        eprintln!("error: bench requires an input image path");
        return EXIT_ARGS;
    };
    let bytes = match fs::read(first) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error reading {first}: {e}");
            return EXIT_FATAL;
        }
    };
    let ext = Path::new(first).extension().and_then(|s| s.to_str());
    let dec = match decode_image(&bytes, ext, &opts.pipeline.decode) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error decoding {first}: {e}");
            return EXIT_FATAL;
        }
    };
    let src = match PlanarF32::from_packed(&dec.image, WorkingSpace::LinearSrgb, global_luts()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error converting image: {e}");
            return EXIT_FATAL;
        }
    };

    let (dst_w, dst_h) = match opts.pipeline.targets.first() {
        Some(t) => match t.resolve(src.width, src.height) {
            Ok(r) => (r.width, r.height),
            Err(e) => {
                eprintln!("error resolving target: {e}");
                return EXIT_ARGS;
            }
        },
        None => ((src.width * 2).min(7680), (src.height * 2).min(4320)),
    };

    let (_, cache, pool) = build_engine_context(opts);
    let algos = if opts.pipeline.resample.filter == ResampleFilter::Auto {
        vec![
            ResampleFilter::Fsr,
            ResampleFilter::EwaLanczosSharp,
            ResampleFilter::Lanczos3,
            ResampleFilter::SsimUp,
            ResampleFilter::Ibp,
            ResampleFilter::Mitchell,
        ]
    } else {
        vec![opts.pipeline.resample.filter]
    };

    if !opts.json_output && opts.report == ReportFormat::Text {
        println!(
            "benchmarking {}x{} -> {}x{} ({} iters, {} threads)",
            src.width,
            src.height,
            dst_w,
            dst_h,
            opts.bench_iters,
            opts.threads.unwrap_or_else(effective_cpu_count),
        );
    }

    let mut json_rows = Vec::new();
    for algo in algos {
        let cfg = ResampleConfig {
            filter: algo,
            ..opts.pipeline.resample.clone()
        };
        let mut best_ms = f64::MAX;
        let mut total_ms = 0.0f64;
        let mut ssim2 = 0.0f64;

        for iter in 0..opts.bench_iters {
            let t0 = Instant::now();
            match resample_planar(&src, dst_w, dst_h, &cfg, &cache, &pool) {
                Ok(out) => {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    best_ms = best_ms.min(ms);
                    total_ms += ms;
                    if iter == 0 {
                        let back_cfg = ResampleConfig {
                            filter: ResampleFilter::SsimDown,
                            ..Default::default()
                        };
                        if let Ok(back) =
                            resample_planar(&out, src.width, src.height, &back_cfg, &cache, &pool)
                            && let Ok(m) = compare_images(&src, &back)
                        {
                            ssim2 = m.ssim2_est;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("error benchmarking {algo:?}: {e}");
                    return EXIT_FATAL;
                }
            }
        }
        let avg_ms = total_ms / opts.bench_iters as f64;
        let mp_s = ((dst_w as f64 * dst_h as f64) / 1_000_000.0) / (best_ms / 1000.0).max(1e-6);
        let name = filter_slug(algo);
        if opts.json_output || opts.report == ReportFormat::Json {
            json_rows.push(format!(
                "  {{\"algo\": \"{name}\", \"best_ms\": {best_ms:.2}, \"avg_ms\": {avg_ms:.2}, \"mp_per_sec\": {mp_s:.2}, \"roundtrip_ssim2\": {ssim2:.2}}}"
            ));
        } else {
            println!(
                "  {name:<18} best={best_ms:>7.2} ms | avg={avg_ms:>7.2} ms | {mp_s:>6.2} MP/s | roundtrip_ssim2={ssim2:>5.2}"
            );
        }
    }

    if opts.json_output || opts.report == ReportFormat::Json {
        println!("[\n{}\n]", json_rows.join(",\n"));
    }
    EXIT_OK
}

fn run_watch_command(opts: &mut CliOptions) -> u8 {
    if opts.inputs.is_empty() && opts.manifest.is_none() {
        eprintln!("error: watch requires at least one directory path");
        return EXIT_ARGS;
    }
    let dirs: Vec<PathBuf> = opts
        .inputs
        .iter()
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
        .collect();
    if dirs.is_empty() {
        eprintln!("error: watch requires at least one existing directory");
        return EXIT_ARGS;
    }

    let mut watcher = match DirectoryWatcher::new(&dirs, opts.recursive) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("error initializing watcher: {e}");
            return EXIT_FATAL;
        }
    };

    if !opts.quiet {
        eprintln!(
            "watching {} directory(ies) (interval {} ms)...",
            dirs.len(),
            opts.watch_interval_ms
        );
    }

    let (budget, cache, pool) = build_engine_context(opts);
    let interval = Duration::from_millis(opts.watch_interval_ms.max(100));

    loop {
        let changed = watcher.poll_changes(interval);
        for path in changed {
            if !is_image_candidate(&path) {
                continue;
            }
            let input = DiscoveredInput::Image {
                path: path.clone(),
                root: path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf(),
            };
            let mut batch_dedup = DedupState::default();
            let mut emitted_paths = Vec::new();
            let res = process_single_discovered(
                &input,
                opts,
                &[],
                &budget,
                &cache,
                &pool,
                &mut batch_dedup,
                &mut emitted_paths,
            );
            for ep in emitted_paths {
                watcher.ignore_output(&ep);
            }
            match res {
                Ok(summaries) => {
                    for s in summaries {
                        if !opts.quiet {
                            eprintln!(
                                "watch: {} -> {} ({}x{} -> {}x{}, {})",
                                s.input,
                                s.output.as_deref().unwrap_or("-"),
                                s.src_width,
                                s.src_height,
                                s.dst_width,
                                s.dst_height,
                                s.status
                            );
                        }
                    }
                }
                Err((code, msg)) => {
                    eprintln!("watch error on {}: {msg} (code {code})", path.display());
                }
            }
        }
    }
}

#[derive(Default)]
struct DedupState {
    exact: HashSet<u128>,
    perceptual: Vec<u64>,
}

fn run_batch_command(opts: &mut CliOptions) -> u8 {
    if opts.inputs.is_empty() && opts.manifest.is_none() {
        eprintln!("error: no inputs provided");
        return EXIT_ARGS;
    }

    let monitors = if opts.screen != ScreenTarget::None {
        let detected = detect_monitors();
        if detected.is_empty() {
            eprintln!("error: --screen requested, but no Wayland, X11, or DRM displays were detected");
            return EXIT_FATAL;
        }
        detected
    } else {
        Vec::new()
    };

    if matches!(opts.command, Command::Fit) && opts.pipeline.targets.is_empty() && monitors.is_empty()
    {
        let fallback = detect_monitors();
        if let Some(m) = fallback.first() {
            opts.pipeline.targets.push(TargetGeometry::Exact {
                width: m.width,
                height: m.height,
                constraint: engine::geometry::spec::SizeConstraint::Any,
            });
        } else {
            eprintln!("error: fit requires a target resolution (e.g. 1920x1080) or --screen");
            return EXIT_ARGS;
        }
    }

    match &opts.screen {
        ScreenTarget::Auto => {
            if let Some(m) = monitors.iter().find(|m| m.primary).or_else(|| monitors.first()) {
                opts.pipeline.targets.push(TargetGeometry::Exact {
                    width: m.width,
                    height: m.height,
                    constraint: engine::geometry::spec::SizeConstraint::Any,
                });
            }
        }
        ScreenTarget::Output(name) => {
            let found = monitors
                .iter()
                .find(|m| m.name.eq_ignore_ascii_case(name))
                .ok_or_else(|| format!("display output '{name}' not found"));
            match found {
                Ok(m) => opts.pipeline.targets.push(TargetGeometry::Exact {
                    width: m.width,
                    height: m.height,
                    constraint: engine::geometry::spec::SizeConstraint::Any,
                }),
                Err(e) => {
                    eprintln!("error: {e}");
                    return EXIT_FATAL;
                }
            }
        }
        ScreenTarget::All | ScreenTarget::PerScreen => {
            for m in &monitors {
                opts.pipeline.targets.push(TargetGeometry::Exact {
                    width: m.width,
                    height: m.height,
                    constraint: engine::geometry::spec::SizeConstraint::Any,
                });
            }
        }
        ScreenTarget::Span => {
            opts.pipeline.fit.mode = FitMode::Split;
        }
        ScreenTarget::None => {}
    }

    let disc_cfg = DiscoveryConfig {
        recursive: opts.recursive,
        include_globs: opts.include_globs.clone(),
        exclude_globs: opts.exclude_globs.clone(),
    };
    let discovered = match discover_inputs(&opts.inputs, opts.manifest.as_deref(), &disc_cfg) {
        Ok(list) => list,
        Err(e) => {
            eprintln!("error: {e}");
            return EXIT_FATAL;
        }
    };
    if discovered.is_empty() {
        eprintln!("error: no matching images or archives found");
        return EXIT_FATAL;
    }

    if opts.stdout_mode && discovered.len() > 1 {
        eprintln!("error: --stdout supports only a single input");
        return EXIT_ARGS;
    }
    if opts.replace && opts.stdout_mode {
        eprintln!("error: --replace cannot be combined with --stdout");
        return EXIT_ARGS;
    }

    let mut journal = match &opts.resume_path {
        Some(jpath) => match ResumeJournal::open(Path::new(jpath)) {
            Ok(j) => Some(j),
            Err(e) => {
                eprintln!("error opening resume journal: {e}");
                return EXIT_FATAL;
            }
        },
        None => None,
    };

    let (budget, cache, pool) = build_engine_context(opts);
    let mut tracker = ProgressTracker::new(discovered.len(), opts.quiet || opts.stdout_mode);
    let mut all_summaries = Vec::new();
    let mut dedup_state = DedupState::default();
    let mut had_partial_failure = false;
    let mut security_failure = false;
    let single_item_mode = discovered.len() == 1;

    for inp in &discovered {
        let resume_key = match inp {
            DiscoveredInput::Image { path, .. } | DiscoveredInput::Archive { path, .. } => {
                Some(ResumeJournal::fingerprint(path))
            }
            DiscoveredInput::Stdin => None,
        };
        if let (Some(j), Some(key)) = (journal.as_ref(), resume_key.as_ref())
            && j.is_completed(key)
        {
            let skip_sum = ItemSummary {
                input: input_display(inp),
                output: None,
                src_width: 0,
                src_height: 0,
                dst_width: 0,
                dst_height: 0,
                in_bytes: 0,
                out_bytes: 0,
                status: "skipped".to_string(),
                message: Some("resumed-already-complete".to_string()),
            };
            tracker.record(&skip_sum);
            all_summaries.push(skip_sum);
            continue;
        }

        let mut emitted = Vec::new();
        match process_single_discovered(
            inp,
            opts,
            &monitors,
            &budget,
            &cache,
            &pool,
            &mut dedup_state,
            &mut emitted,
        ) {
            Ok(items) => {
                if let (Some(j), Some(key)) = (journal.as_mut(), resume_key.as_ref()) {
                    j.mark_completed(key);
                }
                for item in items {
                    tracker.record(&item);
                    if item.status == "failed" {
                        had_partial_failure = true;
                    }
                    all_summaries.push(item);
                }
            }
            Err((code, msg)) => {
                if code == EXIT_SECURITY {
                    security_failure = true;
                }
                if single_item_mode || opts.strict || security_failure {
                    tracker.finish_bar();
                    eprintln!("error: {msg}");
                    return code;
                }
                had_partial_failure = true;
                let fail_sum = ItemSummary {
                    input: input_display(inp),
                    output: None,
                    src_width: 0,
                    src_height: 0,
                    dst_width: 0,
                    dst_height: 0,
                    in_bytes: 0,
                    out_bytes: 0,
                    status: "failed".to_string(),
                    message: Some(msg),
                };
                tracker.record(&fail_sum);
                all_summaries.push(fail_sum);
            }
        }
    }

    tracker.finish_bar();

    if !opts.stdout_mode {
        if opts.report == ReportFormat::Json || opts.json_output {
            println!("{}", tracker.format_summary_json(&all_summaries));
        } else if !opts.quiet {
            println!("{}", tracker.format_summary_text());
        }
    }

    if security_failure {
        EXIT_SECURITY
    } else if had_partial_failure {
        EXIT_PARTIAL
    } else {
        EXIT_OK
    }
}

#[allow(clippy::too_many_arguments)]
fn process_single_discovered(
    inp: &DiscoveredInput,
    opts: &CliOptions,
    monitors: &[ConnectedMonitor],
    budget: &MemoryBudget,
    cache: &PlanCache,
    pool: &ThreadPool,
    dedup_state: &mut DedupState,
    emitted_paths: &mut Vec<PathBuf>,
) -> Result<Vec<ItemSummary>, (u8, String)> {
    match inp {
        DiscoveredInput::Archive { path, root } => {
            process_archive_input(path, root, opts, budget, cache, pool, emitted_paths)
        }
        DiscoveredInput::Stdin => {
            let mut bytes = Vec::new();
            io::stdin()
                .read_to_end(&mut bytes)
                .map_err(|e| (EXIT_FATAL, format!("failed reading stdin: {e}")))?;

            if let Some(arc_fmt) = engine::archive::detect_archive_format(&bytes, None) {
                if opts.dry_run {
                    return Ok(vec![ItemSummary {
                        input: "-".to_string(),
                        output: opts.out_path.clone(),
                        src_width: 0,
                        src_height: 0,
                        dst_width: 0,
                        dst_height: 0,
                        in_bytes: bytes.len() as u64,
                        out_bytes: 0,
                        status: "skipped".to_string(),
                        message: Some("dry-run stdin archive".to_string()),
                    }]);
                }
                let out = process_archive_bytes(
                    &bytes,
                    Some(arc_fmt.canonical_extension()),
                    &opts.pipeline,
                    &opts.archive,
                    budget,
                    cache,
                    pool,
                )
                .map_err(map_engine_err)?;
                let data = out.repacked_bytes.unwrap_or_default();
                if opts.stdout_mode || opts.out_path.is_none() {
                    io::stdout()
                        .write_all(&data)
                        .map_err(|e| (EXIT_FATAL, format!("stdout write failed: {e}")))?;
                } else if let Some(ref op) = opts.out_path {
                    write_output_atomic(Path::new(op), &data, opts.on_conflict, None, false)
                        .map_err(|e| (EXIT_FATAL, e))?;
                }
                return Ok(vec![ItemSummary {
                    input: "-".to_string(),
                    output: opts.out_path.clone(),
                    src_width: 0,
                    src_height: 0,
                    dst_width: 0,
                    dst_height: 0,
                    in_bytes: bytes.len() as u64,
                    out_bytes: data.len() as u64,
                    status: "processed".to_string(),
                    message: None,
                }]);
            }

            let out_vec = process_image_bytes(&bytes, None, &opts.pipeline, budget, cache, pool)
                .map_err(map_engine_err)?;
            let mut summaries = Vec::new();
            for (idx, item) in out_vec.iter().enumerate() {
                if opts.preview || opts.preview_only {
                    let _ = render_terminal_preview(
                        &item.bytes,
                        Some(item.format.canonical_extension()),
                        opts.preview_proto,
                        cache,
                        pool,
                    );
                }
                if opts.preview_only {
                    continue;
                }
                if opts.stdout_mode || opts.out_path.is_none() {
                    if idx == 0 {
                        io::stdout()
                            .write_all(&item.bytes)
                            .map_err(|e| (EXIT_FATAL, format!("stdout write failed: {e}")))?;
                    }
                } else if let Some(ref out_p) = opts.out_path {
                    let dst = resolve_stdin_output_path(Path::new(out_p), item, idx);
                    let (final_p, written) = write_output_atomic(
                        &dst,
                        &item.bytes,
                        opts.on_conflict,
                        None,
                        opts.verify,
                    )
                    .map_err(|e| (exit_verify_or_fatal(opts.verify), e))?;
                    if written {
                        emitted_paths.push(final_p.clone());
                    }
                }
                summaries.push(ItemSummary {
                    input: "-".to_string(),
                    output: if opts.stdout_mode {
                        Some("stdout".to_string())
                    } else {
                        opts.out_path.clone()
                    },
                    src_width: item.src_width,
                    src_height: item.src_height,
                    dst_width: item.dst_width,
                    dst_height: item.dst_height,
                    in_bytes: bytes.len() as u64,
                    out_bytes: item.bytes.len() as u64,
                    status: if item.skipped { "skipped" } else { "processed" }.to_string(),
                    message: None,
                });
            }
            Ok(summaries)
        }
        DiscoveredInput::Image { path, root } => process_image_file_input(
            path,
            root,
            opts,
            monitors,
            budget,
            cache,
            pool,
            dedup_state,
            emitted_paths,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn process_image_file_input(
    path: &Path,
    root: &Path,
    opts: &CliOptions,
    monitors: &[ConnectedMonitor],
    budget: &MemoryBudget,
    cache: &PlanCache,
    pool: &ThreadPool,
    dedup_state: &mut DedupState,
    emitted_paths: &mut Vec<PathBuf>,
) -> Result<Vec<ItemSummary>, (u8, String)> {
    let bytes =
        fs::read(path).map_err(|e| (EXIT_FATAL, format!("failed to read {}: {e}", path.display())))?;
    let in_len = bytes.len() as u64;
    let ext = path.extension().and_then(|s| s.to_str());

    let pre_dec = decode_image(
        &bytes,
        ext,
        &DecodeOptions {
            max_pixels: opts.pipeline.decode.max_pixels,
            auto_orient: false,
            convert_icc_to_srgb: false,
            target_svg_size: None,
        },
    )
    .map_err(map_engine_err)?;
    let (sw, sh) = (pre_dec.image.width, pre_dec.image.height);

    if let Some(req_orient) = opts.only_orientation {
        let match_ok = match req_orient {
            OrientationFilter::Landscape => sw > sh,
            OrientationFilter::Portrait => sh > sw,
            OrientationFilter::Square => sw == sh,
        };
        if !match_ok {
            let mut out_opt = None;
            if (opts.link_skipped || opts.copy_skipped) && let Some(ref op) = opts.out_path {
                let dst = mirror_destination(path, root, Path::new(op), opts.suffix.as_deref(), ext);
                if !opts.dry_run
                    && passthrough_file(path, &dst, opts.link_skipped, opts.copy_skipped, opts.keep_dates)
                        .map_err(|e| (EXIT_FATAL, e))?
                {
                    out_opt = Some(dst.to_string_lossy().into_owned());
                }
            }
            return Ok(vec![ItemSummary {
                input: path.to_string_lossy().into_owned(),
                output: out_opt,
                src_width: sw,
                src_height: sh,
                dst_width: sw,
                dst_height: sh,
                in_bytes: in_len,
                out_bytes: in_len,
                status: "skipped".to_string(),
                message: Some("filtered-by-orientation".to_string()),
            }]);
        }
    }

    if let Some((min_w, min_h)) = opts.min_res
        && (sw < min_w || sh < min_h)
    {
        return Ok(vec![skipped_filter_summary(path, sw, sh, in_len, "below-min-res")]);
    }
    if let Some((max_w, max_h)) = opts.max_res
        && (sw > max_w || sh > max_h)
    {
        return Ok(vec![skipped_filter_summary(path, sw, sh, in_len, "above-max-res")]);
    }

    match opts.archive.dedup {
        DedupMode::None => {}
        DedupMode::Exact => {
            let h = fnv1a_128(&bytes);
            if !dedup_state.exact.insert(h) {
                return Ok(vec![skipped_filter_summary(path, sw, sh, in_len, "dedup-exact")]);
            }
        }
        DedupMode::Perceptual(max_ham) => {
            let p = PlanarF32::from_packed(&pre_dec.image, WorkingSpace::LinearSrgb, global_luts())
                .map_err(map_engine_err)?;
            let dh = compute_dhash_64(&p);
            if dedup_state
                .perceptual
                .iter()
                .any(|&seen| dhash_distance(seen, dh) <= max_ham)
            {
                return Ok(vec![skipped_filter_summary(
                    path,
                    sw,
                    sh,
                    in_len,
                    "dedup-perceptual",
                )]);
            }
            dedup_state.perceptual.push(dh);
        }
    }

    if opts.screen == ScreenTarget::Span && !monitors.is_empty() {
        return process_span_wallpaper_input(
            path,
            root,
            &bytes,
            ext,
            monitors,
            opts,
            cache,
            pool,
            emitted_paths,
        );
    }

    if opts.dry_run {
        return Ok(vec![ItemSummary {
            input: path.to_string_lossy().into_owned(),
            output: opts.out_path.clone(),
            src_width: sw,
            src_height: sh,
            dst_width: sw,
            dst_height: sh,
            in_bytes: in_len,
            out_bytes: 0,
            status: "skipped".to_string(),
            message: Some("dry-run".to_string()),
        }]);
    }

    let outputs = process_image_bytes(&bytes, ext, &opts.pipeline, budget, cache, pool)
        .map_err(map_engine_err)?;
    let mut summaries = Vec::with_capacity(outputs.len());

    for (idx, item) in outputs.iter().enumerate() {
        if opts.preview || opts.preview_only {
            let _ = render_terminal_preview(
                &item.bytes,
                Some(item.format.canonical_extension()),
                opts.preview_proto,
                cache,
                pool,
            );
        }
        if opts.palette_out != PaletteFormat::None {
            let pal = if item.palette.is_empty() {
                if let Ok(dec) = decode_image(&item.bytes, Some(item.format.canonical_extension()), &opts.pipeline.decode)
                    && let Ok(pl) = PlanarF32::from_packed(&dec.image, WorkingSpace::LinearSrgb, global_luts())
                {
                    extract_oklab_palette(&pl, 5)
                } else {
                    Vec::new()
                }
            } else {
                item.palette.clone()
            };
            match opts.palette_out {
                PaletteFormat::Json => eprintln!("{}", export_palette_json(&pal)),
                PaletteFormat::Css => eprintln!("{}", export_palette_css(&pal)),
                PaletteFormat::None => {}
            }
        }
        if opts.preview_only {
            summaries.push(ItemSummary {
                input: path.to_string_lossy().into_owned(),
                output: None,
                src_width: item.src_width,
                src_height: item.src_height,
                dst_width: item.dst_width,
                dst_height: item.dst_height,
                in_bytes: in_len,
                out_bytes: item.bytes.len() as u64,
                status: "processed".to_string(),
                message: Some("preview-only".to_string()),
            });
            continue;
        }
        if opts.stdout_mode {
            if idx == 0 {
                io::stdout()
                    .write_all(&item.bytes)
                    .map_err(|e| (EXIT_FATAL, format!("stdout write failed: {e}")))?;
            }
            summaries.push(ItemSummary {
                input: path.to_string_lossy().into_owned(),
                output: Some("stdout".to_string()),
                src_width: item.src_width,
                src_height: item.src_height,
                dst_width: item.dst_width,
                dst_height: item.dst_height,
                in_bytes: in_len,
                out_bytes: item.bytes.len() as u64,
                status: if item.skipped { "skipped" } else { "processed" }.to_string(),
                message: None,
            });
            continue;
        }

        if item.skipped {
            let out_p = resolve_image_output_path(path, root, item, opts, None);
            if (opts.link_skipped || opts.copy_skipped) && out_p != path {
                passthrough_file(path, &out_p, opts.link_skipped, opts.copy_skipped, opts.keep_dates)
                    .map_err(|e| (EXIT_FATAL, e))?;
            }
            summaries.push(ItemSummary {
                input: path.to_string_lossy().into_owned(),
                output: Some(out_p.to_string_lossy().into_owned()),
                src_width: item.src_width,
                src_height: item.src_height,
                dst_width: item.dst_width,
                dst_height: item.dst_height,
                in_bytes: in_len,
                out_bytes: in_len,
                status: "skipped".to_string(),
                message: Some("skip-if-fits".to_string()),
            });
            continue;
        }

        let screen_label = if matches!(opts.screen, ScreenTarget::All | ScreenTarget::PerScreen) {
            monitors.get(idx).map(|m| m.name.as_str())
        } else if let ScreenTarget::Output(ref s) = opts.screen {
            Some(s.as_str())
        } else {
            None
        };

        let dst_path = resolve_image_output_path(path, root, item, opts, screen_label);
        let written_path = if opts.replace && outputs.len() == 1 {
            replace_input_atomic(
                path,
                &dst_path,
                &item.bytes,
                opts.backup,
                opts.keep_dates,
                opts.verify,
            )
            .map_err(|e| (exit_verify_or_fatal(opts.verify), e))?;
            emitted_paths.push(dst_path.clone());
            dst_path
        } else {
            let keep_src = if opts.keep_dates { Some(path) } else { None };
            let (final_p, written) = write_output_atomic(
                &dst_path,
                &item.bytes,
                opts.on_conflict,
                keep_src,
                opts.verify,
            )
            .map_err(|e| (exit_verify_or_fatal(opts.verify), e))?;
            if written {
                emitted_paths.push(final_p.clone());
            }
            final_p
        };

        if opts.set_wallpaper && idx == 0 {
            if let Err(e) = set_wallpaper_for_output(&written_path, screen_label) {
                eprintln!("warning: wallpaper set failed: {e}");
            }
        } else if opts.set_wallpaper && screen_label.is_some() {
            let _ = set_wallpaper_for_output(&written_path, screen_label);
        }

        summaries.push(ItemSummary {
            input: path.to_string_lossy().into_owned(),
            output: Some(written_path.to_string_lossy().into_owned()),
            src_width: item.src_width,
            src_height: item.src_height,
            dst_width: item.dst_width,
            dst_height: item.dst_height,
            in_bytes: in_len,
            out_bytes: item.bytes.len() as u64,
            status: "processed".to_string(),
            message: if item.was_descaled {
                Some("descaled".to_string())
            } else {
                None
            },
        });
    }

    Ok(summaries)
}

#[allow(clippy::too_many_arguments)]
fn process_span_wallpaper_input(
    path: &Path,
    root: &Path,
    bytes: &[u8],
    ext: Option<&str>,
    monitors: &[ConnectedMonitor],
    opts: &CliOptions,
    cache: &PlanCache,
    pool: &ThreadPool,
    emitted_paths: &mut Vec<PathBuf>,
) -> Result<Vec<ItemSummary>, (u8, String)> {
    let dec = decode_image(bytes, ext, &opts.pipeline.decode).map_err(map_engine_err)?;
    let src = PlanarF32::from_packed(&dec.image, WorkingSpace::LinearSrgb, global_luts())
        .map_err(map_engine_err)?;
    let regions: Vec<_> = monitors.iter().map(ConnectedMonitor::to_region).collect();
    let mut fit_cfg = opts.pipeline.fit.clone();
    fit_cfg.mode = FitMode::SmartCrop;
    let splits = split_multi_monitor(
        &src,
        &regions,
        opts.bezel_px,
        &fit_cfg,
        &opts.pipeline.resample,
        cache,
        pool,
    )
    .map_err(map_engine_err)?;

    let out_fmt = opts
        .pipeline
        .output_format
        .unwrap_or_else(|| dec.format.default_output_format(opts.pipeline.encode.lossless));
    let mut summaries = Vec::with_capacity(splits.len());

    for (mon_name, tile) in splits {
        let packed = tile
            .to_packed(
                if tile.has_alpha {
                    PixelFormat::Rgba8
                } else {
                    PixelFormat::Rgb8
                },
                global_luts(),
            )
            .map_err(map_engine_err)?;
        let out_bytes = encode_image(&packed, out_fmt, &opts.pipeline.encode, Some(&dec.metadata))
            .map_err(map_engine_err)?;
        let item = ProcessedImageOutput {
            bytes: out_bytes,
            format: out_fmt,
            src_width: src.width,
            src_height: src.height,
            dst_width: tile.width,
            dst_height: tile.height,
            skipped: false,
            was_descaled: false,
            dhash_64: 0,
            exact_hash_128: 0,
            palette: Vec::new(),
            estimated_ssim2: None,
        };
        let dst_path = resolve_image_output_path(path, root, &item, opts, Some(&mon_name));
        let keep_src = if opts.keep_dates { Some(path) } else { None };
        let (final_p, written) =
            write_output_atomic(&dst_path, &item.bytes, opts.on_conflict, keep_src, opts.verify)
                .map_err(|e| (exit_verify_or_fatal(opts.verify), e))?;
        if written {
            emitted_paths.push(final_p.clone());
        }
        if opts.set_wallpaper {
            let _ = set_wallpaper_for_output(&final_p, Some(&mon_name));
        }
        summaries.push(ItemSummary {
            input: path.to_string_lossy().into_owned(),
            output: Some(final_p.to_string_lossy().into_owned()),
            src_width: src.width,
            src_height: src.height,
            dst_width: tile.width,
            dst_height: tile.height,
            in_bytes: bytes.len() as u64,
            out_bytes: item.bytes.len() as u64,
            status: "processed".to_string(),
            message: Some(format!("span:{mon_name}")),
        });
    }

    Ok(summaries)
}

fn process_archive_input(
    path: &Path,
    _root: &Path,
    opts: &CliOptions,
    budget: &MemoryBudget,
    cache: &PlanCache,
    pool: &ThreadPool,
    emitted_paths: &mut Vec<PathBuf>,
) -> Result<Vec<ItemSummary>, (u8, String)> {
    let in_len = fs::metadata(path).map_or(0u64, |m| m.len());
    if opts.dry_run {
        return Ok(vec![ItemSummary {
            input: path.to_string_lossy().into_owned(),
            output: opts.out_path.clone(),
            src_width: 0,
            src_height: 0,
            dst_width: 0,
            dst_height: 0,
            in_bytes: in_len,
            out_bytes: 0,
            status: "skipped".to_string(),
            message: Some("dry-run archive".to_string()),
        }]);
    }

    let outcome = process_archive_file(path, &opts.pipeline, &opts.archive, budget, cache, pool)
        .map_err(map_engine_err)?;

    if opts.archive.mode == ArchiveMode::Repack {
        let repacked = outcome.repacked_bytes.unwrap_or_default();
        let ext_str = outcome.format.canonical_extension();
        let dst_path = resolve_archive_output_path(path, ext_str, opts);
        let out_bytes_len = repacked.len() as u64;

        let written_path = if opts.replace {
            replace_input_atomic(
                path,
                &dst_path,
                &repacked,
                opts.backup,
                opts.keep_dates,
                false,
            )
            .map_err(|e| (EXIT_FATAL, e))?;
            dst_path
        } else {
            let keep_src = if opts.keep_dates { Some(path) } else { None };
            let (fp, written) =
                write_output_atomic(&dst_path, &repacked, opts.on_conflict, keep_src, false)
                    .map_err(|e| (EXIT_FATAL, e))?;
            if written {
                emitted_paths.push(fp.clone());
            }
            fp
        };

        return Ok(vec![ItemSummary {
            input: path.to_string_lossy().into_owned(),
            output: Some(written_path.to_string_lossy().into_owned()),
            src_width: 0,
            src_height: 0,
            dst_width: 0,
            dst_height: 0,
            in_bytes: in_len,
            out_bytes: out_bytes_len,
            status: if outcome.stats.failed_entries > 0 {
                "failed"
            } else {
                "processed"
            }
            .to_string(),
            message: Some(format!(
                "images={} skipped={} non_img={} dedup={} nested={}",
                outcome.stats.images_processed,
                outcome.stats.images_skipped,
                outcome.stats.non_images_kept,
                outcome.stats.duplicates_removed,
                outcome.stats.nested_archives,
            )),
        }]);
    }

    let base_dir = if let Some(ref out_p) = opts.out_path {
        PathBuf::from(out_p)
    } else {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("archive");
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .join(format!("{stem}_extracted"))
    };

    let mut total_out = 0u64;
    for entry in &outcome.extracted_entries {
        let dst = base_dir.join(&entry.path);
        let (final_p, written) =
            write_output_atomic(&dst, &entry.data, opts.on_conflict, None, false)
                .map_err(|e| (EXIT_FATAL, e))?;
        if written {
            emitted_paths.push(final_p);
        }
        total_out += entry.data.len() as u64;
    }

    Ok(vec![ItemSummary {
        input: path.to_string_lossy().into_owned(),
        output: Some(base_dir.to_string_lossy().into_owned()),
        src_width: 0,
        src_height: 0,
        dst_width: 0,
        dst_height: 0,
        in_bytes: in_len,
        out_bytes: total_out,
        status: "processed".to_string(),
        message: Some(format!("extracted {} entries", outcome.extracted_entries.len())),
    }])
}

fn resolve_image_output_path(
    src_path: &Path,
    root: &Path,
    item: &ProcessedImageOutput,
    opts: &CliOptions,
    screen_name: Option<&str>,
) -> PathBuf {
    let ext = item.format.canonical_extension();
    let stem = src_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");
    let orient = if item.dst_width > item.dst_height {
        "landscape"
    } else if item.dst_height > item.dst_width {
        "portrait"
    } else {
        "square"
    };

    if opts.replace && opts.out_path.is_none() && opts.name_template.is_none() && screen_name.is_none()
    {
        return src_path.with_extension(ext);
    }

    let rel_dir = src_path
        .parent()
        .and_then(|p| p.strip_prefix(root).ok())
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();

    let rendered_name = if let Some(ref tpl) = opts.name_template {
        let rel_path = if rel_dir.is_empty() {
            stem.to_string()
        } else {
            format!("{rel_dir}/{stem}")
        };
        let mut s = tpl
            .replace("{root}", &root.to_string_lossy())
            .replace("{dir}", &rel_dir)
            .replace("{path}", &rel_path)
            .replace("{stem}", stem)
            .replace("{ext}", ext)
            .replace("{w}", &item.dst_width.to_string())
            .replace("{h}", &item.dst_height.to_string())
            .replace("{algo}", filter_slug(opts.pipeline.resample.filter))
            .replace("{orientation}", orient)
            .replace("{screen}", screen_name.unwrap_or("default"));
        if !tpl.starts_with('/') {
            s = s.trim_start_matches('/').to_string();
        }
        s = s.replace("//", "/");
        if !tpl.contains("{ext}") && !s.ends_with(&format!(".{ext}")) {
            s.push('.');
            s.push_str(ext);
        }
        s
    } else if let Some(ref suf) = opts.suffix {
        if let Some(scr) = screen_name {
            format!("{stem}_{scr}{suf}.{ext}")
        } else {
            format!("{stem}{suf}.{ext}")
        }
    } else if let Some(scr) = screen_name {
        format!("{stem}_{scr}_{}x{}.{ext}", item.dst_width, item.dst_height)
    } else if item.skipped {
        format!("{stem}.{ext}")
    } else if item.dst_width != item.src_width || item.dst_height != item.src_height {
        format!("{stem}_{}x{}.{ext}", item.dst_width, item.dst_height)
    } else {
        format!("{stem}_opt.{ext}")
    };

    if let Some(ref out_spec) = opts.out_path {
        let out_p = Path::new(out_spec);
        let looks_like_dir = out_spec.ends_with('/') || out_p.is_dir();
        if !looks_like_dir && opts.inputs.len() == 1 && screen_name.is_none() && opts.pipeline.targets.len() <= 1
        {
            return out_p.to_path_buf();
        }
        if opts.name_template.is_some() {
            return out_p.join(rendered_name);
        }
        if !rel_dir.is_empty() {
            return out_p.join(rel_dir).join(rendered_name);
        }
        return out_p.join(rendered_name);
    }

    let parent = src_path.parent().unwrap_or_else(|| Path::new("."));
    parent.join(rendered_name)
}

fn resolve_stdin_output_path(base: &Path, item: &ProcessedImageOutput, idx: usize) -> PathBuf {
    if base.is_dir() {
        base.join(format!(
            "stdin_{}_{}x{}.{}",
            idx,
            item.dst_width,
            item.dst_height,
            item.format.canonical_extension()
        ))
    } else {
        base.to_path_buf()
    }
}

fn resolve_archive_output_path(src: &Path, out_ext: &str, opts: &CliOptions) -> PathBuf {
    if opts.replace {
        return strip_multi_archive_ext(src, out_ext);
    }
    let stem_clean = strip_archive_stem(src);
    let fname = match &opts.suffix {
        Some(suf) => format!("{stem_clean}{suf}.{out_ext}"),
        None => format!("{stem_clean}_scaled.{out_ext}"),
    };
    if let Some(ref out_spec) = opts.out_path {
        let op = Path::new(out_spec);
        if out_spec.ends_with('/') || op.is_dir() {
            return op.join(fname);
        }
        return op.to_path_buf();
    }
    src.parent()
        .unwrap_or_else(|| Path::new("."))
        .join(fname)
}

fn strip_archive_stem(path: &Path) -> String {
    let fname = path.file_name().and_then(|s| s.to_str()).unwrap_or("archive");
    for suf in [".tar.gz", ".tar.zst", ".tar.xz", ".tar.bz2"] {
        if let Some(base) = fname.strip_suffix(suf) {
            return base.to_string();
        }
    }
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("archive")
        .to_string()
}

fn strip_multi_archive_ext(path: &Path, new_ext: &str) -> PathBuf {
    let stem = strip_archive_stem(path);
    path.parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{stem}.{new_ext}"))
}

fn mirror_destination(
    src: &Path,
    root: &Path,
    out_dir: &Path,
    suffix: Option<&str>,
    ext: Option<&str>,
) -> PathBuf {
    let rel = src.strip_prefix(root).unwrap_or_else(|_| {
        Path::new(src.file_name().unwrap_or_default())
    });
    if let Some(suf) = suffix {
        let stem = rel.file_stem().and_then(|s| s.to_str()).unwrap_or("file");
        let name = match ext {
            Some(e) => format!("{stem}{suf}.{e}"),
            None => format!("{stem}{suf}"),
        };
        let parent = rel.parent().unwrap_or_else(|| Path::new(""));
        out_dir.join(parent).join(name)
    } else {
        out_dir.join(rel)
    }
}

fn skipped_filter_summary(
    path: &Path,
    w: u32,
    h: u32,
    in_len: u64,
    reason: &str,
) -> ItemSummary {
    ItemSummary {
        input: path.to_string_lossy().into_owned(),
        output: None,
        src_width: w,
        src_height: h,
        dst_width: w,
        dst_height: h,
        in_bytes: in_len,
        out_bytes: 0,
        status: "skipped".to_string(),
        message: Some(reason.to_string()),
    }
}

fn read_input_bytes(inp: &DiscoveredInput) -> Result<(String, Vec<u8>, Option<String>), String> {
    match inp {
        DiscoveredInput::Image { path, .. } | DiscoveredInput::Archive { path, .. } => {
            let bytes = fs::read(path)
                .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .map(str::to_string);
            Ok((path.to_string_lossy().into_owned(), bytes, ext))
        }
        DiscoveredInput::Stdin => {
            let mut bytes = Vec::new();
            io::stdin()
                .read_to_end(&mut bytes)
                .map_err(|e| format!("failed to read stdin: {e}"))?;
            Ok(("-".to_string(), bytes, None))
        }
    }
}

fn input_display(inp: &DiscoveredInput) -> String {
    match inp {
        DiscoveredInput::Image { path, .. } | DiscoveredInput::Archive { path, .. } => {
            path.to_string_lossy().into_owned()
        }
        DiscoveredInput::Stdin => "-".to_string(),
    }
}

fn map_engine_err(err: engine::error::EngineError) -> (u8, String) {
    let code = match err {
        engine::error::EngineError::SecurityLimit(_) => EXIT_SECURITY,
        engine::error::EngineError::InvalidSpec(_) => EXIT_ARGS,
        _ => EXIT_FATAL,
    };
    (code, err.to_string())
}

fn exit_verify_or_fatal(verify: bool) -> u8 {
    if verify { EXIT_SECURITY } else { EXIT_FATAL }
}

fn filter_slug(f: ResampleFilter) -> &'static str {
    match f {
        ResampleFilter::Auto => "auto",
        ResampleFilter::Adaptive => "adaptive",
        ResampleFilter::Fsr => "fsr",
        ResampleFilter::Ibp => "ibp",
        ResampleFilter::EwaLanczos => "ewa-lanczos",
        ResampleFilter::EwaLanczosSharp => "ewa-lanczossharp",
        ResampleFilter::EwaRobidoux => "ewa-robidoux",
        ResampleFilter::EwaRobidouxSharp => "ewa-robidouxsharp",
        ResampleFilter::EwaGinseng => "ewa-ginseng",
        ResampleFilter::Lanczos3 => "lanczos3",
        ResampleFilter::Lanczos4 => "lanczos4",
        ResampleFilter::SsimDown => "ssim-down",
        ResampleFilter::SsimUp => "ssim-up",
        ResampleFilter::Icbi => "icbi",
        ResampleFilter::Nedi => "nedi",
        ResampleFilter::LineArt => "lineart",
        ResampleFilter::Mmpx => "mmpx",
        ResampleFilter::Xbrz => "xbrz",
        ResampleFilter::ScaleFx => "scalefx",
        ResampleFilter::Mitchell => "mitchell",
        ResampleFilter::CatmullRom => "catmull",
        ResampleFilter::Hermite => "hermite",
        ResampleFilter::Box => "box",
        ResampleFilter::Nearest => "nearest",
    }
}

fn escape_json_basic(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
