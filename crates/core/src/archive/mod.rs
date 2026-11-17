pub mod security;
pub mod tar_io;
pub mod zip_io;

use crate::arena::{MemoryBudget, PlanCache};
use crate::archive::security::{ArchiveBudgetTracker, ArchiveSecurityPolicy, sanitize_archive_path};
use crate::archive::tar_io::{read_tar_stream, write_tar_entries};
use crate::archive::zip_io::{read_zip_from_file, read_zip_from_slice, write_zip_entries};
use crate::classify::dhash_distance;
use crate::codec::{ImageFormat, detect_format};
use crate::error::{Diagnostic, EngineError, EngineResult, Severity};
use crate::pipeline::{PipelineConfig, process_image_bytes};
use crate::pool::ThreadPool;
use std::collections::HashSet;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchiveFormat {
    Zip,
    Tar,
    TarGz,
    TarZst,
    TarXz,
    TarBz2,
}

impl ArchiveFormat {
    pub fn canonical_extension(self) -> &'static str {
        match self {
            Self::Zip => "zip",
            Self::Tar => "tar",
            Self::TarGz => "tar.gz",
            Self::TarZst => "tar.zst",
            Self::TarXz => "tar.xz",
            Self::TarBz2 => "tar.bz2",
        }
    }

    pub fn from_extension(path: &str) -> Option<Self> {
        let lower = path.trim().to_ascii_lowercase();
        if lower.ends_with(".tar.gz") || lower.ends_with(".tgz") {
            return Some(Self::TarGz);
        }
        if lower.ends_with(".tar.zst") || lower.ends_with(".tzst") || lower.ends_with(".zst") {
            return Some(Self::TarZst);
        }
        if lower.ends_with(".tar.xz") || lower.ends_with(".txz") || lower.ends_with(".xz") {
            return Some(Self::TarXz);
        }
        if lower.ends_with(".tar.bz2") || lower.ends_with(".tbz2") || lower.ends_with(".tbz") {
            return Some(Self::TarBz2);
        }
        if lower.ends_with(".tar") {
            return Some(Self::Tar);
        }
        if lower.ends_with(".zip")
            || lower.ends_with(".cbz")
            || lower.ends_with(".epub")
            || lower.ends_with(".jar")
        {
            return Some(Self::Zip);
        }
        None
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "zip" | "cbz" => Some(Self::Zip),
            "tar" => Some(Self::Tar),
            "tgz" | "tar.gz" | "targz" | "gz" => Some(Self::TarGz),
            "tzst" | "tar.zst" | "tarzst" | "zst" => Some(Self::TarZst),
            "txz" | "tar.xz" | "tarxz" | "xz" => Some(Self::TarXz),
            "tbz2" | "tar.bz2" | "tarbz2" | "bz2" => Some(Self::TarBz2),
            _ => None,
        }
    }
}

pub fn detect_archive_format(bytes: &[u8], path_hint: Option<&str>) -> Option<ArchiveFormat> {
    if bytes.len() >= 4
        && (bytes[0..4] == [0x50, 0x4B, 0x03, 0x04]
            || bytes[0..4] == [0x50, 0x4B, 0x05, 0x06]
            || bytes[0..4] == [0x50, 0x4B, 0x01, 0x02])
    {
        return Some(ArchiveFormat::Zip);
    }
    if bytes.len() >= 2 && bytes[0] == 0x1F && bytes[1] == 0x8B {
        return Some(ArchiveFormat::TarGz);
    }
    if bytes.len() >= 4 && bytes[0..4] == [0x28, 0xB5, 0x2F, 0xFD] {
        return Some(ArchiveFormat::TarZst);
    }
    if bytes.len() >= 6 && bytes[0..6] == [0xFD, b'7', b'z', b'X', b'Z', 0x00] {
        return Some(ArchiveFormat::TarXz);
    }
    if bytes.len() >= 3 && &bytes[0..3] == b"BZh" {
        return Some(ArchiveFormat::TarBz2);
    }
    if bytes.len() >= 265 && &bytes[257..262] == b"ustar" {
        return Some(ArchiveFormat::Tar);
    }
    path_hint.and_then(ArchiveFormat::from_extension)
}

#[derive(Debug, Clone)]
pub struct ArchiveEntryInput {
    pub path: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ArchiveEntryOutput {
    pub path: String,
    pub data: Vec<u8>,
    pub was_image: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveMode {
    Repack,
    Extract,
    Flatten,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonImagePolicy {
    Keep,
    Drop,
    Extract,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveErrorPolicy {
    Skip,
    CopyRaw,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupMode {
    None,
    Exact,
    Perceptual(u32),
}

#[derive(Debug, Clone)]
pub struct ArchiveProcessOptions {
    pub mode: ArchiveMode,
    pub output_format: Option<ArchiveFormat>,
    pub non_images: NonImagePolicy,
    pub dedup: DedupMode,
    pub on_error: ArchiveErrorPolicy,
    pub security: ArchiveSecurityPolicy,
}

impl Default for ArchiveProcessOptions {
    fn default() -> Self {
        Self {
            mode: ArchiveMode::Repack,
            output_format: None,
            non_images: NonImagePolicy::Keep,
            dedup: DedupMode::None,
            on_error: ArchiveErrorPolicy::CopyRaw,
            security: ArchiveSecurityPolicy::default(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ArchiveStats {
    pub total_entries: usize,
    pub images_processed: usize,
    pub images_skipped: usize,
    pub non_images_kept: usize,
    pub non_images_dropped: usize,
    pub duplicates_removed: usize,
    pub nested_archives: usize,
    pub failed_entries: usize,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone)]
pub struct ArchiveProcessOutcome {
    pub format: ArchiveFormat,
    pub repacked_bytes: Option<Vec<u8>>,
    pub extracted_entries: Vec<ArchiveEntryOutput>,
    pub stats: ArchiveStats,
}

pub fn process_archive_file(
    path: &Path,
    pipeline_cfg: &PipelineConfig,
    opts: &ArchiveProcessOptions,
    budget: &MemoryBudget,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<ArchiveProcessOutcome> {
    let file = File::open(path)?;
    let path_str = path.to_string_lossy();
    let mut head = [0u8; 265];
    let head_len = {
        use std::io::Read;
        let mut hf = File::open(path)?;
        hf.read(&mut head)?
    };
    let in_fmt = detect_archive_format(&head[..head_len], Some(&path_str)).ok_or_else(|| {
        EngineError::UnsupportedFormat(format!("unrecognized archive format: {}", path.display()))
    })?;

    let mut tracker = ArchiveBudgetTracker::new(opts.security.clone());
    let entries = match in_fmt {
        ArchiveFormat::Zip => read_zip_from_file(file, &mut tracker)?,
        _ => read_tar_stream(file, in_fmt, &mut tracker)?,
    };

    process_loaded_archive(
        in_fmt,
        entries,
        pipeline_cfg,
        opts,
        budget,
        plan_cache,
        pool,
    )
}

pub fn process_archive_bytes(
    bytes: &[u8],
    path_hint: Option<&str>,
    pipeline_cfg: &PipelineConfig,
    opts: &ArchiveProcessOptions,
    budget: &MemoryBudget,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<ArchiveProcessOutcome> {
    let in_fmt = detect_archive_format(bytes, path_hint).ok_or_else(|| {
        EngineError::UnsupportedFormat("unrecognized archive byte stream".to_string())
    })?;
    let mut tracker = ArchiveBudgetTracker::new(opts.security.clone());
    let entries = read_entries_from_slice(bytes, in_fmt, &mut tracker)?;
    process_loaded_archive(
        in_fmt,
        entries,
        pipeline_cfg,
        opts,
        budget,
        plan_cache,
        pool,
    )
}

fn read_entries_from_slice(
    bytes: &[u8],
    format: ArchiveFormat,
    tracker: &mut ArchiveBudgetTracker,
) -> EngineResult<Vec<ArchiveEntryInput>> {
    match format {
        ArchiveFormat::Zip => read_zip_from_slice(bytes, tracker),
        _ => read_tar_stream(Cursor::new(bytes), format, tracker),
    }
}

fn process_loaded_archive(
    in_fmt: ArchiveFormat,
    entries: Vec<ArchiveEntryInput>,
    pipeline_cfg: &PipelineConfig,
    opts: &ArchiveProcessOptions,
    budget: &MemoryBudget,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<ArchiveProcessOutcome> {
    let out_fmt = opts.output_format.unwrap_or(in_fmt);
    let mut stats = ArchiveStats::default();
    let mut exact_seen = HashSet::new();
    let mut perceptual_seen: Vec<(u64, u64)> = Vec::new();
    let mut used_names = HashSet::new();

    let transformed = process_entries_recursive(
        entries,
        0,
        pipeline_cfg,
        opts,
        budget,
        plan_cache,
        pool,
        &mut stats,
        &mut exact_seen,
        &mut perceptual_seen,
        &mut used_names,
    )?;

    let repacked_bytes = if opts.mode == ArchiveMode::Repack {
        Some(write_archive_entries(&transformed, out_fmt)?)
    } else {
        None
    };

    Ok(ArchiveProcessOutcome {
        format: out_fmt,
        repacked_bytes,
        extracted_entries: transformed,
        stats,
    })
}

#[allow(clippy::too_many_arguments)]
fn process_entries_recursive(
    entries: Vec<ArchiveEntryInput>,
    depth: u8,
    pipeline_cfg: &PipelineConfig,
    opts: &ArchiveProcessOptions,
    budget: &MemoryBudget,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
    stats: &mut ArchiveStats,
    exact_seen: &mut HashSet<u128>,
    perceptual_seen: &mut Vec<(u64, u64)>,
    used_names: &mut HashSet<String>,
) -> EngineResult<Vec<ArchiveEntryOutput>> {
    let mut out_entries = Vec::new();

    for entry in entries {
        stats.total_entries += 1;
        let clean_path = sanitize_archive_path(&entry.path)?;

        if let Some(nested_fmt) = detect_archive_format(&entry.data, Some(&clean_path))
            && depth < opts.security.max_depth
        {
            stats.nested_archives += 1;
            let mut tracker = ArchiveBudgetTracker::new(opts.security.clone());
            match read_entries_from_slice(&entry.data, nested_fmt, &mut tracker) {
                Ok(nested_inputs) => {
                    let sub_out = process_entries_recursive(
                        nested_inputs,
                        depth + 1,
                        pipeline_cfg,
                        opts,
                        budget,
                        plan_cache,
                        pool,
                        stats,
                        exact_seen,
                        perceptual_seen,
                        used_names,
                    )?;

                    if opts.mode == ArchiveMode::Repack {
                        let repacked = write_archive_entries(&sub_out, nested_fmt)?;
                        out_entries.push(ArchiveEntryOutput {
                            path: clean_path,
                            data: repacked,
                            was_image: false,
                        });
                    } else {
                        let prefix = clean_path
                            .rsplit_once('.')
                            .map_or(clean_path.as_str(), |(base, _)| base);
                        for mut sub_item in sub_out {
                            if opts.mode == ArchiveMode::Extract {
                                sub_item.path = format!("{prefix}/{}", sub_item.path);
                            } else {
                                sub_item.path = deduplicate_output_name(&sub_item.path, used_names);
                            }
                            out_entries.push(sub_item);
                        }
                    }
                    continue;
                }
                Err(e) => {
                    if matches!(e, EngineError::SecurityLimit(_)) {
                        return Err(e);
                    }
                    handle_entry_error(
                        opts.on_error,
                        &clean_path,
                        entry.data,
                        false,
                        e,
                        stats,
                        &mut out_entries,
                    )?;
                    continue;
                }
            }
        }

        let ext = Path::new(&clean_path).extension().and_then(|s| s.to_str());
        let is_img = detect_format(&entry.data).is_some()
            || ext.and_then(ImageFormat::from_extension).is_some();

        if !is_img {
            match opts.non_images {
                NonImagePolicy::Keep | NonImagePolicy::Extract => {
                    stats.non_images_kept += 1;
                    let target_path = if opts.mode == ArchiveMode::Flatten {
                        let fname = Path::new(&clean_path)
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or(&clean_path);
                        deduplicate_output_name(fname, used_names)
                    } else {
                        clean_path
                    };
                    out_entries.push(ArchiveEntryOutput {
                        path: target_path,
                        data: entry.data,
                        was_image: false,
                    });
                }
                NonImagePolicy::Drop => {
                    stats.non_images_dropped += 1;
                }
            }
            continue;
        }

        match process_image_bytes(
            &entry.data,
            ext,
            pipeline_cfg,
            budget,
            plan_cache,
            pool,
        ) {
            Ok(processed_vec) => {
                for item in processed_vec {
                    let is_dup = match opts.dedup {
                        DedupMode::None => false,
                        DedupMode::Exact => !exact_seen.insert(item.exact_hash_128),
                        DedupMode::Perceptual(max_ham) => {
                            let hit = perceptual_seen
                                .iter()
                                .any(|&(h, _)| dhash_distance(h, item.dhash_64) <= max_ham);
                            if !hit {
                                perceptual_seen.push((
                                    item.dhash_64,
                                    (item.src_width as u64) * (item.src_height as u64),
                                ));
                            }
                            hit
                        }
                    };

                    if is_dup {
                        stats.duplicates_removed += 1;
                        continue;
                    }

                    if item.skipped {
                        stats.images_skipped += 1;
                    } else {
                        stats.images_processed += 1;
                    }

                    let formatted_path =
                        format_entry_output_path(&clean_path, item.format, opts.mode, used_names);
                    out_entries.push(ArchiveEntryOutput {
                        path: formatted_path,
                        data: item.bytes,
                        was_image: true,
                    });
                }
            }
            Err(err) => {
                if matches!(err, EngineError::SecurityLimit(_)) {
                    return Err(err);
                }
                handle_entry_error(
                    opts.on_error,
                    &clean_path,
                    entry.data,
                    true,
                    err,
                    stats,
                    &mut out_entries,
                )?;
            }
        }
    }

    Ok(out_entries)
}

fn handle_entry_error(
    policy: ArchiveErrorPolicy,
    clean_path: &str,
    raw: Vec<u8>,
    was_image: bool,
    err: EngineError,
    stats: &mut ArchiveStats,
    out_entries: &mut Vec<ArchiveEntryOutput>,
) -> EngineResult<()> {
    stats.failed_entries += 1;
    stats.diagnostics.push(Diagnostic {
        severity: if policy == ArchiveErrorPolicy::Fail {
            Severity::Error
        } else {
            Severity::Warning
        },
        source: clean_path.to_string(),
        message: err.to_string(),
    });
    match policy {
        ArchiveErrorPolicy::Skip => Ok(()),
        ArchiveErrorPolicy::CopyRaw => {
            out_entries.push(ArchiveEntryOutput {
                path: clean_path.to_string(),
                data: raw,
                was_image,
            });
            Ok(())
        }
        ArchiveErrorPolicy::Fail => Err(err),
    }
}

pub fn write_archive_entries(
    entries: &[ArchiveEntryOutput],
    format: ArchiveFormat,
) -> EngineResult<Vec<u8>> {
    match format {
        ArchiveFormat::Zip => write_zip_entries(Vec::new(), entries),
        _ => write_tar_entries(entries, format),
    }
}

fn format_entry_output_path(
    clean_path: &str,
    out_fmt: ImageFormat,
    mode: ArchiveMode,
    used_names: &mut HashSet<String>,
) -> String {
    let base_name = if mode == ArchiveMode::Flatten {
        Path::new(clean_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(clean_path)
    } else {
        clean_path
    };

    let with_ext = match base_name.rsplit_once('.') {
        Some((stem, old_ext)) => {
            let new_ext = out_fmt.canonical_extension();
            if old_ext.eq_ignore_ascii_case(new_ext)
                || (old_ext.eq_ignore_ascii_case("jpeg") && new_ext == "jpg")
            {
                base_name.to_string()
            } else {
                format!("{stem}.{new_ext}")
            }
        }
        None => format!("{base_name}.{}", out_fmt.canonical_extension()),
    };

    deduplicate_output_name(&with_ext, used_names)
}

fn deduplicate_output_name(candidate: &str, used_names: &mut HashSet<String>) -> String {
    if used_names.insert(candidate.to_string()) {
        return candidate.to_string();
    }
    let (stem, ext_opt) = match candidate.rsplit_once('.') {
        Some((s, e)) => (s, Some(e)),
        None => (candidate, None),
    };
    let mut counter = 2usize;
    loop {
        let next = match ext_opt {
            Some(e) => format!("{stem}_{counter}.{e}"),
            None => format!("{stem}_{counter}"),
        };
        if used_names.insert(next.clone()) {
            return next;
        }
        counter += 1;
    }
}
