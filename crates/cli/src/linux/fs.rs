use crate::args::ConflictPolicy;
use engine::codec::{DecodeOptions, decode_image};
use rustix::fs::{AtFlags, CWD, Timespec, Timestamps, ioctl_ficlone, utimensat};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::UNIX_EPOCH;

static TEMP_SEQ: AtomicU64 = AtomicU64::new(1);

pub fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create directory {}: {e}", parent.display()))?;
    }
    Ok(())
}

pub fn write_output_atomic(
    dst_path: &Path,
    data: &[u8],
    conflict: ConflictPolicy,
    keep_dates_from: Option<&Path>,
    verify: bool,
) -> Result<(PathBuf, bool), String> {
    let resolved = resolve_conflict_path(dst_path, conflict, data.len())?;
    let Some(final_path) = resolved else {
        return Ok((dst_path.to_path_buf(), false));
    };

    ensure_parent_dir(&final_path)?;
    write_via_same_fs_temp(&final_path, data, verify)?;

    if let Some(src_meta_path) = keep_dates_from {
        preserve_file_metadata(src_meta_path, &final_path);
    }
    Ok((final_path, true))
}

pub fn replace_input_atomic(
    src_path: &Path,
    target_path: &Path,
    data: &[u8],
    backup: bool,
    keep_dates: bool,
    verify: bool,
) -> Result<(), String> {
    let orig_meta = fs::metadata(src_path).ok();
    ensure_parent_dir(target_path)?;

    let parent = target_path.parent().unwrap_or_else(|| Path::new("."));
    let tmp_path = temp_sibling_path(parent, target_path);

    let write_res = (|| -> Result<(), String> {
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
            .map_err(|e| format!("failed creating temp file {}: {e}", tmp_path.display()))?;
        f.write_all(data)
            .map_err(|e| format!("failed writing temp file: {e}"))?;
        f.flush()
            .map_err(|e| format!("failed flushing temp file: {e}"))?;
        if verify {
            verify_written_file(&tmp_path, target_path)?;
        }
        Ok(())
    })();

    if let Err(e) = write_res {
        let _ = fs::remove_file(&tmp_path);
        return Err(e);
    }

    if backup {
        let bak_path = PathBuf::from(format!("{}.bak", src_path.to_string_lossy()));
        let _ = copy_or_reflink(src_path, &bak_path);
    }

    if let Some(ref m) = orig_meta {
        let _ = fs::set_permissions(&tmp_path, m.permissions());
    }

    fs::rename(&tmp_path, target_path).map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        format!(
            "atomic replace {} -> {} failed: {e}",
            tmp_path.display(),
            target_path.display()
        )
    })?;

    if target_path != src_path {
        let _ = fs::remove_file(src_path);
    }

    if keep_dates && let Some(ref m) = orig_meta {
        apply_metadata_timestamps(m, target_path);
    }

    Ok(())
}

pub fn passthrough_file(
    src: &Path,
    dst: &Path,
    link_skipped: bool,
    copy_skipped: bool,
    keep_dates: bool,
) -> Result<bool, String> {
    if src == dst || (!link_skipped && !copy_skipped) {
        return Ok(false);
    }
    ensure_parent_dir(dst)?;
    if link_skipped && fs::hard_link(src, dst).is_ok() {
        return Ok(true);
    }
    copy_or_reflink(src, dst)?;
    if keep_dates {
        preserve_file_metadata(src, dst);
    }
    Ok(true)
}

pub fn copy_or_reflink(src: &Path, dst: &Path) -> Result<(), String> {
    if let Ok(in_f) = File::open(src)
        && let Ok(out_f) = OpenOptions::new().write(true).create(true).truncate(true).open(dst)
        && ioctl_ficlone(&out_f, &in_f).is_ok()
    {
        return Ok(());
    }
    fs::copy(src, dst).map_err(|e| {
        format!(
            "copy failed {} -> {}: {e}",
            src.display(),
            dst.display()
        )
    })?;
    Ok(())
}

fn write_via_same_fs_temp(final_path: &Path, data: &[u8], verify: bool) -> Result<(), String> {
    let parent = final_path.parent().unwrap_or_else(|| Path::new("."));
    let tmp_path = temp_sibling_path(parent, final_path);

    let res = (|| -> Result<(), String> {
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
            .map_err(|e| format!("failed creating temp file {}: {e}", tmp_path.display()))?;
        f.write_all(data)
            .map_err(|e| format!("failed writing temp file: {e}"))?;
        f.flush()
            .map_err(|e| format!("failed flushing temp file: {e}"))?;
        if verify {
            verify_written_file(&tmp_path, final_path)?;
        }
        Ok(())
    })();

    if let Err(e) = res {
        let _ = fs::remove_file(&tmp_path);
        return Err(e);
    }

    fs::rename(&tmp_path, final_path).map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        format!("atomic rename to {} failed: {e}", final_path.display())
    })
}

fn verify_written_file(tmp_path: &Path, ext_source: &Path) -> Result<(), String> {
    let bytes =
        fs::read(tmp_path).map_err(|e| format!("verify read failed {}: {e}", tmp_path.display()))?;
    let ext = ext_source.extension().and_then(|s| s.to_str());
    decode_image(
        &bytes,
        ext,
        &DecodeOptions {
            auto_orient: false,
            ..Default::default()
        },
    )
    .map_err(|e| format!("post-write verify failed for {}: {e}", ext_source.display()))?;
    Ok(())
}

fn resolve_conflict_path(
    dst: &Path,
    conflict: ConflictPolicy,
    new_len: usize,
) -> Result<Option<PathBuf>, String> {
    if !dst.exists() {
        return Ok(Some(dst.to_path_buf()));
    }
    match conflict {
        ConflictPolicy::Skip => Ok(None),
        ConflictPolicy::Overwrite => Ok(Some(dst.to_path_buf())),
        ConflictPolicy::Smaller => {
            let old_len = fs::metadata(dst).map_or(u64::MAX, |m| m.len());
            if (new_len as u64) < old_len {
                Ok(Some(dst.to_path_buf()))
            } else {
                Ok(None)
            }
        }
        ConflictPolicy::Rename => {
            let parent = dst.parent().unwrap_or_else(|| Path::new(""));
            let stem = dst
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");
            let ext = dst.extension().and_then(|s| s.to_str());
            let mut n = 1usize;
            loop {
                let name = match ext {
                    Some(e) => format!("{stem}_{n}.{e}"),
                    None => format!("{stem}_{n}"),
                };
                let cand = parent.join(name);
                if !cand.exists() {
                    return Ok(Some(cand));
                }
                n += 1;
            }
        }
    }
}

fn temp_sibling_path(parent: &Path, target: &Path) -> PathBuf {
    let pid = std::process::id();
    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let base = target
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("out");
    parent.join(format!(".{base}.{pid}.{seq}.tmp"))
}

pub fn preserve_file_metadata(src: &Path, dst: &Path) {
    if let Ok(m) = fs::metadata(src) {
        let _ = fs::set_permissions(dst, m.permissions());
        apply_metadata_timestamps(&m, dst);
    }
}

fn apply_metadata_timestamps(meta: &fs::Metadata, dst: &Path) {
    let to_spec = |st: std::io::Result<std::time::SystemTime>| -> Timespec {
        match st.and_then(|t| {
            t.duration_since(UNIX_EPOCH)
                .map_err(|_| std::io::Error::other("pre-epoch"))
        }) {
            Ok(dur) => Timespec {
                tv_sec: dur.as_secs() as i64,
                tv_nsec: dur.subsec_nanos() as i64,
            },
            Err(_) => Timespec {
                tv_sec: 0,
                tv_nsec: 0,
            },
        }
    };
    let times = Timestamps {
        last_access: to_spec(meta.accessed()),
        last_modification: to_spec(meta.modified()),
    };
    let _ = utimensat(CWD, dst, &times, AtFlags::empty());
}
