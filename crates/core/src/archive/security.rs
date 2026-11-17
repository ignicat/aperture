use crate::error::{EngineError, EngineResult};
use std::io::Read;
use std::path::{Component, Path};

#[derive(Debug, Clone)]
pub struct ArchiveSecurityPolicy {
    pub max_entry_bytes: u64,
    pub max_total_bytes: u64,
    pub max_entries: usize,
    pub max_ratio: f64,
    pub max_depth: u8,
    pub allow_symlinks: bool,
}

impl Default for ArchiveSecurityPolicy {
    fn default() -> Self {
        Self {
            max_entry_bytes: 512 * 1024 * 1024,
            max_total_bytes: 4 * 1024 * 1024 * 1024,
            max_entries: 100_000,
            max_ratio: 100.0,
            max_depth: 4,
            allow_symlinks: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArchiveBudgetTracker {
    policy: ArchiveSecurityPolicy,
    entries_seen: usize,
    total_bytes: u64,
}

impl ArchiveBudgetTracker {
    pub fn new(policy: ArchiveSecurityPolicy) -> Self {
        Self {
            policy,
            entries_seen: 0,
            total_bytes: 0,
        }
    }

    pub fn policy(&self) -> &ArchiveSecurityPolicy {
        &self.policy
    }

    pub fn record_entry_header(
        &mut self,
        declared_uncompressed: u64,
        declared_compressed: Option<u64>,
    ) -> EngineResult<()> {
        self.entries_seen += 1;
        if self.entries_seen > self.policy.max_entries {
            return Err(EngineError::SecurityLimit(format!(
                "archive entry count {} exceeds max_entries {}",
                self.entries_seen, self.policy.max_entries
            )));
        }
        if declared_uncompressed > self.policy.max_entry_bytes {
            return Err(EngineError::SecurityLimit(format!(
                "archive entry size {} exceeds max_entry_bytes {}",
                declared_uncompressed, self.policy.max_entry_bytes
            )));
        }
        if self.total_bytes.saturating_add(declared_uncompressed) > self.policy.max_total_bytes {
            return Err(EngineError::SecurityLimit(format!(
                "archive cumulative size {} exceeds max_total_bytes {}",
                self.total_bytes.saturating_add(declared_uncompressed),
                self.policy.max_total_bytes
            )));
        }
        if let Some(comp) = declared_compressed
            && declared_uncompressed > 1_048_576
        {
            let ratio = declared_uncompressed as f64 / comp.max(1) as f64;
            if ratio > self.policy.max_ratio {
                return Err(EngineError::SecurityLimit(format!(
                    "suspicious compression ratio {ratio:.1}x exceeds max_ratio {:.1}x",
                    self.policy.max_ratio
                )));
            }
        }
        Ok(())
    }

    pub fn read_bounded<R: Read>(
        &mut self,
        mut reader: R,
        compressed_hint: Option<u64>,
    ) -> EngineResult<Vec<u8>> {
        let remaining_total = self
            .policy
            .max_total_bytes
            .saturating_sub(self.total_bytes);
        let ratio_cap = compressed_hint.map_or(self.policy.max_entry_bytes, |comp| {
            let scaled = (comp.max(16_384) as f64 * self.policy.max_ratio).ceil() as u64;
            scaled.max(1_048_576)
        });
        let limit = self
            .policy
            .max_entry_bytes
            .min(remaining_total)
            .min(ratio_cap);

        let mut out = Vec::new();
        let mut buf = [0u8; 32 * 1024];
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            let next_len = (out.len() as u64).saturating_add(n as u64);
            if next_len > limit {
                return Err(EngineError::SecurityLimit(format!(
                    "streamed archive entry exceeded security limit ({limit} bytes)"
                )));
            }
            out.extend_from_slice(&buf[..n]);
        }
        self.total_bytes = self.total_bytes.saturating_add(out.len() as u64);
        Ok(out)
    }

    pub fn add_accounted_bytes(&mut self, bytes: u64) -> EngineResult<()> {
        let next = self.total_bytes.saturating_add(bytes);
        if next > self.policy.max_total_bytes {
            return Err(EngineError::SecurityLimit(format!(
                "archive total decoded size {next} exceeds max_total_bytes {}",
                self.policy.max_total_bytes
            )));
        }
        self.total_bytes = next;
        Ok(())
    }
}

pub fn sanitize_archive_path(raw: &str) -> EngineResult<String> {
    if raw.is_empty() || raw.contains('\0') {
        return Err(EngineError::SecurityLimit(
            "archive entry contains empty path or NUL byte".to_string(),
        ));
    }
    if raw.len() > 4096 {
        return Err(EngineError::SecurityLimit(
            "archive entry path exceeds 4096 bytes".to_string(),
        ));
    }

    let slash_norm = raw.replace('\\', "/");
    let trimmed = slash_norm.trim_start_matches('/');
    if trimmed.is_empty() {
        return Err(EngineError::SecurityLimit(format!(
            "archive path resolves to root: {raw}"
        )));
    }

    let bytes = trimmed.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        return Err(EngineError::SecurityLimit(format!(
            "drive-prefixed archive path rejected: {raw}"
        )));
    }

    let mut clean_parts: Vec<&str> = Vec::new();
    for comp in Path::new(trimmed).components() {
        match comp {
            Component::Normal(s) => {
                let st = s.to_str().ok_or_else(|| {
                    EngineError::SecurityLimit("non-UTF-8 archive path component".to_string())
                })?;
                if st == ".." || st.contains('\0') {
                    return Err(EngineError::SecurityLimit(format!(
                        "path traversal rejected: {raw}"
                    )));
                }
                clean_parts.push(st);
            }
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(EngineError::SecurityLimit(format!(
                    "unsafe path traversal rejected: {raw}"
                )));
            }
        }
    }

    if clean_parts.is_empty() {
        return Err(EngineError::SecurityLimit(format!(
            "archive path resolves empty: {raw}"
        )));
    }

    Ok(clean_parts.join("/"))
}

pub fn check_non_overlapping_ranges(ranges: &mut [(u64, u64)]) -> EngineResult<()> {
    if ranges.len() < 2 {
        return Ok(());
    }
    ranges.sort_unstable_by_key(|r| (r.0, r.1));
    for win in ranges.windows(2) {
        let prev = win[0];
        let next = win[1];
        if prev.1 > next.0 && next.1 > next.0 {
            return Err(EngineError::SecurityLimit(format!(
                "overlapping archive payload offsets ({}..{} overlaps {}..{})",
                prev.0, prev.1, next.0, next.1
            )));
        }
    }
    Ok(())
}
