use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

#[derive(Debug)]
pub struct ResumeJournal {
    path: PathBuf,
    completed: HashSet<String>,
    writer: Option<BufWriter<File>>,
}

impl ResumeJournal {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut completed = HashSet::new();
        if path.exists() {
            let text = fs::read_to_string(path)
                .map_err(|e| format!("failed to read resume journal {}: {e}", path.display()))?;
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("DONE\t") {
                    completed.insert(rest.to_string());
                }
            }
        } else if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create journal directory: {e}"))?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("failed to open resume journal {}: {e}", path.display()))?;

        Ok(Self {
            path: path.to_path_buf(),
            completed,
            writer: Some(BufWriter::new(file)),
        })
    }

    pub fn fingerprint(path: &Path) -> String {
        let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let (len, mtime_sec) = fs::metadata(path).map_or((0u64, 0u64), |m| {
            let sec = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());
            (m.len(), sec)
        });
        format!("{}\t{}\t{}", canonical.to_string_lossy(), len, mtime_sec)
    }

    pub fn is_completed(&self, key: &str) -> bool {
        self.completed.contains(key)
    }

    pub fn mark_completed(&mut self, key: &str) {
        if self.completed.insert(key.to_string())
            && let Some(ref mut w) = self.writer
        {
            let _ = writeln!(w, "DONE\t{key}");
            let _ = w.flush();
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}
