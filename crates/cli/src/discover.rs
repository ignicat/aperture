use engine::archive::{ArchiveFormat, detect_archive_format};
use engine::codec::{ImageFormat, detect_format};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub enum DiscoveredInput {
    Image { path: PathBuf, root: PathBuf },
    Archive { path: PathBuf, root: PathBuf },
    Stdin,
}

#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    pub recursive: bool,
    pub include_globs: Vec<String>,
    pub exclude_globs: Vec<String>,
}

pub fn discover_inputs(
    raw_inputs: &[String],
    manifest: Option<&str>,
    cfg: &DiscoveryConfig,
) -> Result<Vec<DiscoveredInput>, String> {
    let mut expanded_paths: Vec<String> = Vec::new();
    for item in raw_inputs {
        if let Some(mpath) = item.strip_prefix('@') {
            load_manifest(Path::new(mpath), &mut expanded_paths)?;
        } else {
            expanded_paths.push(item.clone());
        }
    }
    if let Some(mpath) = manifest {
        load_manifest(Path::new(mpath), &mut expanded_paths)?;
    }

    let mut out = Vec::new();
    let mut visited_dirs: HashSet<(u64, u64)> = HashSet::new();
    let mut seen_files: HashSet<(u64, u64)> = HashSet::new();

    for spec in expanded_paths {
        if spec == "-" {
            out.push(DiscoveredInput::Stdin);
            continue;
        }
        let path = PathBuf::from(&spec);
        if path.exists() {
            collect_path(&path, &path, cfg, &mut visited_dirs, &mut seen_files, &mut out)?;
        } else if contains_glob_meta(&spec) {
            let matches = expand_glob_spec(&spec)?;
            if matches.is_empty() {
                return Err(format!("glob pattern matched no files: {spec}"));
            }
            for m in matches {
                let root = m.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
                collect_path(&m, &root, cfg, &mut visited_dirs, &mut seen_files, &mut out)?;
            }
        } else {
            return Err(format!("input path does not exist: {spec}"));
        }
    }

    Ok(out)
}

fn load_manifest(path: &Path, out: &mut Vec<String>) -> Result<(), String> {
    let raw = fs::read(path).map_err(|e| format!("failed to read manifest {}: {e}", path.display()))?;
    if raw.contains(&0u8) {
        for part in raw.split(|&b| b == 0u8) {
            if let Ok(s) = std::str::from_utf8(part) {
                let t = s.trim();
                if !t.is_empty() {
                    out.push(t.to_string());
                }
            }
        }
    } else {
        let text = String::from_utf8_lossy(&raw);
        for line in text.lines() {
            let t = line.trim();
            if !t.is_empty() {
                out.push(t.to_string());
            }
        }
    }
    Ok(())
}

fn collect_path(
    path: &Path,
    root: &Path,
    cfg: &DiscoveryConfig,
    visited_dirs: &mut HashSet<(u64, u64)>,
    seen_files: &mut HashSet<(u64, u64)>,
    out: &mut Vec<DiscoveredInput>,
) -> Result<(), String> {
    let meta = fs::metadata(path)
        .map_err(|e| format!("cannot stat {}: {e}", path.display()))?;
    let key = (meta.dev(), meta.ino());

    if meta.is_dir() {
        if !visited_dirs.insert(key) {
            return Ok(());
        }
        let entries = fs::read_dir(path)
            .map_err(|e| format!("cannot read dir {}: {e}", path.display()))?;
        let mut children: Vec<PathBuf> = Vec::new();
        for entry_res in entries {
            let entry = entry_res.map_err(|e| format!("dir iteration error: {e}"))?;
            children.push(entry.path());
        }
        children.sort();

        for child in children {
            let fname = child
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            if fname.starts_with('.') && fname.ends_with(".tmp") {
                continue;
            }
            let cmeta = match fs::metadata(&child) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if cmeta.is_dir() {
                if cfg.recursive {
                    collect_path(&child, root, cfg, visited_dirs, seen_files, out)?;
                }
            } else if cmeta.is_file() {
                maybe_push_file(&child, root, cmeta.dev(), cmeta.ino(), cfg, seen_files, out, false);
            }
        }
        return Ok(());
    }

    if meta.is_file() {
        maybe_push_file(
            path,
            root.parent().unwrap_or(root),
            meta.dev(),
            meta.ino(),
            cfg,
            seen_files,
            out,
            true,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn maybe_push_file(
    path: &Path,
    root: &Path,
    dev: u64,
    ino: u64,
    cfg: &DiscoveryConfig,
    seen_files: &mut HashSet<(u64, u64)>,
    out: &mut Vec<DiscoveredInput>,
    explicit_file: bool,
) {
    let path_str = path.to_string_lossy();
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(&path_str);

    if !cfg.exclude_globs.is_empty()
        && cfg
            .exclude_globs
            .iter()
            .any(|g| glob_match(g, &path_str) || glob_match(g, file_name))
    {
        return;
    }
    if !cfg.include_globs.is_empty()
        && !cfg
            .include_globs
            .iter()
            .any(|g| glob_match(g, &path_str) || glob_match(g, file_name))
    {
        return;
    }

    if !seen_files.insert((dev, ino)) {
        return;
    }

    if is_archive_candidate(path) {
        out.push(DiscoveredInput::Archive {
            path: path.to_path_buf(),
            root: root.to_path_buf(),
        });
        return;
    }

    if explicit_file || is_image_candidate(path) {
        out.push(DiscoveredInput::Image {
            path: path.to_path_buf(),
            root: root.to_path_buf(),
        });
    }
}

pub fn is_archive_candidate(path: &Path) -> bool {
    let s = path.to_string_lossy();
    if ArchiveFormat::from_extension(&s).is_some() {
        return true;
    }
    let mut buf = [0u8; 265];
    if let Ok(mut f) = File::open(path)
        && let Ok(n) = f.read(&mut buf)
    {
        return detect_archive_format(&buf[..n], Some(&s)).is_some();
    }
    false
}

pub fn is_image_candidate(path: &Path) -> bool {
    if let Some(ext) = path.extension().and_then(|s| s.to_str())
        && ImageFormat::from_extension(ext).is_some()
    {
        return true;
    }
    let mut buf = [0u8; 512];
    if let Ok(mut f) = File::open(path)
        && let Ok(n) = f.read(&mut buf)
    {
        return detect_format(&buf[..n]).is_some();
    }
    false
}

fn contains_glob_meta(s: &str) -> bool {
    s.contains('*') || s.contains('?') || s.contains('[')
}

fn expand_glob_spec(pattern: &str) -> Result<Vec<PathBuf>, String> {
    let p = Path::new(pattern);
    let mut root_parts = Vec::new();
    let mut pat_parts = Vec::new();
    let mut hit_meta = false;
    for comp in p.components() {
        let s = comp.as_os_str().to_string_lossy();
        if !hit_meta && !contains_glob_meta(&s) {
            root_parts.push(comp.as_os_str().to_os_string());
        } else {
            hit_meta = true;
            pat_parts.push(s.into_owned());
        }
    }
    let base: PathBuf = if root_parts.is_empty() {
        PathBuf::from(".")
    } else {
        let mut b = PathBuf::new();
        for part in root_parts {
            b.push(part);
        }
        b
    };

    let mut matches = Vec::new();
    if base.is_dir() {
        let recursive = pat_parts.iter().any(|part| part.contains("**"));
        let mut stack = vec![base];
        while let Some(dir) = stack.pop() {
            let Ok(rd) = fs::read_dir(&dir) else {
                continue;
            };
            for ent in rd.flatten() {
                let ep = ent.path();
                if ep.is_dir() && recursive {
                    stack.push(ep);
                } else if ep.is_file() {
                    let s = ep.to_string_lossy();
                    let s_clean = s.strip_prefix("./").unwrap_or(&s);
                    let fname_hit = ep
                        .file_name()
                        .is_some_and(|n| glob_match(pattern, &n.to_string_lossy()));
                    if glob_match(pattern, s_clean) || glob_match(pattern, &s) || fname_hit {
                        matches.push(ep);
                    }
                }
            }
        }
    }
    matches.sort();
    Ok(matches)
}

pub fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    glob_match_chars(&p, &t)
}

fn glob_match_chars(p: &[char], t: &[char]) -> bool {
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star_p, mut star_t) = (None, 0usize);

    while ti < t.len() {
        if pi < p.len() && p[pi] == '*' {
            while pi < p.len() && p[pi] == '*' {
                pi += 1;
            }
            star_p = Some(pi);
            star_t = ti;
        } else if pi < p.len() && (p[pi] == '?' || p[pi].eq_ignore_ascii_case(&t[ti])) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '[' {
            if let Some((matched, next_pi)) = match_char_class(&p[pi..], t[ti])
                && matched
            {
                pi += next_pi;
                ti += 1;
                continue;
            }
            if let Some(sp) = star_p {
                star_t += 1;
                ti = star_t;
                pi = sp;
            } else {
                return false;
            }
        } else if let Some(sp) = star_p {
            star_t += 1;
            ti = star_t;
            pi = sp;
        } else {
            return false;
        }
    }

    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

fn match_char_class(p: &[char], ch: char) -> Option<(bool, usize)> {
    let end = p.iter().position(|&c| c == ']')?;
    if end < 2 {
        return None;
    }
    let body = &p[1..end];
    let (neg, slice) = if body[0] == '!' || body[0] == '^' {
        (true, &body[1..])
    } else {
        (false, body)
    };
    let mut found = false;
    let mut i = 0usize;
    while i < slice.len() {
        if i + 2 < slice.len() && slice[i + 1] == '-' {
            if (slice[i]..=slice[i + 2]).contains(&ch) {
                found = true;
            }
            i += 3;
        } else {
            if slice[i].eq_ignore_ascii_case(&ch) {
                found = true;
            }
            i += 1;
        }
    }
    Some((if neg { !found } else { found }, end + 1))
}
