use crate::discover::is_image_candidate;
use rustix::fs::inotify::{self, CreateFlags, WatchFlags};
use rustix::io::Errno;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::mem::MaybeUninit;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub struct DirectoryWatcher {
    inotify_fd: Option<rustix::fd::OwnedFd>,
    watched_dirs: HashMap<i32, PathBuf>,
    root_dirs: Vec<PathBuf>,
    recursive: bool,
    seen_files: HashMap<PathBuf, (u64, u64)>,
    ignored_outputs: HashSet<PathBuf>,
}

impl DirectoryWatcher {
    pub fn new(dirs: &[PathBuf], recursive: bool) -> Result<Self, String> {
        let mut watcher = Self {
            inotify_fd: inotify::init(CreateFlags::NONBLOCK | CreateFlags::CLOEXEC).ok(),
            watched_dirs: HashMap::new(),
            root_dirs: dirs.to_vec(),
            recursive,
            seen_files: HashMap::new(),
            ignored_outputs: HashSet::new(),
        };

        for d in dirs {
            watcher.register_dir_tree(d);
            watcher.seed_existing(d);
        }
        Ok(watcher)
    }

    pub fn ignore_output(&mut self, path: &Path) {
        if let Ok(c) = fs::canonicalize(path) {
            self.ignored_outputs.insert(c);
        }
        self.ignored_outputs.insert(path.to_path_buf());
    }

    fn register_dir_tree(&mut self, dir: &Path) {
        let Some(ref fd) = self.inotify_fd else {
            return;
        };
        let flags = WatchFlags::CLOSE_WRITE | WatchFlags::MOVED_TO | WatchFlags::CREATE;
        if let Ok(wd) = inotify::add_watch(fd, dir, flags) {
            self.watched_dirs.insert(wd, dir.to_path_buf());
        }
        if self.recursive
            && let Ok(rd) = fs::read_dir(dir)
        {
            for ent in rd.flatten() {
                let p = ent.path();
                if p.is_dir() {
                    self.register_dir_tree(&p);
                }
            }
        }
    }

    fn seed_existing(&mut self, dir: &Path) {
        let Ok(rd) = fs::read_dir(dir) else {
            return;
        };
        for ent in rd.flatten() {
            let p = ent.path();
            if p.is_dir() {
                if self.recursive {
                    self.seed_existing(&p);
                }
            } else if p.is_file()
                && let Ok(m) = fs::metadata(&p)
            {
                let stamp = file_stamp(&m);
                self.seen_files.insert(p, stamp);
            }
        }
    }

    pub fn poll_changes(&mut self, settle: Duration) -> Vec<PathBuf> {
        thread::sleep(settle);
        let mut changed = Vec::new();
        let mut candidate_dirs: HashSet<PathBuf> = HashSet::new();

        let mut new_dirs = Vec::new();
        if let Some(ref fd) = self.inotify_fd {
            let mut buf = [MaybeUninit::uninit(); 1024];
            let mut reader = inotify::Reader::new(fd, &mut buf);
            loop {
                match reader.next() {
                    Ok(ev) => {
                        if let Some(base) = self.watched_dirs.get(&ev.wd()).cloned() {
                            if let Some(name) = ev.file_name() {
                                let name_os = std::ffi::OsStr::new(name.to_str().unwrap_or(""));
                                let full = base.join(name_os);
                                if full.is_dir() && self.recursive {
                                    new_dirs.push(full);
                                }
                            }
                            candidate_dirs.insert(base);
                        }
                    }
                    Err(Errno::WOULDBLOCK) => break,
                    Err(_) => {
                        for d in &self.root_dirs {
                            candidate_dirs.insert(d.clone());
                        }
                        break;
                    }
                }
            }
        } else {
            for d in &self.root_dirs {
                candidate_dirs.insert(d.clone());
            }
        }
        for d in new_dirs {
            self.register_dir_tree(&d);
        }

        if candidate_dirs.is_empty() && self.inotify_fd.is_none() {
            for d in &self.root_dirs {
                candidate_dirs.insert(d.clone());
            }
        }

        let scan_roots: Vec<PathBuf> = if candidate_dirs.is_empty() {
            Vec::new()
        } else {
            candidate_dirs.into_iter().collect()
        };

        for root in scan_roots {
            self.collect_changed_in_dir(&root, &mut changed);
        }

        changed.sort();
        changed.dedup();
        changed
    }

    fn collect_changed_in_dir(&mut self, dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = fs::read_dir(dir) else {
            return;
        };
        for ent in rd.flatten() {
            let p = ent.path();
            let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or_default();
            if fname.starts_with('.') || fname.ends_with(".tmp") || fname.ends_with(".bak") {
                continue;
            }
            if p.is_dir() {
                if self.recursive {
                    self.collect_changed_in_dir(&p, out);
                }
                continue;
            }
            if !p.is_file() {
                continue;
            }
            if self.ignored_outputs.contains(&p)
                || fs::canonicalize(&p)
                    .ok()
                    .is_some_and(|c| self.ignored_outputs.contains(&c))
            {
                continue;
            }
            let Ok(meta) = fs::metadata(&p) else {
                continue;
            };
            let stamp = file_stamp(&meta);
            let prev = self.seen_files.get(&p).copied();
            if prev != Some(stamp) && is_image_candidate(&p) {
                self.seen_files.insert(p.clone(), stamp);
                out.push(p);
            }
        }
    }
}

fn file_stamp(meta: &fs::Metadata) -> (u64, u64) {
    let mtime = meta
        .modified()
        .unwrap_or(SystemTime::UNIX_EPOCH)
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    (meta.size(), mtime)
}
