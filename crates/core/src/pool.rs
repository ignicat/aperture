use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

pub fn effective_cpu_count() -> usize {
    let hw = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let cgroup_cap = read_cgroup_v2_cpu_cap().or_else(read_cgroup_v1_cpu_cap);
    match cgroup_cap {
        Some(cap) => hw.min(cap).max(1),
        None => hw.max(1),
    }
}

fn read_cgroup_v2_cpu_cap() -> Option<usize> {
    let content = fs::read_to_string("/sys/fs/cgroup/cpu.max").ok()?;
    let mut parts = content.split_whitespace();
    let quota_str = parts.next()?;
    let period_str = parts.next()?;
    if quota_str == "max" {
        return None;
    }
    let quota: usize = quota_str.parse().ok()?;
    let period: usize = period_str.parse().ok()?;
    if period == 0 {
        return None;
    }
    Some(quota.div_ceil(period).max(1))
}

fn read_cgroup_v1_cpu_cap() -> Option<usize> {
    let quota_raw = fs::read_to_string("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").ok()?;
    let period_raw = fs::read_to_string("/sys/fs/cgroup/cpu/cpu.cfs_period_us").ok()?;
    let quota: isize = quota_raw.trim().parse().ok()?;
    let period: isize = period_raw.trim().parse().ok()?;
    if quota <= 0 || period <= 0 {
        return None;
    }
    Some((quota as usize).div_ceil(period as usize).max(1))
}

#[derive(Debug, Clone, Copy)]
pub struct ThreadPool {
    threads: usize,
}

impl ThreadPool {
    pub fn new(threads: usize) -> Self {
        let count = if threads == 0 {
            effective_cpu_count()
        } else {
            threads
        };
        Self { threads: count }
    }

    pub fn thread_count(&self) -> usize {
        self.threads
    }

    pub fn scope_indexed<F>(&self, total: usize, work: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        if total == 0 {
            return;
        }
        let workers = self.threads.min(total);
        if workers <= 1 {
            for idx in 0..total {
                work(0, idx);
            }
            return;
        }
        let next = AtomicUsize::new(0);
        thread::scope(|s| {
            for worker_id in 0..workers {
                let next_ref = &next;
                let work_ref = &work;
                s.spawn(move || loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= total {
                        break;
                    }
                    work_ref(worker_id, idx);
                });
            }
        });
    }

    pub fn parallel_map<T, R, F>(&self, items: &[T], f: F) -> Vec<R>
    where
        T: Sync,
        R: Send,
        F: Fn(usize, &T) -> R + Sync,
    {
        let total = items.len();
        if total == 0 {
            return Vec::new();
        }
        let workers = self.threads.min(total);
        if workers <= 1 {
            return items.iter().enumerate().map(|(i, item)| f(i, item)).collect();
        }
        let next = AtomicUsize::new(0);
        let collected = thread::scope(|s| {
            let mut handles = Vec::with_capacity(workers);
            for _ in 0..workers {
                let next_ref = &next;
                let f_ref = &f;
                handles.push(s.spawn(move || {
                    let mut local = Vec::new();
                    loop {
                        let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                        if idx >= total {
                            break;
                        }
                        local.push((idx, f_ref(idx, &items[idx])));
                    }
                    local
                }));
            }
            let mut slots: Vec<Option<R>> = (0..total).map(|_| None).collect();
            for handle in handles {
                if let Ok(chunk) = handle.join() {
                    for (idx, val) in chunk {
                        slots[idx] = Some(val);
                    }
                }
            }
            slots
        });
        collected.into_iter().flatten().collect()
    }

    pub fn parallel_for_rows<F>(&self, height: usize, chunk_rows: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        if height == 0 {
            return;
        }
        let step = chunk_rows.max(1);
        let chunks = height.div_ceil(step);
        self.scope_indexed(chunks, |_, chunk_idx| {
            let start = chunk_idx * step;
            let end = (start + step).min(height);
            f(start, end);
        });
    }

    pub fn parallel_row_stripes_mut<T, F>(&self, buf: &mut [T], row_len: usize, f: F)
    where
        T: Send,
        F: Fn(usize, &mut [T]) + Sync,
    {
        if row_len == 0 || buf.is_empty() {
            return;
        }
        let total_rows = buf.len() / row_len;
        if total_rows == 0 {
            return;
        }
        let workers = self.threads.min(total_rows).max(1);
        if workers <= 1 {
            f(0, buf);
            return;
        }
        let rows_per_worker = total_rows.div_ceil(workers);
        let chunk_elems = rows_per_worker * row_len;
        thread::scope(|s| {
            let f_ref = &f;
            for (idx, stripe) in buf.chunks_mut(chunk_elems).enumerate() {
                let y0 = idx * rows_per_worker;
                s.spawn(move || f_ref(y0, stripe));
            }
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn parallel_row_stripes_4_mut<F>(
        &self,
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
        a: &mut [f32],
        row_len: usize,
        f: F,
    ) where
        F: Fn(usize, &mut [f32], &mut [f32], &mut [f32], Option<&mut [f32]>) + Sync,
    {
        if row_len == 0 || r.is_empty() {
            return;
        }
        let total_rows = r.len() / row_len;
        if total_rows == 0 {
            return;
        }
        let workers = self.threads.min(total_rows).max(1);
        if workers <= 1 {
            let opt_a = if a.is_empty() { None } else { Some(a) };
            f(0, r, g, b, opt_a);
            return;
        }
        let rows_per_worker = total_rows.div_ceil(workers);
        let chunk_elems = rows_per_worker * row_len;
        let has_alpha = !a.is_empty();
        thread::scope(|s| {
            let f_ref = &f;
            if has_alpha {
                let mut it_r = r.chunks_mut(chunk_elems);
                let mut it_g = g.chunks_mut(chunk_elems);
                let mut it_b = b.chunks_mut(chunk_elems);
                let mut it_a = a.chunks_mut(chunk_elems);
                let mut idx = 0usize;
                while let (Some(sr), Some(sg), Some(sb), Some(sa)) =
                    (it_r.next(), it_g.next(), it_b.next(), it_a.next())
                {
                    let y0 = idx * rows_per_worker;
                    idx += 1;
                    s.spawn(move || f_ref(y0, sr, sg, sb, Some(sa)));
                }
            } else {
                let mut it_r = r.chunks_mut(chunk_elems);
                let mut it_g = g.chunks_mut(chunk_elems);
                let mut it_b = b.chunks_mut(chunk_elems);
                let mut idx = 0usize;
                while let (Some(sr), Some(sg), Some(sb)) =
                    (it_r.next(), it_g.next(), it_b.next())
                {
                    let y0 = idx * rows_per_worker;
                    idx += 1;
                    s.spawn(move || f_ref(y0, sr, sg, sb, None));
                }
            }
        });
    }
}
