use crate::error::{EngineError, EngineResult};
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct MemoryBudget {
    limit: usize,
    used: AtomicUsize,
}

impl MemoryBudget {
    pub fn new(limit_bytes: usize) -> Self {
        Self {
            limit: limit_bytes,
            used: AtomicUsize::new(0),
        }
    }

    pub fn unlimited() -> Self {
        Self::new(0)
    }

    pub fn from_system_fraction(num: usize, den: usize) -> Self {
        if let Some(avail) = read_mem_available_bytes() {
            let scaled = avail.saturating_mul(num) / den.max(1);
            let floor = 256 * 1024 * 1024;
            return Self::new(scaled.max(floor));
        }
        Self::new(2usize * 1024 * 1024 * 1024)
    }

    pub fn limit_bytes(&self) -> usize {
        self.limit
    }

    pub fn used_bytes(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }

    pub fn try_reserve(&self, bytes: usize) -> EngineResult<MemoryLease<'_>> {
        if self.limit == 0 || bytes == 0 {
            if bytes > 0 {
                self.used.fetch_add(bytes, Ordering::AcqRel);
            }
            return Ok(MemoryLease {
                budget: self,
                bytes,
            });
        }
        let mut current = self.used.load(Ordering::Acquire);
        loop {
            let Some(next) = current.checked_add(bytes) else {
                return Err(EngineError::MemoryBudgetExceeded {
                    requested: bytes,
                    available: self.limit.saturating_sub(current),
                });
            };
            if next > self.limit {
                return Err(EngineError::MemoryBudgetExceeded {
                    requested: bytes,
                    available: self.limit.saturating_sub(current),
                });
            }
            match self.used.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(MemoryLease {
                        budget: self,
                        bytes,
                    });
                }
                Err(actual) => current = actual,
            }
        }
    }
}

#[derive(Debug)]
pub struct MemoryLease<'a> {
    budget: &'a MemoryBudget,
    bytes: usize,
}

impl Drop for MemoryLease<'_> {
    fn drop(&mut self) {
        if self.bytes > 0 {
            self.budget.used.fetch_sub(self.bytes, Ordering::Release);
        }
    }
}

fn read_mem_available_bytes() -> Option<usize> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb_str = rest.split_whitespace().next()?;
            let kb: usize = kb_str.parse().ok()?;
            return kb.checked_mul(1024);
        }
    }
    None
}

#[derive(Debug, Default)]
pub struct WorkerArena {
    pub scratch_u8: Vec<u8>,
    pub scratch_u16: Vec<u16>,
    pub scratch_f32_a: Vec<f32>,
    pub scratch_f32_b: Vec<f32>,
    pub scratch_f32_c: Vec<f32>,
    pub scratch_f32_d: Vec<f32>,
    max_retain_bytes: usize,
}

impl WorkerArena {
    pub fn new(max_retain_bytes: usize) -> Self {
        Self {
            scratch_u8: Vec::new(),
            scratch_u16: Vec::new(),
            scratch_f32_a: Vec::new(),
            scratch_f32_b: Vec::new(),
            scratch_f32_c: Vec::new(),
            scratch_f32_d: Vec::new(),
            max_retain_bytes,
        }
    }

    pub fn ensure_u8(&mut self, len: usize) -> &mut [u8] {
        if self.scratch_u8.len() < len {
            self.scratch_u8.resize(len, 0);
        }
        &mut self.scratch_u8[..len]
    }

    pub fn ensure_u16(&mut self, len: usize) -> &mut [u16] {
        if self.scratch_u16.len() < len {
            self.scratch_u16.resize(len, 0);
        }
        &mut self.scratch_u16[..len]
    }

    pub fn ensure_f32_planes(
        &mut self,
        len: usize,
    ) -> (&mut [f32], &mut [f32], &mut [f32], &mut [f32]) {
        if self.scratch_f32_a.len() < len {
            self.scratch_f32_a.resize(len, 0.0);
        }
        if self.scratch_f32_b.len() < len {
            self.scratch_f32_b.resize(len, 0.0);
        }
        if self.scratch_f32_c.len() < len {
            self.scratch_f32_c.resize(len, 0.0);
        }
        if self.scratch_f32_d.len() < len {
            self.scratch_f32_d.resize(len, 0.0);
        }
        (
            &mut self.scratch_f32_a[..len],
            &mut self.scratch_f32_b[..len],
            &mut self.scratch_f32_c[..len],
            &mut self.scratch_f32_d[..len],
        )
    }

    pub fn shrink_excess(&mut self) {
        if self.max_retain_bytes == 0 {
            return;
        }
        if self.scratch_u8.capacity() > self.max_retain_bytes {
            self.scratch_u8 = Vec::new();
        }
        let f32_cap_limit = self.max_retain_bytes / std::mem::size_of::<f32>();
        if self.scratch_f32_a.capacity() > f32_cap_limit {
            self.scratch_f32_a = Vec::new();
        }
        if self.scratch_f32_b.capacity() > f32_cap_limit {
            self.scratch_f32_b = Vec::new();
        }
        if self.scratch_f32_c.capacity() > f32_cap_limit {
            self.scratch_f32_c = Vec::new();
        }
        if self.scratch_f32_d.capacity() > f32_cap_limit {
            self.scratch_f32_d = Vec::new();
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledAxisPlan {
    pub src_len: u32,
    pub dst_len: u32,
    pub src_starts: Vec<u32>,
    pub tap_counts: Vec<u16>,
    pub weight_offsets: Vec<u32>,
    pub weights: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResamplePlanKey {
    pub src_w: u32,
    pub src_h: u32,
    pub dst_w: u32,
    pub dst_h: u32,
    pub kernel_id: u32,
    pub quantized_param: i32,
}

#[derive(Debug, Clone)]
pub struct CompiledResamplePlan {
    pub horiz: CompiledAxisPlan,
    pub vert: CompiledAxisPlan,
}

#[derive(Debug)]
pub struct PlanCache {
    capacity: usize,
    entries: Mutex<Vec<(ResamplePlanKey, Arc<CompiledResamplePlan>)>>,
}

impl PlanCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: Mutex::new(Vec::new()),
        }
    }

    pub fn get_or_build<F>(&self, key: ResamplePlanKey, builder: F) -> Arc<CompiledResamplePlan>
    where
        F: FnOnce() -> CompiledResamplePlan,
    {
        if let Ok(mut guard) = self.entries.lock()
            && let Some(pos) = guard.iter().position(|(k, _)| *k == key)
        {
            let item = guard.remove(pos);
            let cloned = Arc::clone(&item.1);
            guard.push(item);
            return cloned;
        }
        let built = Arc::new(builder());
        if let Ok(mut guard) = self.entries.lock() {
            if let Some(pos) = guard.iter().position(|(k, _)| *k == key) {
                let item = guard.remove(pos);
                let cloned = Arc::clone(&item.1);
                guard.push(item);
                return cloned;
            }
            if guard.len() >= self.capacity {
                guard.remove(0);
            }
            guard.push((key, Arc::clone(&built)));
        }
        built
    }

    pub fn len(&self) -> usize {
        self.entries.lock().map(|g| g.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
