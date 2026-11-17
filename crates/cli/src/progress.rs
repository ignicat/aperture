use json_fmt::escape_json_str;
use std::io::{self, IsTerminal, Write};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct ItemSummary {
    pub input: String,
    pub output: Option<String>,
    pub src_width: u32,
    pub src_height: u32,
    pub dst_width: u32,
    pub dst_height: u32,
    pub in_bytes: u64,
    pub out_bytes: u64,
    pub status: String,
    pub message: Option<String>,
}

pub struct ProgressTracker {
    total: usize,
    done: usize,
    processed: usize,
    skipped: usize,
    failed: usize,
    pixels_processed: u64,
    bytes_in: u64,
    bytes_out: u64,
    start: Instant,
    is_tty: bool,
    quiet: bool,
}

impl ProgressTracker {
    pub fn new(total: usize, quiet: bool) -> Self {
        Self {
            total,
            done: 0,
            processed: 0,
            skipped: 0,
            failed: 0,
            pixels_processed: 0,
            bytes_in: 0,
            bytes_out: 0,
            start: Instant::now(),
            is_tty: io::stderr().is_terminal(),
            quiet,
        }
    }

    pub fn record(&mut self, item: &ItemSummary) {
        self.done += 1;
        self.bytes_in += item.in_bytes;
        self.bytes_out += item.out_bytes;
        match item.status.as_str() {
            "processed" => {
                self.processed += 1;
                self.pixels_processed += (item.dst_width as u64) * (item.dst_height as u64);
            }
            "skipped" => self.skipped += 1,
            _ => self.failed += 1,
        }
        if !self.quiet && self.is_tty {
            let elapsed = self.start.elapsed().as_secs_f64().max(1e-4);
            let mp_s = (self.pixels_processed as f64 / 1_000_000.0) / elapsed;
            let mb_s = (self.bytes_in as f64 / (1024.0 * 1024.0)) / elapsed;
            let pct = (self.done * 100) / self.total.max(1);
            let bar_w = 24usize;
            let filled = (pct * bar_w) / 100;
            let mut bar = String::with_capacity(bar_w);
            for i in 0..bar_w {
                bar.push(if i < filled { '#' } else { '-' });
            }
            let mut err = io::stderr().lock();
            let _ = write!(
                err,
                "\r[{bar}] {done}/{total} ({pct:3}%) | {mp_s:.1} MP/s | {mb_s:.1} MB/s",
                done = self.done,
                total = self.total.max(1),
            );
            let _ = err.flush();
        }
    }

    pub fn finish_bar(&self) {
        if !self.quiet && self.is_tty && self.done > 0 {
            let mut err = io::stderr().lock();
            let _ = writeln!(err);
        }
    }

    pub fn format_summary_text(&self) -> String {
        let elapsed = self.start.elapsed().as_secs_f64();
        let ratio = if self.bytes_in > 0 {
            self.bytes_out as f64 / self.bytes_in as f64 * 100.0
        } else {
            100.0
        };
        format!(
            "done: {} processed, {} skipped, {} failed in {:.3}s | in={}B -> out={}B ({:.1}%)",
            self.processed,
            self.skipped,
            self.failed,
            elapsed,
            self.bytes_in,
            self.bytes_out,
            ratio,
        )
    }

    pub fn format_summary_json(&self, items: &[ItemSummary]) -> String {
        let elapsed = self.start.elapsed().as_secs_f64();
        let mp_s = (self.pixels_processed as f64 / 1_000_000.0) / elapsed.max(1e-6);
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str(&format!("  \"processed\": {},\n", self.processed));
        out.push_str(&format!("  \"skipped\": {},\n", self.skipped));
        out.push_str(&format!("  \"failed\": {},\n", self.failed));
        out.push_str(&format!("  \"bytes_in\": {},\n", self.bytes_in));
        out.push_str(&format!("  \"bytes_out\": {},\n", self.bytes_out));
        out.push_str(&format!("  \"elapsed_secs\": {:.4},\n", elapsed));
        out.push_str(&format!("  \"megapixels_per_sec\": {:.3},\n", mp_s));
        out.push_str("  \"items\": [\n");
        for (i, item) in items.iter().enumerate() {
            let comma = if i + 1 < items.len() { "," } else { "" };
            let out_json = item
                .output
                .as_ref()
                .map_or_else(|| "null".to_string(), |s| format!("\"{}\"", escape_json_str(s)));
            let msg_json = item
                .message
                .as_ref()
                .map_or_else(|| "null".to_string(), |s| format!("\"{}\"", escape_json_str(s)));
            out.push_str(&format!(
                "    {{\"input\": \"{}\", \"output\": {out_json}, \"src\": \"{}x{}\", \"dst\": \"{}x{}\", \"in_bytes\": {}, \"out_bytes\": {}, \"status\": \"{}\", \"message\": {msg_json}}}{comma}\n",
                escape_json_str(&item.input),
                item.src_width,
                item.src_height,
                item.dst_width,
                item.dst_height,
                item.in_bytes,
                item.out_bytes,
                escape_json_str(&item.status),
            ));
        }
        out.push_str("  ]\n}");
        out
    }
}

mod json_fmt {
    pub fn escape_json_str(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                other => out.push(other),
            }
        }
        out
    }
}
