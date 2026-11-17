use crate::archive::security::{
    ArchiveBudgetTracker, check_non_overlapping_ranges, sanitize_archive_path,
};
use crate::archive::{ArchiveEntryInput, ArchiveEntryOutput};
use crate::error::{EngineError, EngineResult};
use flate2::Compression;
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use rawzip::{CompressionMethod, ReaderAt, ZipArchive, ZipArchiveWriter};
use std::fs::File;
use std::io::Write;

struct VerifiedMeta {
    clean_path: String,
    method: CompressionMethod,
    compressed_size: u64,
    uncompressed_size: u64,
    wayfinder: rawzip::ZipArchiveEntryWayfinder,
}

pub fn read_zip_from_slice(
    bytes: &[u8],
    tracker: &mut ArchiveBudgetTracker,
) -> EngineResult<Vec<ArchiveEntryInput>> {
    let slice_arch = ZipArchive::from_slice(bytes)
        .map_err(|e| EngineError::DecodeFailed(format!("invalid zip archive: {e}")))?;
    let archive = slice_arch.into_reader_archive();
    let mut dir_buf = vec![0u8; rawzip::MAX_CENTRAL_DIRECTORY_RECORD_SIZE];
    read_zip_archive(&archive, &mut dir_buf, tracker)
}

pub fn read_zip_from_file(
    file: File,
    tracker: &mut ArchiveBudgetTracker,
) -> EngineResult<Vec<ArchiveEntryInput>> {
    let mut dir_buf = vec![0u8; rawzip::MAX_CENTRAL_DIRECTORY_RECORD_SIZE];
    let archive = ZipArchive::from_file(file, &mut dir_buf)
        .map_err(|e| EngineError::DecodeFailed(format!("invalid zip file: {e}")))?;
    read_zip_archive(&archive, &mut dir_buf, tracker)
}

fn read_zip_archive<R: ReaderAt>(
    archive: &ZipArchive<R>,
    dir_buf: &mut [u8],
    tracker: &mut ArchiveBudgetTracker,
) -> EngineResult<Vec<ArchiveEntryInput>> {
    let mut metas = Vec::new();
    let mut ranges = Vec::new();

    {
        let mut entries_iter = archive.entries(dir_buf);
        while let Some(rec) = entries_iter
            .next_entry()
            .map_err(|e| EngineError::DecodeFailed(format!("zip central directory error: {e}")))?
        {
            let raw_central = std::str::from_utf8(rec.file_path().as_bytes())
                .map_err(|_| EngineError::SecurityLimit("non-UTF-8 zip path".to_string()))?;

            if rec.is_dir() || raw_central.ends_with('/') {
                continue;
            }
            if !tracker.policy().allow_symlinks && rec.mode().is_symlink() {
                return Err(EngineError::SecurityLimit(format!(
                    "symlink entry rejected in zip archive: {raw_central}"
                )));
            }
            if rec.flags().is_encrypted() {
                return Err(EngineError::SecurityLimit(format!(
                    "encrypted entry unsupported/rejected in zip archive: {raw_central}"
                )));
            }

            let clean_path = sanitize_archive_path(raw_central)?;
            let comp = rec.compressed_size_hint();
            let uncomp = rec.uncompressed_size_hint();
            tracker.record_entry_header(uncomp, Some(comp))?;

            let mut local_hdr = [0u8; 30];
            archive
                .get_ref()
                .read_exact_at(&mut local_hdr, rec.local_header_offset())
                .map_err(|e| EngineError::DecodeFailed(format!("zip local header error: {e}")))?;
            if local_hdr[0..4] != [0x50, 0x4B, 0x03, 0x04] {
                return Err(EngineError::SecurityLimit(format!(
                    "invalid local zip signature for {clean_path}"
                )));
            }
            let local_flags = u16::from_le_bytes([local_hdr[6], local_hdr[7]]);
            if local_flags & 1 != 0 {
                return Err(EngineError::SecurityLimit(format!(
                    "zip cloaking encrypted local flag mismatch for {clean_path}"
                )));
            }
            let local_method = u16::from_le_bytes([local_hdr[8], local_hdr[9]]);
            if local_method != rec.compression_method().as_u16() {
                return Err(EngineError::SecurityLimit(format!(
                    "zip cloaking method mismatch for {clean_path}"
                )));
            }
            let local_name_len = u16::from_le_bytes([local_hdr[26], local_hdr[27]]) as usize;
            let mut local_name_buf = vec![0u8; local_name_len];
            archive
                .get_ref()
                .read_exact_at(&mut local_name_buf, rec.local_header_offset() + 30)
                .map_err(|e| EngineError::DecodeFailed(format!("zip local name error: {e}")))?;
            let local_name_str = std::str::from_utf8(&local_name_buf)
                .map_err(|_| EngineError::SecurityLimit("non-UTF-8 local zip filename".to_string()))?;
            let local_clean = sanitize_archive_path(local_name_str)?;
            if local_clean != clean_path {
                return Err(EngineError::SecurityLimit(format!(
                    "zip cloaking path mismatch between local '{local_clean}' and central '{clean_path}'"
                )));
            }

            let wayfinder = rec.wayfinder();
            let zip_entry = archive
                .get_entry(wayfinder)
                .map_err(|e| EngineError::DecodeFailed(format!("zip entry locate error: {e}")))?;
            ranges.push(zip_entry.compressed_data_range());

            metas.push(VerifiedMeta {
                clean_path,
                method: rec.compression_method(),
                compressed_size: comp,
                uncompressed_size: uncomp,
                wayfinder,
            });
        }
    }

    check_non_overlapping_ranges(&mut ranges)?;

    let mut out = Vec::with_capacity(metas.len());
    for meta in metas {
        let zip_entry = archive
            .get_entry(meta.wayfinder)
            .map_err(|e| EngineError::DecodeFailed(format!("zip entry reopen error: {e}")))?;
        let data = match meta.method {
            CompressionMethod::STORE => {
                let r = zip_entry.verifying_reader(zip_entry.reader());
                tracker.read_bounded(r, Some(meta.compressed_size))?
            }
            CompressionMethod::DEFLATE => {
                let dec = DeflateDecoder::new(zip_entry.reader());
                let r = zip_entry.verifying_reader(dec);
                tracker.read_bounded(r, Some(meta.compressed_size))?
            }
            other => {
                return Err(EngineError::UnsupportedFormat(format!(
                    "unsupported zip compression method {} for {}",
                    other.as_u16(),
                    meta.clean_path
                )));
            }
        };
        if (data.len() as u64) != meta.uncompressed_size && meta.uncompressed_size > 0 {
            return Err(EngineError::SecurityLimit(format!(
                "zip size mismatch for {}",
                meta.clean_path
            )));
        }
        out.push(ArchiveEntryInput {
            path: meta.clean_path,
            data,
        });
    }

    Ok(out)
}

pub fn write_zip_entries<W: Write>(
    writer: W,
    entries: &[ArchiveEntryOutput],
) -> EngineResult<W> {
    let mut zip_writer = ZipArchiveWriter::new(writer);
    for entry in entries {
        let method = if should_store_uncompressed(&entry.path) {
            CompressionMethod::STORE
        } else {
            CompressionMethod::DEFLATE
        };
        let (mut z_entry, cfg) = zip_writer
            .new_file(&entry.path)
            .compression_method(method)
            .start()
            .map_err(|e| EngineError::EncodeFailed(format!("zip write header failed: {e}")))?;

        let descriptor = match method {
            CompressionMethod::STORE => {
                let mut w = cfg.wrap(&mut z_entry);
                w.write_all(&entry.data)?;
                let (_, desc) = w
                    .finish()
                    .map_err(|e| EngineError::EncodeFailed(format!("zip store finish failed: {e}")))?;
                desc
            }
            _ => {
                let enc = DeflateEncoder::new(&mut z_entry, Compression::new(6));
                let mut w = cfg.wrap(enc);
                w.write_all(&entry.data)?;
                let (enc, desc) = w
                    .finish()
                    .map_err(|e| EngineError::EncodeFailed(format!("zip deflate finish failed: {e}")))?;
                enc.finish()?;
                desc
            }
        };
        z_entry
            .finish(descriptor)
            .map_err(|e| EngineError::EncodeFailed(format!("zip entry finalize failed: {e}")))?;
    }
    zip_writer
        .finish()
        .map_err(|e| EngineError::EncodeFailed(format!("zip central dir write failed: {e}")))
}

fn should_store_uncompressed(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".png")
        || lower.ends_with(".webp")
        || lower.ends_with(".avif")
        || lower.ends_with(".jxl")
        || lower.ends_with(".gif")
        || lower.ends_with(".zip")
        || lower.ends_with(".zst")
        || lower.ends_with(".gz")
        || lower.ends_with(".xz")
        || lower.ends_with(".bz2")
        || lower.ends_with(".7z")
}
