use crate::archive::security::{ArchiveBudgetTracker, sanitize_archive_path};
use crate::archive::{ArchiveEntryInput, ArchiveEntryOutput, ArchiveFormat};
use crate::error::{EngineError, EngineResult};
use bzip2_rs::DecoderReader as Bz2Decoder;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use lzma_rs::xz_compress;
use lzma_rs::xz_decompress;
use ruzstd::decoding::StreamingDecoder as ZstdDecoder;
use ruzstd::encoding::{CompressionLevel as ZstdLevel, compress as zstd_compress};
use std::io::{BufReader, Cursor, Read, Write};
use tar::{Archive, Builder, Header};

pub fn read_tar_stream<R: Read>(
    reader: R,
    format: ArchiveFormat,
    tracker: &mut ArchiveBudgetTracker,
) -> EngineResult<Vec<ArchiveEntryInput>> {
    match format {
        ArchiveFormat::Tar => read_tar_inner(reader, tracker),
        ArchiveFormat::TarGz => read_tar_inner(GzDecoder::new(reader), tracker),
        ArchiveFormat::TarZst => {
            let dec = ZstdDecoder::new(reader)
                .map_err(|e| EngineError::DecodeFailed(format!("zstd stream init failed: {e}")))?;
            read_tar_inner(dec, tracker)
        }
        ArchiveFormat::TarXz => {
            let mut buf_reader = BufReader::new(reader);
            let mut raw_tar = Vec::new();
            let max_tar = tracker.policy().max_total_bytes.min(512 * 1024 * 1024);
            let mut limited = LimitedVecWriter {
                buf: &mut raw_tar,
                limit: max_tar,
            };
            xz_decompress(&mut buf_reader, &mut limited)
                .map_err(|e| EngineError::DecodeFailed(format!("xz decompress failed: {e}")))?;
            read_tar_inner(Cursor::new(raw_tar), tracker)
        }
        ArchiveFormat::TarBz2 => read_tar_inner(Bz2Decoder::new(reader), tracker),
        _ => Err(EngineError::UnsupportedFormat(
            "expected tar-family archive format".to_string(),
        )),
    }
}

struct LimitedVecWriter<'a> {
    buf: &'a mut Vec<u8>,
    limit: u64,
}

impl Write for LimitedVecWriter<'_> {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        let next = (self.buf.len() as u64).saturating_add(data.len() as u64);
        if next > self.limit {
            return Err(std::io::Error::other("xz stream exceeded max total byte cap"));
        }
        self.buf.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn read_tar_inner<R: Read>(
    reader: R,
    tracker: &mut ArchiveBudgetTracker,
) -> EngineResult<Vec<ArchiveEntryInput>> {
    let mut archive = Archive::new(reader);
    archive.set_ignore_zeros(false);
    let mut out = Vec::new();

    let entries = archive
        .entries()
        .map_err(|e| EngineError::DecodeFailed(format!("tar read error: {e}")))?;
    for entry_res in entries {
        let entry =
            entry_res.map_err(|e| EngineError::DecodeFailed(format!("tar entry error: {e}")))?;
        let etype = entry.header().entry_type();
        let raw_path = entry
            .path()
            .map_err(|e| EngineError::SecurityLimit(format!("invalid tar path: {e}")))?
            .to_string_lossy()
            .into_owned();

        if etype.is_dir() {
            continue;
        }
        if etype.is_symlink() || etype.is_hard_link() {
            if !tracker.policy().allow_symlinks {
                return Err(EngineError::SecurityLimit(format!(
                    "symlink/hardlink entry rejected in tar archive: {raw_path}"
                )));
            }
            continue;
        }
        if etype.is_character_special() || etype.is_block_special() || etype.is_fifo() {
            return Err(EngineError::SecurityLimit(format!(
                "special device/fifo entry rejected in tar archive: {raw_path}"
            )));
        }
        if !etype.is_file() {
            continue;
        }

        let clean_path = sanitize_archive_path(&raw_path)?;
        let declared_size = entry.header().size().unwrap_or(0);
        tracker.record_entry_header(declared_size, None)?;
        let data = tracker.read_bounded(entry, None)?;
        out.push(ArchiveEntryInput {
            path: clean_path,
            data,
        });
    }

    Ok(out)
}

pub fn write_tar_entries(
    entries: &[ArchiveEntryOutput],
    format: ArchiveFormat,
) -> EngineResult<Vec<u8>> {
    let raw_tar = build_tar_bytes(entries)?;
    match format {
        ArchiveFormat::Tar => Ok(raw_tar),
        ArchiveFormat::TarGz => {
            let mut enc = GzEncoder::new(Vec::new(), Compression::new(6));
            enc.write_all(&raw_tar)?;
            Ok(enc.finish()?)
        }
        ArchiveFormat::TarZst => {
            let mut out = Vec::new();
            zstd_compress(Cursor::new(&raw_tar), &mut out, ZstdLevel::Fastest);
            Ok(out)
        }
        ArchiveFormat::TarXz => {
            let mut out = Vec::new();
            let mut reader = BufReader::new(Cursor::new(&raw_tar));
            xz_compress(&mut reader, &mut out)
                .map_err(|e| EngineError::EncodeFailed(format!("xz compress failed: {e}")))?;
            Ok(out)
        }
        ArchiveFormat::TarBz2 => Err(EngineError::UnsupportedFormat(
            "writing .tar.bz2 is unsupported in pure-Rust mode; use zip, tar.zst, tar.gz, or tar.xz".to_string(),
        )),
        _ => Err(EngineError::UnsupportedFormat(
            "unsupported tar output format".to_string(),
        )),
    }
}

fn build_tar_bytes(entries: &[ArchiveEntryOutput]) -> EngineResult<Vec<u8>> {
    let mut builder = Builder::new(Vec::new());
    for entry in entries {
        let mut header = Header::new_gnu();
        header.set_size(entry.data.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, &entry.path, entry.data.as_slice())
            .map_err(|e| EngineError::EncodeFailed(format!("tar entry append failed: {e}")))?;
    }
    builder
        .into_inner()
        .map_err(|e| EngineError::EncodeFailed(format!("tar finish failed: {e}")))
}
