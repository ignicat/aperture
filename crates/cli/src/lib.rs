#![forbid(unsafe_code)]

pub mod args;
pub mod config;
pub mod discover;
pub mod journal;
pub mod linux;
pub mod preview;
pub mod progress;
pub mod runner;
pub mod watch;

pub fn run(program: &str) -> u8 {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    runner::execute(program, &raw_args)
}

#[cfg(test)]
mod tests {
    use super::args::{Command, parse_args, parse_byte_size};
    use super::runner::{EXIT_ARGS, EXIT_OK, execute};
    use engine::codec::{EncodeOptions, ImageFormat, decode_image, encode_image};
    use engine::color::ColorSpace;
    use engine::pixel::{PackedImage, PixelFormat};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIR_SEQ: AtomicU64 = AtomicU64::new(1);

    fn make_temp_dir(label: &str) -> PathBuf {
        let id = TEST_DIR_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "img_cli_test_{label}_{}_{}",
            std::process::id(),
            id
        ));
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn cli_parser_supports_shorthand_cyrillic_and_presets() {
        let raw = vec![
            "photo.png".to_string(),
            "1920х1080".to_string(),
            "--algo=fsr".to_string(),
            "-m".to_string(),
            "smart-crop".to_string(),
            "--safe-zone".to_string(),
            "top:48px,bottom:5%".to_string(),
            "--replace".to_string(),
            "--backup".to_string(),
            "--preset".to_string(),
            "web-opt".to_string(),
        ];
        let parsed = parse_args("aperture", &raw);
        assert!(parsed.is_ok(), "parse failed: {parsed:?}");
        if let Ok(opts) = parsed {
            assert_eq!(opts.command, Command::Process);
            assert_eq!(opts.inputs, vec!["photo.png"]);
            assert!(opts.replace);
            assert!(opts.backup);
            assert_eq!(opts.pipeline.targets.len(), 1);
        }

        assert_eq!(parse_byte_size("10MiB").ok(), Some(10 * 1024 * 1024));
        assert_eq!(parse_byte_size("500KB").ok(), Some(500_000));
    }

    #[test]
    fn cli_end_to_end_alongside_replace_compare_inspect_and_archive() {
        let dir = make_temp_dir("e2e");
        let mut rgba = Vec::with_capacity(24 * 16 * 4);
        for y in 0..16u32 {
            for x in 0..24u32 {
                rgba.push((x * 10) as u8);
                rgba.push((y * 15) as u8);
                rgba.push(140);
                rgba.push(255);
            }
        }
        let img = PackedImage::new(24, 16, PixelFormat::Rgba8, ColorSpace::Srgb, rgba);
        assert!(img.is_ok());
        let Ok(img) = img else {
            return;
        };
        let png = encode_image(&img, ImageFormat::Png, &EncodeOptions::default(), None);
        assert!(png.is_ok());
        let Ok(png_bytes) = png else {
            return;
        };

        let sample_path = dir.join("wall.png");
        assert!(fs::write(&sample_path, &png_bytes).is_ok());

        let code = execute(
            "aperture",
            &[
                sample_path.to_string_lossy().into_owned(),
                "48х32".to_string(),
                "--algo".to_string(),
                "fsr".to_string(),
                "--verify".to_string(),
                "--keep-dates".to_string(),
            ],
        );
        assert_eq!(code, EXIT_OK);
        let alongside_out = dir.join("wall_48x32.png");
        assert!(alongside_out.exists());

        if let Ok(out_bytes) = fs::read(&alongside_out)
            && let Ok(dec) = decode_image(&out_bytes, Some("png"), &Default::default())
        {
            assert_eq!((dec.image.width, dec.image.height), (48, 32));
        } else {
            assert_eq!(0, 1);
        }

        let replace_src = dir.join("target.png");
        assert!(fs::write(&replace_src, &png_bytes).is_ok());
        let resume_file = dir.join("job.journal");
        let rep_code = execute(
            "aperture",
            &[
                replace_src.to_string_lossy().into_owned(),
                "2x".to_string(),
                "-i".to_string(),
                "--backup".to_string(),
                "--resume".to_string(),
                resume_file.to_string_lossy().into_owned(),
            ],
        );
        assert_eq!(rep_code, EXIT_OK);
        assert!(dir.join("target.png.bak").exists());
        assert!(resume_file.exists());
        if let Ok(rep_bytes) = fs::read(&replace_src)
            && let Ok(dec) = decode_image(&rep_bytes, Some("png"), &Default::default())
        {
            assert_eq!((dec.image.width, dec.image.height), (48, 32));
        }

        let insp_code = execute(
            "aperture",
            &[
                "inspect".to_string(),
                alongside_out.to_string_lossy().into_owned(),
                "--json".to_string(),
            ],
        );
        assert_eq!(insp_code, EXIT_OK);

        let cmp_code = execute(
            "aperture",
            &[
                "compare".to_string(),
                alongside_out.to_string_lossy().into_owned(),
                replace_src.to_string_lossy().into_owned(),
                "--json".to_string(),
            ],
        );
        assert_eq!(cmp_code, EXIT_OK);

        let zip_path = dir.join("pack.zip");
        let arc_bytes = engine::archive::write_archive_entries(
            &[engine::archive::ArchiveEntryOutput {
                path: "sub/item.png".to_string(),
                data: png_bytes,
                was_image: true,
            }],
            engine::archive::ArchiveFormat::Zip,
        );
        assert!(arc_bytes.is_ok());
        if let Ok(ab) = arc_bytes {
            assert!(fs::write(&zip_path, ab).is_ok());
            let arc_out = dir.join("pack_out.tar.zst");
            let arc_code = execute(
                "aperture",
                &[
                    zip_path.to_string_lossy().into_owned(),
                    "32x32".to_string(),
                    "--fit".to_string(),
                    "contain".to_string(),
                    "--archive-format".to_string(),
                    "tar.zst".to_string(),
                    "--out".to_string(),
                    arc_out.to_string_lossy().into_owned(),
                ],
            );
            assert_eq!(arc_code, EXIT_OK);
            assert!(arc_out.exists());
        }

        let bad_code = execute("aperture", &["--unknown-flag".to_string()]);
        assert_eq!(bad_code, EXIT_ARGS);

        let _ = fs::remove_dir_all(&dir);
    }
}
