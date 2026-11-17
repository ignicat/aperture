# User Guide

Practical walkthroughs covering `aperture` workflows from basic scaling to multi-monitor wallpapers, archive batches, forensics, and embedding `engine` in Rust projects.

For the exhaustive flag-by-flag specification, see [CLI Reference](CLI.md).

---

## 1. Basic Scaling and Output Placement

If you pass an input image and a target resolution, `aperture` leaves the original file untouched and writes a new file alongside it (`<stem>_<W>x<H>.<ext>`):

```sh
aperture photo.png 3840x2160
```

ASCII `x`, Cyrillic `х`, `*`, `:`, and resolution aliases (`hd`, `fhd`, `qhd`, `uwqhd`, `4k`, `8k`) are all valid:

```sh
aperture photo.png 1920х1080
aperture photo.png 4k
aperture photo.png 50%
aperture photo.png 2x
aperture photo.png 1920w
```

Control where and how outputs are written:

```sh
# Explicit output file or target directory
aperture photo.png 1920x1080 -o scaled/photo_fhd.png

# Generate several sizes in one decode pass
aperture hero.png 1280x720,1920x1080,3840x2160 -o ./exports --name "{stem}_{w}x{h}.{ext}"

# Atomically overwrite the original file (writes temp file, fsyncs, backs up, renames)
aperture photo.jpg 2560x1440 -i --backup --keep-dates --verify

# Read from stdin and write encoded PNG bytes to stdout
cat raw.ppm | aperture - 800x600 -f png --stdout > out.png
```

---

## 2. Choosing Scaling Algorithms and Color Space

When `--algo` is `auto` (the default), `aperture` classifies the image (photograph, flat/UI graphics, line art, or pixel art) and picks a suitable kernel automatically. You can override this explicitly:

| Workload | Recommended `--algo` | Notes |
| --- | --- | --- |
| Mixed photos & sharp illustrations | `adaptive` | Classifies each 8x8 tile independently between smooth/sharp edge and EWA passes |
| Crisp enlargement without halos | `fsr` | AMD FSR 1.0 EASU 12-tap edge direction + RCAS sharpening (`--sharpen`) |
| General photography (up or down) | `ewa-lanczossharp` | Polar 2D Jinc-windowed Lanczos with `--antiring` clamping |
| Detail-preserving upscaling | `ibp` or `ssim-up` | Iterative back-projection (`--ibp-iters`) and SSIM super-resolution |
| High-ratio downscaling | `ssim-down` or `box` | Perceptual structural downscaling (`ssim-down`) or exact box averaging (`box`) |
| Anime, manga, scanned UI/line art | `lineart` | Non-neural DoG deblurring with dark/thin-line reinforcement |
| Pixel-art sprites | `mmpx`, `xbrz`, `scalefx` | Clean corner/diagonal interpolation; `mmpx` and `scalefx` preserve palette colors |

### Working Color Space

Linear-light resampling (`--linear`, enabled by default) prevents gamma darkening on thin lines, gradients, and fine textures. Two alternatives help in specific cases:

- `--sigmoidal`: compresses high-contrast edges before filtering (contrast factor `6.5` by default, tunable up to `12.0`) to suppress dark/bright ringing halos.
- `--oklab`: filters in perceptual Oklab space to prevent hue shifts on saturated transitions.
- `--chroma krig`: enables `KrigBilateral` chroma-from-luma interpolation during upscaling.
- `--no-alpha-fix`: disables transparent border color bleed repair (`--alpha-fix` is on by default).

```sh
aperture illustration.png 4k --algo fsr --sigmoidal --sharpen 35 --chroma krig
```

---

## 3. Inspecting and Repairing Degraded Inputs

Before blindly upscaling an image from the web, inspect it to see if it has already been upscaled with blurry filters, JPEG artifacts, or banding:

```sh
aperture inspect suspect.jpg
```

If `forensics` reports `fake_upscale=true` with a confident `native=WxH`, use `--descaled` so `aperture` downscales back to native grid before applying edge-aware upscaling:

```sh
# Undo blurry synthetic upscale, remove JPEG blocks, deband, and re-upscale with dithering
aperture suspect.jpg 3840x2160 --descaled --dejpeg 50 --denoise 20 --algo ewa-lanczossharp --deband 35 --dither blue-noise
```

To compare any processed result quantitatively against a reference image:

```sh
aperture compare reference.png candidate.png
```

To benchmark kernel speeds and roundtrip `SSIM2` on a specific image size on your CPU:

```sh
aperture bench sample.png 2560x1440 --iters 5
```

---

## 4. Wallpapers and Multi-Monitor Layouts

Aperture queries your Wayland (`wl_output` / `xdg_output_manager_v1`), X11 (`RandR`), or `/sys/class/drm` outputs directly when `--screen` is passed.

### Automatic Single-Monitor Wallpaper

```sh
# Query active monitor, preserve focal subject, match colors, and set wallpaper
aperture wallpaper.png --screen auto --fit auto --algo adaptive --dither blue-noise --set
```

### Aspect Ratio Fitting Modes (`-m` / `--fit`)

When source aspect ratio differs from the target monitor:

- `auto`: uses smart saliency-guided cropping for small aspect differences, and automatically falls back to blurred ambient framing when cropping would discard important subject area.
- `smart-crop`: crops using integral-image saliency (edge energy, skin tone cues, saturation, spectral residual) weighted by `--gravity` and `--safe-zone`.
- `seam-carve`: forward-energy dynamic programming retargeting that shrinks/expands low-energy sky or background bands while leaving primary subjects undistorted.
- `ambient`: centers the scaled image inside a blurred (`--bg-blur`), dimmed (`--bg-dim`) copy of itself with drop shadow (`--shadow`) and optional corner rounding (`--radius`).
- `contain`, `extend`, `mirror`, `gradient`, `stretch`, `tile`, `split`: explicit letterbox/matte (`--bg-color`), edge extension, mirrored borders, Oklab gradient fill, exact stretch, tiling, or multi-monitor split.

### Avoiding Desktop Panels and Dock Bars (`--safe-zone`)

Keep subjects out of regions covered by bars like Waybar or GNOME top panels:

```sh
aperture wall.jpg --screen auto --fit smart-crop --safe-zone top:48px,bottom:5% --set
```

### Multi-Monitor Span with Bezel Compensation

Split one wide panoramic source across physical displays while subtracting bezel gaps:

```sh
aperture panorama.jpg --screen span --bezel 24px -o ./wallpapers
```

Extract matching UI accent colors at the same time:

```sh
aperture wall.png --screen auto --fit auto --palette css > theme-colors.css
```

---

## 5. Compression, Lossless Mode, and Size Budgets

Lossless sources (`png`, `qoi`, `bmp`, `tga`, `tiff`, `pnm`, `ff`, `hdr`, `exr`) encode losslessly by default. Lossy JPEG inputs re-encode at `quality=98` with `4:4:4` chroma subsampling (`--chroma-subsampling 444`) unless you customize encoding:

```sh
# Force lossless output (PNG/WebP/JXL depending on format)
aperture photo.jpg 2x --lossless -f webp

# Quantize PNG to 128 Oklab palette colors with compression optimization
aperture ui_capture.png 1280x720 --png-quant 128 --optimize

# Automatically find highest compression achieving perceptual SSIM2 >= 86
aperture photo.png 1920x1080 --target-ssim2 86 -f jpg --meta strip-privacy

# Enforce strict byte budget
aperture photo.png 1920x1080 -f webp --max-bytes 250KB --meta strip-all
```

---

## 6. Batch Processing, Archives, Resume, and Watch Mode

### Recursive Directories and Filtering

```sh
# Process only landscape photos larger than 1000x600 while skipping those already small enough
aperture ./photos "1920x1080>" -r --include "*.jpg" --include "*.png" \
  --exclude "*thumb*" --only-orientation landscape --min-res 1000x600 \
  --link-skipped -o ./export --name "{dir}/{stem}.{ext}"
```

### Nested Archives and Deduplication

Process `.zip` / `.cbz` / `.tar` / `.tar.gz` / `.tar.zst` / `.tar.xz` / `.tar.bz2` inputs in memory under strict decompression budgets without temporary extract directories:

```sh
# Scale comic/image archive, drop exact duplicates, keep non-image files, repack as tar.zst
aperture collection.zip 1920w -f webp --target-ssim2 88 \
  --archive-mode repack --archive-format tar.zst \
  --dedup exact --non-images keep -o collection_1920w.tar.zst
```

Adjust archive limits if working with unusually large datasets: `--max-entry-size 1GiB`, `--max-total-size 8GiB`, `--max-ratio 200`, `--archive-depth 6`.

### Resumable Batches and Live Directory Watch

```sh
# Save progress incrementally so interrupted runs resume cleanly
aperture ./huge_dataset 2048w -r -o ./out --resume ./batch.journal --verify

# Automatically process new or updated images dropped into a directory
aperture ./incoming 1920x1080 --watch --fit smart-crop --algo fsr -o ./ready
```

### Presets and Terminal Previews

```sh
# Inspect any image directly inside foot/kitty/truecolor terminals
aperture hero.png 640x360 --preview-only --preview-proto sixel

# Save a reusable preset and apply it later
aperture preset save blog-hero "1600w --algo ewa-lanczossharp --target-ssim2 88 --meta strip-privacy"
aperture draft.png -p blog-hero -o ./blog/hero.jpg
```

---

## 7. Using the Core Library in Rust

Add `core` (`lib` name `engine`) from this repository to your `Cargo.toml`:

```toml
[dependencies]
engine = { package = "core", path = "crates/core" }
```

Minimal scaling and encoding pipeline:

```rust
use engine::arena::{MemoryBudget, PlanCache};
use engine::error::EngineResult;
use engine::geometry::spec::TargetGeometry;
use engine::pipeline::{PipelineConfig, process_image_bytes};
use engine::pool::ThreadPool;
use std::fs;

fn main() -> EngineResult<()> {
    let bytes = fs::read("input.png")?;
    let mut cfg = PipelineConfig::default();
    cfg.targets.push(TargetGeometry::parse("1920x1080")?);

    let budget = MemoryBudget::from_system_fraction(1, 2);
    let cache = PlanCache::new(32);
    let pool = ThreadPool::new(0);

    let outputs = process_image_bytes(&bytes, Some("png"), &cfg, &budget, &cache, &pool)?;
    if let Some(first) = outputs.first() {
        fs::write("output.png", &first.bytes)?;
    }
    Ok(())
}
```

For lower-level control over decoded planes, inspect `engine::codec::decode_image`, `engine::resample::resample_planar`, `engine::geometry::fit::fit_planar`, `engine::inspect::inspect_image_bytes`, and `engine::inspect::metrics::compare_images`.
