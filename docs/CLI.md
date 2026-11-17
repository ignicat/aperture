# CLI Reference

Complete command, syntax, flag, and environment reference for `aperture`.

## Synopsis

```text
aperture <inputs...> [target] [flags]
aperture <subcommand> [options]
```

## Subcommands

| Command | Aliases | Behavior |
| --- | --- | --- |
| `process` | (default) | Decode, scale/restore, and write images or directories |
| `fit` | `wp` | Same pipeline with wallpaper fitting workflows |
| `archive` | `arc` | Process archive files (`.zip`, `.tar`, `.tar.gz`, `.tar.zst`, `.tar.xz`, `.tar.bz2`) |
| `inspect` | `info` | Analyze headers, signal stats, palette, and synthetic upscale probability |
| `compare` | `diff` | Compute `MSE`, `PSNR`, `SSIM`, `SSIM2`, `Butteraugli`, and `DSSIM` between two images |
| `bench` | - | Benchmark scaling kernels (`fsr`, `ewa-lanczossharp`, `lanczos3`, `ssim-up`, `ibp`, `mitchell`) and report throughput plus roundtrip `SSIM2` |
| `watch` | - | Monitor input directories with Linux `inotify` and re-process modified images |
| `preset` | - | Manage saved presets (`list`, `path`, `show <name>`, `save <name> <flags...>`, `delete <name>`) |
| `completions` | - | Emit shell completion scripts (`bash`, `zsh`, `fish`, `nushell`) |

Use `--` before positional arguments that start with `-` (except `-` alone, which reads image bytes from `stdin`).

---

## Target Geometry DSL

Targets may be passed positionally (`aperture in.png 1920x1080`), via `-t` / `--target <spec>`, as comma-separated lists (`1280x720,1920x1080`), or resolved from active Linux monitors (`--screen`).

### Separators and Wildcards

| Form | Example | Behavior |
| --- | --- | --- |
| Exact box | `1920x1080`, `1920х1080`, `1920*1080`, `1920:1080`, `1920×1080` | Fit or scale into `1920x1080` (accepts ASCII `x`/`X`, Cyrillic `х`/`Х`, `*`, `:`, `×`) |
| Width-driven | `1920w`, `1920x_`, `1920x0` | Set width to `1920`, compute height preserving aspect ratio |
| Height-driven | `1080h`, `_x1080`, `0x1080` | Set height to `1080`, compute width preserving aspect ratio |
| Scale factor | `2x`, `2х`, `2×`, `150%` | Multiply source dimensions by `factor` or percentage |
| Aspect ratio | `ar:21:9`, `ar:21:9+` | Crop (`ar:W:H`) or pad/expand (`ar:W:H+`) to exact aspect ratio |
| Pixel budget | `mp:12`, `@2000000` | Scale preserving aspect ratio to match target pixel count |

### Constraint Modifiers (Suffixes)

| Suffix | Meaning |
| --- | --- |
| `>` | Shrink only: skip if source is already smaller than target |
| `<` | Enlarge only: skip if source is already larger than target |
| `^` | Cover minimum: scale proportionally so both dimensions are at least the target box |
| `!` | Force exact stretch without aspect preservation (`FitMode::Stretch`) |

### Built-in Target Aliases

| Alias | Dimensions | Alias | Dimensions |
| --- | --- | --- | --- |
| `vga` | `640x480` | `uwfhd` | `2560x1080` |
| `hd`, `720p` | `1280x720` | `uwqhd` | `3440x1440` |
| `fhd`, `1080p` | `1920x1080` | `4k`, `uhd`, `2160p` | `3840x2160` |
| `wuxga` | `1920x1200` | `dci4k` | `4096x2160` |
| `qhd`, `wqhd`, `2k`, `1440p` | `2560x1440` | `5k` | `5120x2880` |
| `wqxga` | `2560x1600` | `dqhd`, `suwqhd` | `5120x1440` |
| `6k` | `6016x3384` | `8k`, `4320p` | `7680x4320` |
| `fhd-portrait` | `1080x1920` | `qhd-portrait` | `1440x2560` |
| `og`, `og-image` | `1200x630` | `a4-300` / `a4-landscape` | `2480x3508` / `3508x2480` |

---

## Byte Sizes and Percentages

Flags taking sizes (`--max-bytes`, `--memory-limit`, `--max-entry-size`, `--max-total-size`) accept decimal SI (`b`, `k`/`kb`, `m`/`mb`, `g`/`gb`) and binary IEC units (`ki`/`kib`, `mi`/`mib`, `gi`/`gib`). Examples: `500KB`, `10MiB`, `2G`.

Strength flags (`--antiring`, `--dejpeg`, `--denoise`, `--deband`, `--sharpen`, `--grain`, `--bg-dim`, `--shadow`, `--dim`, `--vignette`) accept either normalized `0.0..1.0` floats (`0.35`) or integer/float percentages (`35`, `35%`). Values strictly above `1.0` without `%` are divided by `100.0`.

---

## Flag Reference

### General

| Flag | Short | Argument | Default | Description |
| --- | --- | --- | --- | --- |
| `--help` | `-h` | - | `false` | Print command help and exit |
| `--version` | `-V` | - | `false` | Print binary name and version and exit |
| `--preset` | `-p` | `<name>` | none | Prepend flags stored under `<name>` in config or built-in presets |
| `--target` | `-t` | `<spec[,spec...]>` | none | Add one or more comma-separated target geometries |
| `--screen` | - | `[mode]` | none | Resolve target from Linux displays: `auto`, `<OUTPUT_NAME>`, `all`, `per-screen`, `span` |
| `--dry-run` | `-n` | - | `false` | Plan actions without writing output files or setting wallpapers |

### Scaling and Color Pipeline

| Flag | Short | Argument | Default | Description |
| --- | --- | --- | --- | --- |
| `--algo` | `-a` | `<kernel>` | `auto` | Set resampling algorithm (see table below) |
| `--ibp-iters` | - | `<0..6>` | `2` | Number of iterative back-projection refinement iterations |
| `--antiring` | - | `<0..1 \| %>` | `0.65` | Clamping anti-ringing weight for EWA/sinc passes |
| `--linear` | - | - | active default | Resample in linearized sRGB light (`WorkingSpace::LinearSrgb`) |
| `--oklab` | - | - | off | Resample in perceptual Oklab color space (`WorkingSpace::Oklab`) |
| `--sigmoidal` | - | `[1.0..12.0]` | `6.5` when set | Resample through sigmoidal contrast compression to reduce high-contrast ringing |
| `--chroma` | - | `<krig \| lanczos>` | `lanczos` | Chroma-from-luma upscaling (`krig` enables `KrigBilateral` refinement) |
| `--premultiply` | - | `<true \| false \| 1 \| 0 \| yes \| no>` | `true` | Premultiply RGB by alpha during resampling |
| `--alpha-fix` | - | - | `true` | Inpaint transparent-edge bleed before premultiplication |
| `--no-alpha-fix` | - | - | `false` | Disable transparent-edge alpha bleed repair |

Accepted `--algo` values: `auto`, `adaptive`, `fsr` (`easu`), `ibp`, `ewa-lanczos` (`ewalanczos`), `ewa-lanczossharp` (`lanczossharp`), `ewa-robidoux` (`robidoux`), `ewa-robidouxsharp` (`robidouxsharp`), `ewa-ginseng` (`ginseng`), `lanczos3` (`lanczos`), `lanczos4`, `ssim-down` (`ssimdown`), `ssim-up` (`ssimup`, `ssimsuperres`), `icbi`, `nedi`, `lineart` (`anime`), `mmpx`, `xbrz`, `scalefx`, `mitchell`, `catmull` (`catmull-rom`, `bicubic`), `hermite`, `box` (`area`), `nearest` (`point`).

### Restoration and Finishing

| Flag | Short | Argument | Default | Description |
| --- | --- | --- | --- | --- |
| `--descaled` | - | - | `false` | Detect suspected prior fake upscaling and downscale to native resolution first |
| `--dejpeg` | - | `[0..1 \| %]` | `0` (`0.6` if bare) | Apply DCT/boundary JPEG block smoothing before scaling |
| `--denoise` | - | `[0..1 \| %]` | `0` (`0.4` if bare) | Apply edge-preserving guided denoising before scaling |
| `--deband` | - | `[0..1 \| %]` | `0` (`0.5` if bare) | Apply stochastic multi-radius debanding after scaling |
| `--sharpen` | - | `[0..1 \| %]` | `0.35` (`0.5` if bare) | RCAS contrast-adaptive sharpening strength (alias: `--rcas`) |
| `--rcas` | - | `[0..1 \| %]` | `0.35` (`0.5` if bare) | Alias for `--sharpen` |
| `--dither` | - | `<mode>` | `none` | Dithering before quantization: `none` (`off`), `bayer` (`ordered`), `floyd` (`floyd-steinberg`, `fs`), `blue-noise` (`blue`, `bluenoise`) |
| `--grain` | - | `[0..1 \| %]` | `0` (`0.35` if bare) | Fine luma film grain added during finishing |

### Aspect Fitting and Display Layout

| Flag | Short | Argument | Default | Description |
| --- | --- | --- | --- | --- |
| `--fit` | `-m` | `<mode>` | `auto` | Aspect-fit mode: `auto`, `smart-crop`, `seam-carve`, `ambient`, `extend`, `mirror`, `gradient`, `contain`, `stretch`, `tile`, `split` |
| `--gravity` | - | `<anchor>` | `auto` | Crop/placement anchor: `auto`, `c`, `n`, `s`, `e`, `w`, `ne`, `nw`, `se`, `sw`, or `x%,y%` |
| `--safe-zone` | - | `<edge:val[,...]>` | `0` | Protect screen margins from crop focus: `<top \| bottom \| left \| right \| all>:<px \| %>` |
| `--trim-borders` | - | - | `false` | Trim solid/letterbox borders before fitting or scaling |
| `--bg-color` | - | `<#hex \| named>` | `#000000` | Matte background color (`#rgb`, `#rrggbb`, `#rrggbbaa`, `oled`, `black`, `white`, `transparent`) |
| `--bg-blur` | - | `<N[px]>` | `28` | Blur radius for `ambient` padding background |
| `--bg-dim` | - | `<0..1 \| %>` | `45%` | Background dimming under `ambient` foreground frame |
| `--shadow` | - | `[0..1 \| %]` | `35%` | Drop shadow intensity under inset image in `ambient`/`contain` modes |
| `--radius` | - | `<N[px]>` | `0` | Rounded corner radius in pixels for inset compositing |
| `--dim` | - | `<0..1 \| %>` | `0` | Darken final image foreground (`0..100%`) |
| `--vignette` | - | `<0..1 \| %>` | `0` | Radial edge vignette intensity |
| `--bezel` | - | `<N[px]>` | `0` | Physical monitor bezel width in pixels subtracted between spans under `--screen span` |
| `--palette` | - | `[json \| css]` | none | Extract 5-color Oklab palette (`json` when flag passed bare, or `css` custom properties) |
| `--set` | - | - | `false` | Apply generated output image(s) as desktop wallpaper |

### Output, Formats, and Encoding

| Flag | Short | Argument | Default | Description |
| --- | --- | --- | --- | --- |
| `--alongside` | - | - | `true` | Save outputs next to source files with dimension/template suffix |
| `--replace` | `-i` | - | `false` | Atomically overwrite source file (`temp` + `fsync` + `rename`; alias `--in-place`) |
| `--in-place` | `-i` | - | `false` | Alias for `--replace` |
| `--backup` | - | - | `false` | Save `<file>.bak` before replacing under `-i` / `--replace` |
| `--out` | `-o` | `<path>` | none | Output file path, output directory, or output archive path |
| `--name` | - | `<template>` | none | Naming template (relative to `-o` or working directory) |
| `--suffix` | - | `<text>` | `_<W>x<H>` | Custom filename suffix when saving alongside or mirroring directories |
| `--on-conflict` | - | `<policy>` | `rename` | Existing destination behavior: `skip`, `overwrite`, `rename`, `smaller` |
| `--format` | `-f` | `<fmt>` | `auto` | Output container: `auto`, `png`, `jpg`, `webp`, `qoi`, `tiff`, `bmp`, `tga`, `pnm`, `ico`, `ff`, `hdr`, `exr`, `jxl`, `avif`, `svg` |
| `--lossless` | - | - | `false` | Force lossless encoding when container supports it |
| `--optimize` | - | - | `true` | Apply lossless stream/filter optimization |
| `--png-quant` | - | `[2..256]` | none (`256` if bare) | Quantize PNG output to an Oklab palette of up to `N` colors |
| `--quality` | `-q` | `<1..100>` | `98` | Quality level for lossy encoders |
| `--target-ssim2` | - | `<10.0..99.9>` | none | Binary-search encoder quality to hit perceptual SSIM2 threshold |
| `--max-bytes` | - | `<size>` | none | Upper file-size bound for encoder quality search |
| `--chroma-subsampling` | - | `<mode>` | `444` | JPEG chroma layout: `auto`, `444` (`4:4:4`), `422` (`4:2:2`), `420` (`4:2:0`) |
| `--ico-sizes` | - | `<n[,n...]>` | `16,32,48,64,128,256` | Icon frame sizes (`16..256`) synthesized for `.ico` output |
| `--meta` | - | `<policy>` | `keep` | Metadata handling: `keep` (`all`), `strip-privacy` (`privacy`), `strip-all` (`strip`, `none`) |
| `--keep-dates` | - | - | `false` | Copy source `atime` and `mtime` to output files |
| `--verify` | - | - | `false` | Re-decode every emitted image after write to verify structural integrity |
| `--stdout` | - | - | `false` | Write single encoded output directly to standard output |

### `--name` Template Placeholders

| Token | Expands to |
| --- | --- |
| `{root}` | Discovered input root path |
| `{dir}` | Relative parent directory under root (empty if at root) |
| `{path}` | Relative path without extension |
| `{stem}` | Input filename stem |
| `{ext}` | Resolved output file extension (without dot) |
| `{w}`, `{h}` | Output pixel width and height |
| `{algo}` | Canonical slug of the selected `--algo` filter |
| `{orientation}` | Output orientation: `landscape`, `portrait`, or `square` |
| `{screen}` | Matched output name (or `default`) |

### Discovery, Filtering, Archives, and Automation

| Flag | Short | Argument | Default | Description |
| --- | --- | --- | --- | --- |
| `--recursive` | `-r` | - | `false` | Traverse input directories recursively |
| `--manifest` | - | `<file>` | none | Read newline-delimited input paths from `<file>` (also supports `@file` positional syntax) |
| `--include` | - | `<glob>` | none | Include only paths matching glob pattern (repeatable) |
| `--exclude` | - | `<glob>` | none | Exclude paths matching glob pattern (repeatable) |
| `--only-orientation` | - | `<kind>` | none | Filter inputs by aspect: `landscape` (`h`), `portrait` (`v`), `square` (`sq`) |
| `--min-res` | - | `<WxH>` | none | Skip inputs smaller than width or height |
| `--max-res` | - | `<WxH>` | none | Skip inputs larger than width or height |
| `--skip-if-fits` | - | - | `false` | Skip processing if source already satisfies resolved target |
| `--link-skipped` | - | - | `false` | Hardlink or reflink skipped inputs into destination directory |
| `--copy-skipped` | - | - | `false` | Copy skipped inputs into destination directory |
| `--on-error` | - | `<policy>` | `skip` | Batch/archive entry error handling: `skip`, `copy-raw` (`copy`), `fail` |
| `--strict` | - | - | `false` | Enable strict failure (`--on-error fail`) |
| `--fail-on` | - | `<level>` | `error` | Minimum severity triggering failure: `error`, `truncated`, `warn` |
| `--archive-mode` | - | `<mode>` | `repack` | Archive workflow: `repack`, `extract`, `flatten` |
| `--archive-format` | - | `<fmt>` | `auto` | Output archive container: `auto`, `zip` (`cbz`), `tar`, `tar.gz` (`tgz`), `tar.zst` (`tzst`), `tar.xz` (`txz`), `tar.bz2` (`tbz2`) |
| `--archive-depth` | - | `<N>` | `4` | Maximum nested archive recursion depth |
| `--non-images` | - | `<policy>` | `keep` | Non-image archive entries: `keep`, `drop`, `extract` |
| `--dedup` | - | `[mode]` | `none` (`perceptual` if bare) | Skip duplicate images: `none` (`off`), `exact`, `perceptual` (`dhash`, Hamming `5`), or integer `<N>` Hamming threshold |
| `--max-entry-size` | - | `<size>` | `512MiB` | Max decompressed bytes per archive entry |
| `--max-total-size` | - | `<size>` | `4GiB` | Max aggregate decompressed bytes per archive job |
| `--max-pixels` | - | `<N>` | `268435456` | Decode pixel ceiling per image (`width * height`) |
| `--max-ratio` | - | `<f64>` | `100.0` | Maximum allowed archive compression ratio |
| `--threads` | `-j` | `<N>` | CPU / cgroup quota | Worker thread pool size (`-jN` inline form also accepted) |
| `--memory-limit` | - | `<size>` | half available RAM | Working arena memory budget |
| `--resume` | - | `[path]` | none (`.resume.journal` if bare) | Track completed inputs in crash-safe journal to skip finished entries on rerun |
| `--watch` | - | - | `false` | Enter `inotify` watch loop on input directories |
| `--iters` | - | `<N>` | `3` | Iteration count per kernel for `bench` |
| `--report` | - | `<text \| json>` | `text` | Summary report output format |
| `--json` | - | - | `false` | Emit JSON summary / inspect / compare / bench output |
| `--preview` | - | - | `false` | Display rendered terminal preview alongside normal processing |
| `--preview-only` | - | - | `false` | Display terminal preview on `stderr` without writing output files |
| `--preview-proto` | - | `<proto>` | `auto` | Preview protocol: `auto`, `kitty`, `sixel`, `sextant` (`unicode`), `half` (`half-block`) |
| `--quiet` | - | - | `false` | Suppress progress bar and standard summaries |
| `--verbose` | - | - | `false` | Print per-item telemetry and security diagnostics |

---

## Configuration and Environment

- Presets file: `$XDG_CONFIG_HOME/aperture/presets.conf` (fallback `$HOME/.config/aperture/presets.conf`), formatted as one `name = <flags...>` entry per line.
- Display/DE environment variables read on Linux: `WAYLAND_DISPLAY`, `DISPLAY`, `XDG_CURRENT_DESKTOP`, `HYPRLAND_INSTANCE_SIGNATURE`, `SWAYSOCK`.
- Terminal preview detection variables: `TERM`, `TERM_PROGRAM`, `KITTY_WINDOW_ID`, `GHOSTTY_RESOURCES_DIR`.

## Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | All operations completed successfully |
| `1` | Fatal I/O, decode, encode, or system backend error |
| `2` | Invalid flag, malformed target spec, or invalid CLI invocation |
| `3` | Partial batch failure (one or more batch/archive items failed) |
| `4` | Archive security budget/path violation or `--verify` integrity mismatch |
