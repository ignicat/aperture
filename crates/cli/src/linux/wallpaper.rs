use engine::codec::{DecodeOptions, decode_image};
use engine::pixel::PixelFormat;
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;
use x11rb::connection::Connection;
use x11rb::protocol::xproto::{AtomEnum, ChangeWindowAttributesAux, CloseDown, ConnectionExt, CreateGCAux, ImageFormat, PropMode};
use x11rb::wrapper::ConnectionExt as WrapperExt;

pub fn set_wallpaper_for_output(image_path: &Path, output_name: Option<&str>) -> Result<String, String> {
    let abs_path = fs::canonicalize(image_path).unwrap_or_else(|_| image_path.to_path_buf());
    let path_str = abs_path.to_string_lossy().into_owned();

    if env::var_os("WAYLAND_DISPLAY").is_some() {
        if command_exists("awww") {
            let mut cmd = Command::new("awww");
            cmd.arg("img").arg(&path_str);
            if let Some(out) = output_name {
                cmd.arg("--outputs").arg(out);
            }
            if run_ok(cmd) {
                return Ok("awww".to_string());
            }
        }
        if command_exists("swww") {
            let mut cmd = Command::new("swww");
            cmd.arg("img").arg(&path_str);
            if let Some(out) = output_name {
                cmd.arg("--outputs").arg(out);
            }
            if run_ok(cmd) {
                return Ok("swww".to_string());
            }
        }
        if env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some() && command_exists("hyprctl") {
            let mut preload = Command::new("hyprctl");
            preload.args(["hyprpaper", "preload", &path_str]);
            let _ = preload.output();
            let target = output_name.unwrap_or("");
            let spec = format!("{target},{path_str}");
            let mut wp = Command::new("hyprctl");
            wp.args(["hyprpaper", "wallpaper", &spec]);
            if run_ok(wp) {
                return Ok("hyprpaper".to_string());
            }
        }
        if env::var_os("SWAYSOCK").is_some() && command_exists("swaymsg") {
            let out = output_name.unwrap_or("*");
            let mut cmd = Command::new("swaymsg");
            cmd.args(["output", out, "bg", &path_str, "fill"]);
            if run_ok(cmd) {
                return Ok("sway".to_string());
            }
        }
    }

    let desktop = env::var("XDG_CURRENT_DESKTOP")
        .unwrap_or_default()
        .to_ascii_lowercase();

    if desktop.contains("kde") && command_exists("plasma-apply-wallpaperimage") {
        let mut cmd = Command::new("plasma-apply-wallpaperimage");
        cmd.arg(&path_str);
        if run_ok(cmd) {
            return Ok("plasma".to_string());
        }
    }

    if (desktop.contains("gnome") || desktop.contains("cinnamon") || desktop.contains("mate"))
        && command_exists("gsettings")
    {
        let uri = format!("file://{path_str}");
        let mut cmd1 = Command::new("gsettings");
        cmd1.args(["set", "org.gnome.desktop.background", "picture-uri", &uri]);
        let mut cmd2 = Command::new("gsettings");
        cmd2.args(["set", "org.gnome.desktop.background", "picture-uri-dark", &uri]);
        if run_ok(cmd1) {
            let _ = cmd2.output();
            return Ok("gsettings".to_string());
        }
    }

    if desktop.contains("xfce") && command_exists("xfconf-query") {
        let mut cmd = Command::new("xfconf-query");
        cmd.args([
            "-c",
            "xfce4-desktop",
            "-p",
            "/backdrop/screen0/monitor0/workspace0/last-image",
            "-s",
            &path_str,
        ]);
        if run_ok(cmd) {
            return Ok("xfce".to_string());
        }
    }

    if env::var_os("DISPLAY").is_some() {
        set_x11_root_pixmap(&abs_path)?;
        return Ok("x11-root-pixmap".to_string());
    }

    Err("no active Wayland/DE wallpaper IPC daemon or X11 display found to apply wallpaper".to_string())
}

fn set_x11_root_pixmap(path: &Path) -> Result<(), String> {
    let raw = fs::read(path).map_err(|e| format!("failed reading wallpaper {}: {e}", path.display()))?;
    let ext = path.extension().and_then(|s| s.to_str());
    let decoded = decode_image(
        &raw,
        ext,
        &DecodeOptions {
            auto_orient: true,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())?;

    let w = decoded.image.width;
    let h = decoded.image.height;
    let bpp_in = decoded.image.format.bytes_per_pixel();
    let mut bgrx = vec![0u8; (w as usize) * (h as usize) * 4];

    match decoded.image.format {
        PixelFormat::Rgb8 => {
            for (src, dst) in decoded
                .image
                .data
                .as_chunks::<3>()
                .0
                .iter()
                .zip(bgrx.as_chunks_mut::<4>().0.iter_mut())
            {
                *dst = [src[2], src[1], src[0], 255];
            }
        }
        PixelFormat::Rgba8 => {
            for (src, dst) in decoded
                .image
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .zip(bgrx.as_chunks_mut::<4>().0.iter_mut())
            {
                *dst = [src[2], src[1], src[0], src[3]];
            }
        }
        _ => {
            let img_data = &decoded.image.data;
            for i in 0..(w as usize * h as usize) {
                let s = i * bpp_in;
                let r = img_data.get(s).copied().unwrap_or(0);
                let g = img_data.get(s + 1).copied().unwrap_or(r);
                let b = img_data.get(s + 2).copied().unwrap_or(r);
                let d = i * 4;
                bgrx[d] = b;
                bgrx[d + 1] = g;
                bgrx[d + 2] = r;
                bgrx[d + 3] = 255;
            }
        }
    }

    let (conn, screen_num) = x11rb::connect(None).map_err(|e| format!("X11 connect failed: {e}"))?;
    let screen = conn
        .setup()
        .roots
        .get(screen_num)
        .ok_or_else(|| "missing X11 root screen".to_string())?;
    let root = screen.root;
    let depth = screen.root_depth;

    let atom_xroot = conn
        .intern_atom(false, b"_XROOTPMAP_ID")
        .map_err(|e| e.to_string())?
        .reply()
        .map_err(|e| e.to_string())?
        .atom;
    let atom_eset = conn
        .intern_atom(false, b"ESETROOT_PMAP_ID")
        .map_err(|e| e.to_string())?
        .reply()
        .map_err(|e| e.to_string())?
        .atom;

    let pixmap = conn.generate_id().map_err(|e| e.to_string())?;
    let gc = conn.generate_id().map_err(|e| e.to_string())?;

    conn.create_pixmap(depth, pixmap, root, w as u16, h as u16)
        .map_err(|e| e.to_string())?;
    conn.create_gc(gc, pixmap, &CreateGCAux::new())
        .map_err(|e| e.to_string())?;

    let strip_h = 32usize;
    let row_bytes = (w as usize) * 4;
    let mut y = 0usize;
    while y < h as usize {
        let rows_here = (h as usize - y).min(strip_h);
        let start = y * row_bytes;
        let end = start + rows_here * row_bytes;
        conn.put_image(
            ImageFormat::Z_PIXMAP,
            pixmap,
            gc,
            w as u16,
            rows_here as u16,
            0,
            y as i16,
            0,
            depth,
            &bgrx[start..end],
        )
        .map_err(|e| e.to_string())?;
        y += rows_here;
    }

    conn.change_property32(PropMode::REPLACE, root, atom_xroot, AtomEnum::PIXMAP, &[pixmap])
        .map_err(|e| e.to_string())?;
    conn.change_property32(PropMode::REPLACE, root, atom_eset, AtomEnum::PIXMAP, &[pixmap])
        .map_err(|e| e.to_string())?;
    conn.change_window_attributes(root, &ChangeWindowAttributesAux::new().background_pixmap(pixmap))
        .map_err(|e| e.to_string())?;
    conn.clear_area(false, root, 0, 0, 0, 0)
        .map_err(|e| e.to_string())?;
    conn.set_close_down_mode(CloseDown::RETAIN_PERMANENT)
        .map_err(|e| e.to_string())?;
    conn.sync().map_err(|e| e.to_string())?;

    Ok(())
}

fn command_exists(name: &str) -> bool {
    let Some(path_env) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&path_env).any(|dir| dir.join(name).is_file())
}

fn run_ok(mut cmd: Command) -> bool {
    cmd.output().is_ok_and(|o| o.status.success())
}
