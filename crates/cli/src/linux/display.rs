use engine::geometry::fit::MonitorRegion;
use std::env;
use std::fs;
use std::path::Path;
use wayland_client::protocol::{wl_output, wl_registry};
use wayland_client::{Connection, Dispatch, QueueHandle, WEnum};
use wayland_protocols::xdg::xdg_output::zv1::client::{zxdg_output_manager_v1, zxdg_output_v1};
use x11rb::connection::Connection as X11Connection;
use x11rb::protocol::randr;

#[derive(Debug, Clone)]
pub struct ConnectedMonitor {
    pub name: String,
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub scale: f32,
    pub primary: bool,
}

impl ConnectedMonitor {
    pub fn to_region(&self) -> MonitorRegion {
        MonitorRegion {
            name: self.name.clone(),
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
        }
    }
}

pub fn detect_monitors() -> Vec<ConnectedMonitor> {
    if env::var_os("WAYLAND_DISPLAY").is_some() {
        let wl = detect_wayland_monitors();
        if !wl.is_empty() {
            return wl;
        }
    }

    if env::var_os("DISPLAY").is_some() {
        let x11 = detect_x11_monitors();
        if !x11.is_empty() {
            return x11;
        }
    }

    let drm = detect_drm_sysfs_monitors();
    if !drm.is_empty() {
        return drm;
    }

    Vec::new()
}

struct WaylandOutputState {
    registry_name: u32,
    wl_out: wl_output::WlOutput,
    name: String,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    logical_x: Option<i32>,
    logical_y: Option<i32>,
    logical_w: Option<i32>,
    logical_h: Option<i32>,
    scale: i32,
}

struct WaylandProbeState {
    xdg_mgr: Option<zxdg_output_manager_v1::ZxdgOutputManagerV1>,
    outputs: Vec<WaylandOutputState>,
    xdg_outputs: Vec<zxdg_output_v1::ZxdgOutputV1>,
}

impl Dispatch<wl_registry::WlRegistry, ()> for WaylandProbeState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            if interface == "wl_output" {
                let idx = state.outputs.len();
                let wl_out = registry.bind::<wl_output::WlOutput, usize, Self>(
                    name,
                    version.min(4),
                    qh,
                    idx,
                );
                state.outputs.push(WaylandOutputState {
                    registry_name: name,
                    wl_out,
                    name: format!("OUT-{}", idx + 1),
                    x: 0,
                    y: 0,
                    width: 0,
                    height: 0,
                    logical_x: None,
                    logical_y: None,
                    logical_w: None,
                    logical_h: None,
                    scale: 1,
                });
            } else if interface == "zxdg_output_manager_v1" && state.xdg_mgr.is_none() {
                let mgr = registry.bind::<zxdg_output_manager_v1::ZxdgOutputManagerV1, (), Self>(
                    name,
                    version.min(3),
                    qh,
                    (),
                );
                state.xdg_mgr = Some(mgr);
            }
        }
    }
}

impl Dispatch<wl_output::WlOutput, usize> for WaylandProbeState {
    fn event(
        state: &mut Self,
        _: &wl_output::WlOutput,
        event: wl_output::Event,
        idx: &usize,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let Some(entry) = state.outputs.get_mut(*idx) else {
            return;
        };
        match event {
            wl_output::Event::Geometry { x, y, .. } => {
                entry.x = x;
                entry.y = y;
            }
            wl_output::Event::Mode {
                flags,
                width,
                height,
                ..
            } => {
                if let WEnum::Value(mode_flags) = flags
                    && mode_flags.contains(wl_output::Mode::Current)
                {
                    entry.width = width;
                    entry.height = height;
                } else if entry.width == 0 && entry.height == 0 {
                    entry.width = width;
                    entry.height = height;
                }
            }
            wl_output::Event::Scale { factor } => {
                entry.scale = factor.max(1);
            }
            wl_output::Event::Name { name } if !name.is_empty() => {
                entry.name = name;
            }
            _ => {}
        }
    }
}

impl Dispatch<zxdg_output_manager_v1::ZxdgOutputManagerV1, ()> for WaylandProbeState {
    fn event(
        _: &mut Self,
        _: &zxdg_output_manager_v1::ZxdgOutputManagerV1,
        _: zxdg_output_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zxdg_output_v1::ZxdgOutputV1, usize> for WaylandProbeState {
    fn event(
        state: &mut Self,
        _: &zxdg_output_v1::ZxdgOutputV1,
        event: zxdg_output_v1::Event,
        idx: &usize,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let Some(entry) = state.outputs.get_mut(*idx) else {
            return;
        };
        match event {
            zxdg_output_v1::Event::LogicalPosition { x, y } => {
                entry.logical_x = Some(x);
                entry.logical_y = Some(y);
            }
            zxdg_output_v1::Event::LogicalSize { width, height } => {
                entry.logical_w = Some(width);
                entry.logical_h = Some(height);
            }
            zxdg_output_v1::Event::Name { name } if !name.is_empty() => {
                entry.name = name;
            }
            _ => {}
        }
    }
}

fn detect_wayland_monitors() -> Vec<ConnectedMonitor> {
    let Ok(conn) = Connection::connect_to_env() else {
        return Vec::new();
    };
    let mut queue = conn.new_event_queue::<WaylandProbeState>();
    let qh = queue.handle();
    let display = conn.display();
    let _registry = display.get_registry(&qh, ());

    let mut state = WaylandProbeState {
        xdg_mgr: None,
        outputs: Vec::new(),
        xdg_outputs: Vec::new(),
    };

    if queue.roundtrip(&mut state).is_err() {
        return Vec::new();
    }

    if let Some(ref mgr) = state.xdg_mgr {
        for (idx, o) in state.outputs.iter().enumerate() {
            let xdg_out = mgr.get_xdg_output(&o.wl_out, &qh, idx);
            state.xdg_outputs.push(xdg_out);
        }
        let _ = queue.roundtrip(&mut state);
    }

    let _ = queue.roundtrip(&mut state);

    state
        .outputs
        .into_iter()
        .enumerate()
        .filter_map(|(idx, o)| {
            let _ = o.registry_name;
            let w = o.logical_w.unwrap_or(o.width).max(0) as u32;
            let h = o.logical_h.unwrap_or(o.height).max(0) as u32;
            if w == 0 || h == 0 {
                return None;
            }
            Some(ConnectedMonitor {
                name: o.name,
                x: o.logical_x.unwrap_or(o.x),
                y: o.logical_y.unwrap_or(o.y),
                width: w,
                height: h,
                scale: o.scale as f32,
                primary: idx == 0,
            })
        })
        .collect()
}

fn detect_x11_monitors() -> Vec<ConnectedMonitor> {
    let Ok((conn, screen_num)) = x11rb::connect(None) else {
        return Vec::new();
    };
    let Some(screen) = conn.setup().roots.get(screen_num) else {
        return Vec::new();
    };
    let root = screen.root;
    let Ok(res_cookie) = randr::get_screen_resources_current(&conn, root) else {
        return Vec::new();
    };
    let Ok(res) = res_cookie.reply() else {
        return Vec::new();
    };
    let primary_out = randr::get_output_primary(&conn, root)
        .ok()
        .and_then(|c| c.reply().ok())
        .map(|r| r.output);

    let mut monitors = Vec::new();
    for out in res.outputs {
        let Ok(info_cookie) = randr::get_output_info(&conn, out, res.config_timestamp) else {
            continue;
        };
        let Ok(info) = info_cookie.reply() else {
            continue;
        };
        if info.connection != randr::Connection::CONNECTED || info.crtc == 0 {
            continue;
        }
        let Ok(crtc_cookie) = randr::get_crtc_info(&conn, info.crtc, res.config_timestamp) else {
            continue;
        };
        let Ok(crtc) = crtc_cookie.reply() else {
            continue;
        };
        if crtc.width == 0 || crtc.height == 0 {
            continue;
        }
        let name = String::from_utf8_lossy(&info.name).into_owned();
        monitors.push(ConnectedMonitor {
            name,
            x: crtc.x as i32,
            y: crtc.y as i32,
            width: crtc.width as u32,
            height: crtc.height as u32,
            scale: 1.0,
            primary: Some(out) == primary_out,
        });
    }

    if monitors.is_empty() && screen.width_in_pixels > 0 && screen.height_in_pixels > 0 {
        monitors.push(ConnectedMonitor {
            name: "X11-0".to_string(),
            x: 0,
            y: 0,
            width: screen.width_in_pixels as u32,
            height: screen.height_in_pixels as u32,
            scale: 1.0,
            primary: true,
        });
    }

    monitors
}

fn detect_drm_sysfs_monitors() -> Vec<ConnectedMonitor> {
    let drm_dir = Path::new("/sys/class/drm");
    let Ok(rd) = fs::read_dir(drm_dir) else {
        return Vec::new();
    };

    let mut entries: Vec<_> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();

    let mut monitors = Vec::new();
    let mut cur_x = 0i32;

    for path in entries {
        let fname = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if !fname.starts_with("card") || !fname.contains('-') {
            continue;
        }
        let status_path = path.join("status");
        let Ok(status) = fs::read_to_string(status_path) else {
            continue;
        };
        if status.trim() != "connected" {
            continue;
        }
        let modes_path = path.join("modes");
        let Ok(modes) = fs::read_to_string(modes_path) else {
            continue;
        };
        let Some(first_mode) = modes.lines().next() else {
            continue;
        };
        let clean = first_mode
            .trim()
            .split(|c: char| !c.is_ascii_digit() && c != 'x')
            .next()
            .unwrap_or("");
        if let Some((ws, hs)) = clean.split_once('x')
            && let (Ok(w), Ok(h)) = (ws.parse::<u32>(), hs.parse::<u32>())
            && w > 0
            && h > 0
        {
            let short_name = fname
                .split_once('-')
                .map_or(fname, |(_, rest)| rest)
                .to_string();
            monitors.push(ConnectedMonitor {
                name: short_name,
                x: cur_x,
                y: 0,
                width: w,
                height: h,
                scale: 1.0,
                primary: monitors.is_empty(),
            });
            cur_x += w as i32;
        }
    }

    monitors
}
