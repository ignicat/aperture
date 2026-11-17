use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

pub fn preset_file_path(program: &str) -> Option<PathBuf> {
    if let Ok(xdg) = env::var("XDG_CONFIG_HOME")
        && !xdg.is_empty()
    {
        return Some(PathBuf::from(xdg).join(program).join("presets.conf"));
    }
    if let Ok(home) = env::var("HOME")
        && !home.is_empty()
    {
        return Some(
            PathBuf::from(home)
                .join(".config")
                .join(program)
                .join("presets.conf"),
        );
    }
    None
}

pub fn builtin_presets() -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    map.insert(
        "wallpaper-4k".to_string(),
        "3840x2160 --fit auto --algo fsr --deband 35 --dither blue-noise".to_string(),
    );
    map.insert(
        "ultrawide".to_string(),
        "3440x1440 --fit smart-crop --algo fsr --dither blue-noise".to_string(),
    );
    map.insert(
        "retina-2x".to_string(),
        "2x --algo ewa-lanczossharp --lossless --optimize".to_string(),
    );
    map.insert(
        "pixel-2x".to_string(),
        "2x --algo mmpx --lossless --optimize".to_string(),
    );
    map.insert(
        "anime-4k".to_string(),
        "4k --algo lineart --deband 50 --dither blue-noise".to_string(),
    );
    map.insert(
        "web-opt".to_string(),
        "--optimize --target-ssim2 86 --meta strip-privacy".to_string(),
    );
    map.insert(
        "oled-fit".to_string(),
        "--screen auto --fit contain --bg-color #000000 --dither blue-noise".to_string(),
    );
    map
}

pub fn load_presets(program: &str) -> BTreeMap<String, String> {
    let mut map = builtin_presets();
    if let Some(path) = preset_file_path(program)
        && let Ok(text) = fs::read_to_string(path)
    {
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some((name, val)) = trimmed.split_once('=') {
                let k = name.trim().to_string();
                let v = val.trim().to_string();
                if !k.is_empty() && !v.is_empty() {
                    map.insert(k, v);
                }
            }
        }
    }
    map
}

pub fn save_preset(program: &str, name: &str, flags_line: &str) -> Result<PathBuf, String> {
    let path = preset_file_path(program)
        .ok_or_else(|| "unable to resolve XDG config directory".to_string())?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("failed to create config dir: {e}"))?;
    }
    let mut map = BTreeMap::new();
    if let Ok(existing) = fs::read_to_string(&path) {
        for line in existing.lines() {
            let trimmed = line.trim();
            if let Some((k, v)) = trimmed.split_once('=') {
                map.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
    }
    map.insert(name.trim().to_string(), flags_line.trim().to_string());
    let mut out = String::new();
    for (k, v) in map {
        out.push_str(&format!("{k} = {v}\n"));
    }
    fs::write(&path, out).map_err(|e| format!("failed to write preset file: {e}"))?;
    Ok(path)
}

pub fn delete_preset(program: &str, name: &str) -> Result<bool, String> {
    let path = preset_file_path(program)
        .ok_or_else(|| "unable to resolve XDG config directory".to_string())?;
    let Ok(existing) = fs::read_to_string(&path) else {
        return Ok(false);
    };
    let mut map = BTreeMap::new();
    for line in existing.lines() {
        let trimmed = line.trim();
        if let Some((k, v)) = trimmed.split_once('=') {
            map.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    let removed = map.remove(name.trim()).is_some();
    let mut out = String::new();
    for (k, v) in map {
        out.push_str(&format!("{k} = {v}\n"));
    }
    fs::write(&path, out).map_err(|e| format!("failed to update preset file: {e}"))?;
    Ok(removed)
}

pub fn split_shell_tokens(line: &str) -> Result<Vec<String>, String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut has_token = false;

    for ch in line.chars() {
        if escaped {
            cur.push(ch);
            escaped = false;
            has_token = true;
            continue;
        }
        if ch == '\\' && !in_single {
            escaped = true;
            has_token = true;
            continue;
        }
        if ch == '\'' && !in_double {
            in_single = !in_single;
            has_token = true;
            continue;
        }
        if ch == '"' && !in_single {
            in_double = !in_double;
            has_token = true;
            continue;
        }
        if ch.is_ascii_whitespace() && !in_single && !in_double {
            if has_token {
                tokens.push(std::mem::take(&mut cur));
                has_token = false;
            }
            continue;
        }
        cur.push(ch);
        has_token = true;
    }

    if in_single || in_double {
        return Err("unterminated quote in preset definition".to_string());
    }
    if has_token {
        tokens.push(cur);
    }
    Ok(tokens)
}
