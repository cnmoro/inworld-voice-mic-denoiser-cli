//! Locates (and downloads on first use) the offline Voice Isolator checkpoint.

use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const URL: &str = "https://huggingface.co/inworld/voice-isolator/resolve/9b0285b3fe8403aa8e2f1dbb2165a80a4294c970/denoiser.safetensors";
const SHA256: &str = "647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53";

fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 22];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

pub fn resolve(custom: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = custom {
        if !p.exists() {
            bail!("model file {} does not exist", p.display());
        }
        return Ok(p.to_path_buf());
    }
    let dir = dirs::cache_dir().context("no cache directory available")?.join("inworldvoice");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("voice-isolator-v1-denoiser.safetensors");
    if path.exists() {
        if sha256_file(&path)? == SHA256 {
            return Ok(path);
        }
        eprintln!("cached model failed its checksum, downloading again");
    }
    eprintln!("Downloading the Inworld Voice Isolator weights (~256 MB, Apache-2.0) to {}", path.display());
    let resp = ureq::get(URL).call().context("downloading the model failed")?;
    let total: u64 = resp.header("Content-Length").and_then(|v| v.parse().ok()).unwrap_or(0);
    let bar = ProgressBar::new(total);
    bar.set_style(ProgressStyle::with_template("{bar:40} {bytes}/{total_bytes} ({eta})")?);
    let part = path.with_extension("part");
    let mut out = std::fs::File::create(&part)?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        bar.inc(n as u64);
    }
    out.flush()?;
    drop(out);
    bar.finish_and_clear();
    if sha256_file(&part)? != SHA256 {
        let _ = std::fs::remove_file(&part);
        bail!("downloaded model failed its sha256 check");
    }
    std::fs::rename(&part, &path)?;
    Ok(path)
}
