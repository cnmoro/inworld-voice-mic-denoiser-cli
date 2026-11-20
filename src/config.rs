//! Checkpoint configuration shared by the CPU and GPU backends.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

fn one() -> usize {
    1
}

#[derive(Deserialize, Clone, Debug)]
pub struct Config {
    pub width: usize,
    pub block_types: Vec<String>,
    pub num_heads: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub sample_rate: usize,
    pub band_plan: Vec<[usize; 2]>,
    pub correction_bound: f32,
    pub chunk_seconds: f64,
    pub overlap_seconds: f64,
    pub exclusive_self_attention: bool,
    pub exclusive_axes: String,
    pub mamba_state: usize,
    pub mamba_conv: usize,
    pub mamba_expand: usize,
    #[serde(default)]
    pub causal: bool,
    #[serde(default)]
    pub dual_mode: bool,
    #[serde(default)]
    pub mode_film: bool,
    #[serde(default)]
    pub lookahead_frames_choices: Vec<usize>,
    #[serde(default = "one")]
    pub offline_passes: usize,
}

impl Config {
    pub fn chunk_samples(&self) -> usize {
        (self.chunk_seconds * self.sample_rate as f64) as usize
    }
    pub fn overlap_samples(&self) -> usize {
        ((self.overlap_seconds * self.sample_rate as f64) as usize).min(self.chunk_samples() / 2)
    }
}

pub fn band_edges(c: &Config) -> Vec<(usize, usize)> {
    let n_bins = c.n_fft / 2 + 1;
    let hz_per_bin = c.sample_rate as f64 / c.n_fft as f64;
    let mut edges = Vec::new();
    let mut start = 0usize;
    for [width_hz, upto_hz] in &c.band_plan {
        let width = ((*width_hz as f64 / hz_per_bin).round_ties_even() as usize).max(1);
        let upto = n_bins.min((*upto_hz as f64 / hz_per_bin).round_ties_even() as usize);
        while start < upto {
            let end = (start + width).min(upto);
            edges.push((start, end));
            start = end;
        }
    }
    if start < n_bins {
        edges.push((start, n_bins));
    }
    edges
}

pub fn read_config(path: &Path) -> Result<Config> {
        let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut header)?;
    let v: serde_json::Value = serde_json::from_slice(&header)?;
    let raw = v["__metadata__"]["config"]
        .as_str()
        .context("safetensors metadata has no 'config' entry")?;
    let cfg: Config = serde_json::from_str(raw)?;
    if cfg.causal
        || cfg.dual_mode
        || cfg.mode_film
        || !cfg.lookahead_frames_choices.is_empty()
        || cfg.exclusive_axes != "time"
        || cfg.offline_passes != 1
    {
        bail!("this checkpoint variant is not supported; use the offline voice-isolator-v1 weights");
    }
    Ok(cfg)
}


/// A tensor of a .safetensors file, converted to f32.
pub struct RawTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Minimal safetensors reader (F32 tensors only).
pub fn read_safetensors(path: &Path) -> Result<HashMap<String, RawTensor>> {
    let bytes = std::fs::read(path)?;
    let hlen = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + hlen])?;
    let body = &bytes[8 + hlen..];
    let mut out = HashMap::new();
    for (name, v) in header.as_object().context("bad safetensors header")? {
        if name == "__metadata__" {
            continue;
        }
        if v["dtype"] != "F32" {
            bail!("tensor {name} is not F32");
        }
        let shape: Vec<usize> = v["shape"].as_array().context("shape")?.iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let (s, e) = (v["data_offsets"][0].as_u64().unwrap() as usize, v["data_offsets"][1].as_u64().unwrap() as usize);
        let data = body[s..e].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        out.insert(name.clone(), RawTensor { shape, data });
    }
    Ok(out)
}

/// What the chunked renderer needs from a backend.
pub trait Denoiser {
    fn cfg(&self) -> &Config;
    /// Runs the network over one chunk of mono 48 kHz audio.
    fn run(&self, audio: &[f32]) -> Result<Vec<f32>>;
}

#[cfg(test)]
pub fn test_config(sample_rate: usize, chunk_seconds: f64, overlap_seconds: f64) -> Config {
    serde_json::from_value(serde_json::json!({
        "width": 512, "block_types": ["transformer"], "num_heads": 8, "n_fft": 2048, "hop_length": 480,
        "sample_rate": sample_rate, "band_plan": [[100, 1000], [250, 4000], [500, 8000], [1000, 16000], [2000, 24000]],
        "correction_bound": 0.5, "chunk_seconds": chunk_seconds, "overlap_seconds": overlap_seconds,
        "exclusive_self_attention": true, "exclusive_axes": "time", "mamba_state": 16, "mamba_conv": 4, "mamba_expand": 2
    }))
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_cover_every_bin_once() {
        let cfg = test_config(48000, 8.0, 2.0);
        let edges = band_edges(&cfg);
        assert_eq!(edges.len(), 46, "the released checkpoint has 46 bands");
        assert_eq!(edges[0].0, 0);
        assert_eq!(edges.last().unwrap().1, 1025);
        for w in edges.windows(2) {
            assert_eq!(w[0].1, w[1].0);
        }
    }
}
