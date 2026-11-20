//! CPU backend: Rust port of the offline Inworld Voice Isolator network (inference only).
//!
//! A band-split dual-path network over the 48 kHz STFT: one token per band and
//! frame, blocks of time/band attention and bidirectional Mamba, per-band
//! decoder heads that predict a bounded complex mask plus a bounded residual.
//! Matrix products go through candle (gemm); everything else is in \`fast\`.

use crate::config::{band_edges, read_config, Config, Denoiser};
use crate::dsp::Stft;
use crate::fast;
use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::ops::softmax_last_dim;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

struct Weights(HashMap<String, Tensor>);

impl Weights {
    fn take(&mut self, name: &str) -> Result<Tensor> {
        self.0.remove(name).with_context(|| format!("checkpoint is missing tensor '{name}'"))
    }
}

fn host(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
}

struct Lin {
    wt: Tensor, // [in, out]
    b: Option<Tensor>,
    inp: usize,
    out: usize,
}

impl Lin {
    fn from(weight: Tensor, b: Option<Tensor>) -> Result<Self> {
        let (out, inp) = weight.dims2()?;
        Ok(Self { wt: weight.t()?.contiguous()?, b, inp, out })
    }
    fn load(w: &mut Weights, prefix: &str, bias: bool) -> Result<Self> {
        let weight = w.take(&format!("{prefix}.weight"))?;
        let b = if bias { Some(w.take(&format!("{prefix}.bias"))?) } else { None };
        Self::from(weight, b)
    }
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        let mut shape = x.dims().to_vec();
        let y = x.reshape(((), self.inp))?.matmul(&self.wt)?;
        let y = match &self.b {
            Some(b) => fast::bias_add(&y, b)?,
            None => y,
        };
        *shape.last_mut().unwrap() = self.out;
        Ok(y.reshape(shape)?)
    }
}

struct Ln {
    w: Tensor,
    b: Tensor,
}

impl Ln {
    fn load(w: &mut Weights, prefix: &str) -> Result<Self> {
        Ok(Self { w: w.take(&format!("{prefix}.weight"))?, b: w.take(&format!("{prefix}.bias"))? })
    }
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        fast::layer_norm(&x.contiguous()?, &self.w, &self.b)
    }
}

fn gelu(x: &Tensor) -> Result<Tensor> {
    fast::gelu(&x.contiguous()?)
}

fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    fast::add(&a.contiguous()?, &b.contiguous()?)
}

/// Bands per attention call, bounding the [G, H, T, T] score tensor.
const ATTN_GROUP: usize = 8;

struct Rope {
    cos: Vec<f32>, // [T, hd/2]
    sin: Vec<f32>,
}

impl Rope {
    fn new(frames: usize, head_dim: usize) -> Self {
        let half = head_dim / 2;
        let inv: Vec<f32> = (0..half).map(|i| 10000f32.powf(-(i as f32) / half as f32)).collect();
        let mut cos = Vec::with_capacity(frames * half);
        let mut sin = Vec::with_capacity(frames * half);
        for p in 0..frames {
            for i in 0..half {
                let a = p as f32 * inv[i];
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        Self { cos, sin }
    }
}

/// Multi-head self attention over the middle axis of x [G, T, W].
fn mha(x: &Tensor, qkv: &Lin, proj: &Lin, heads: usize, rope: Option<&Rope>, excl: bool) -> Result<Tensor> {
    let (g, _, w) = x.dims3()?;
    let scale = (1.0 / ((w / heads) as f64).sqrt()) as f32;
    let qkv = qkv.fwd(x)?;
    let (q, kt, v) = fast::split_qkv(&qkv, heads, scale, rope.map(|r| (r.cos.as_slice(), r.sin.as_slice())))?;
    let mut outs = Vec::new();
    let mut s = 0;
    while s < g {
        let n = ATTN_GROUP.min(g - s);
        let (qs, kts, vs) = (q.narrow(0, s, n)?, kt.narrow(0, s, n)?, v.narrow(0, s, n)?);
        let probs = softmax_last_dim(&qs.matmul(&kts)?)?;
        let o = probs.matmul(&vs)?;
        let (oc, vc) = (o.contiguous()?, vs.contiguous()?);
        outs.push(fast::merge_heads(&oc, if excl { Some(&vc) } else { None })?);
        s += n;
    }
    let o = if outs.len() == 1 { outs.remove(0) } else { Tensor::cat(&outs, 0)? };
    proj.fwd(&o)
}

struct TimeBlock {
    norm1: Ln,
    qkv: Lin,
    proj: Lin,
    norm2: Ln,
    ffn0: Lin,
    ffn3: Lin,
    excl: bool,
}

struct BandBlock {
    norm1: Ln,
    in_proj: Lin,
    out_proj: Lin,
    linear1: Lin,
    linear2: Lin,
    norm2: Ln,
}

struct Mamba {
    in_proj: Lin,
    conv_w: Vec<f32>, // [inner, kernel]
    conv_b: Vec<f32>,
    x_proj: Lin,
    dt_proj: Lin,
    a: Vec<f32>, // [inner, state], already -exp(A_log)
    d: Vec<f32>,
    out_proj: Lin,
    inner: usize,
    state: usize,
    dt_rank: usize,
    kernel: usize,
}

struct BiMamba {
    fwd: Mamba,
    bwd: Mamba,
    out_b: Tensor,
    w1: Tensor, // output projection split into the two halves: [W, W] each
    w2: Tensor,
    norm: Ln,
}

enum Block {
    Attn { time: TimeBlock, band: BandBlock },
    Mamba { time: BiMamba, freq: BiMamba },
}

struct Head {
    ln: Ln,
    l1: Lin,
    l2: Lin,
}

pub struct Model {
    pub cfg: Config,
    edges: Vec<(usize, usize)>,
    encoders: Vec<(Ln, Lin)>,
    band_embedding: Tensor, // [bands, 1, W]
    blocks: Vec<Block>,
    cond_scale: Tensor,
    cond_shift: Tensor,
    heads: Vec<Head>,
    stft: Stft,
}

impl Mamba {
    fn load(w: &mut Weights, prefix: &str, cfg: &Config) -> Result<Self> {
        let inner = cfg.mamba_expand * cfg.width;
        Ok(Self {
            in_proj: Lin::load(w, &format!("{prefix}.in_proj"), false)?,
            conv_w: host(&w.take(&format!("{prefix}.conv1d.weight"))?)?,
            conv_b: host(&w.take(&format!("{prefix}.conv1d.bias"))?)?,
            x_proj: Lin::load(w, &format!("{prefix}.x_proj"), false)?,
            dt_proj: Lin::load(w, &format!("{prefix}.dt_proj"), true)?,
            a: host(&w.take(&format!("{prefix}.A_log"))?)?.iter().map(|v| -v.exp()).collect(),
            d: host(&w.take(&format!("{prefix}.D"))?)?,
            out_proj: Lin::load(w, &format!("{prefix}.out_proj"), false)?,
            inner,
            state: cfg.mamba_state,
            dt_rank: cfg.width.div_ceil(16),
            kernel: cfg.mamba_conv,
        })
    }

    /// x: [B, T, W] -> [B, T, W]; \`rev\` runs the time-reversed (backward) scan without flipping.
    fn fwd(&self, x: &Tensor, rev: bool) -> Result<Tensor> {
        let xz = self.in_proj.fwd(x)?;
        let u = fast::conv_silu(&xz, self.inner, &self.conv_w, &self.conv_b, self.kernel, rev)?;
        let proj = self.x_proj.fwd(&u)?;
        let dt = proj.narrow(2, 0, self.dt_rank)?.contiguous()?;
        let pre = self.dt_proj.fwd(&dt)?;
        let y = fast::scan(&u, &pre, &xz, &proj, &self.a, &self.d, self.dt_rank, self.state, rev)?;
        self.out_proj.fwd(&y)
    }
}

impl BiMamba {
    fn load(w: &mut Weights, prefix: &str, cfg: &Config) -> Result<Self> {
        let weight = w.take(&format!("{prefix}.output_proj.weight"))?; // [W, 2W]
        let wt = weight.t()?.contiguous()?; // [2W, W]
        let wd = cfg.width;
        Ok(Self {
            fwd: Mamba::load(w, &format!("{prefix}.forward_blocks"), cfg)?,
            bwd: Mamba::load(w, &format!("{prefix}.backward_blocks"), cfg)?,
            out_b: w.take(&format!("{prefix}.output_proj.bias"))?,
            w1: wt.narrow(0, 0, wd)?.contiguous()?,
            w2: wt.narrow(0, wd, wd)?.contiguous()?,
            norm: Ln::load(w, &format!("{prefix}.norm"))?,
        })
    }

    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        let (bsz, t, wd) = x.dims3()?;
        let f = fast::add(&self.fwd.fwd(x, false)?, x)?;
        let b = fast::add(&self.bwd.fwd(x, true)?, x)?;
        let y = f.reshape(((), wd))?.matmul(&self.w1)?.add(&b.reshape(((), wd))?.matmul(&self.w2)?)?;
        let y = fast::bias_add(&y, &self.out_b)?.reshape((bsz, t, wd))?;
        self.norm.fwd(&y)
    }
}

impl TimeBlock {
    fn fwd(&self, x: &Tensor, heads: usize, rope: &Rope) -> Result<Tensor> {
        let a = mha(&self.norm1.fwd(x)?, &self.qkv, &self.proj, heads, Some(rope), self.excl)?;
        let x = add(x, &a)?;
        let f = self.ffn3.fwd(&gelu(&self.ffn0.fwd(&self.norm2.fwd(&x)?)?)?)?;
        add(&x, &f)
    }
}

impl BandBlock {
    fn fwd(&self, x: &Tensor, heads: usize) -> Result<Tensor> {
        let a = mha(&self.norm1.fwd(x)?, &self.in_proj, &self.out_proj, heads, None, false)?;
        let x = add(x, &a)?;
        let f = self.linear2.fwd(&gelu(&self.linear1.fwd(&self.norm2.fwd(&x)?)?)?)?;
        add(&x, &f)
    }
}

impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        let cfg = read_config(path)?;
        let bytes = std::fs::read(path)?;
        let mut w = Weights(candle_core::safetensors::load_buffer(&bytes, &Device::Cpu)?);
        drop(bytes);
        let edges = band_edges(&cfg);
        let encoders = (0..edges.len())
            .map(|i| Ok((Ln::load(&mut w, &format!("encoders.{i}.0"))?, Lin::load(&mut w, &format!("encoders.{i}.1"), true)?)))
            .collect::<Result<Vec<_>>>()?;
        let band_embedding = w.take("band_embedding")?.unsqueeze(1)?;
        let mut blocks = Vec::new();
        for (i, kind) in cfg.block_types.iter().enumerate() {
            let p = format!("blocks.{i}");
            blocks.push(match kind.as_str() {
                "transformer" => Block::Attn {
                    time: TimeBlock {
                        norm1: Ln::load(&mut w, &format!("{p}.time_transformer.norm1"))?,
                        qkv: Lin::load(&mut w, &format!("{p}.time_transformer.attn.qkv"), true)?,
                        proj: Lin::load(&mut w, &format!("{p}.time_transformer.attn.proj"), true)?,
                        norm2: Ln::load(&mut w, &format!("{p}.time_transformer.norm2"))?,
                        ffn0: Lin::load(&mut w, &format!("{p}.time_transformer.ffn.0"), true)?,
                        ffn3: Lin::load(&mut w, &format!("{p}.time_transformer.ffn.3"), true)?,
                        excl: cfg.exclusive_self_attention,
                    },
                    band: BandBlock {
                        norm1: Ln::load(&mut w, &format!("{p}.band_transformer.norm1"))?,
                        in_proj: {
                            let n = format!("{p}.band_transformer.self_attn.in_proj");
                            Lin::from(w.take(&format!("{n}_weight"))?, Some(w.take(&format!("{n}_bias"))?))?
                        },
                        out_proj: Lin::load(&mut w, &format!("{p}.band_transformer.self_attn.out_proj"), true)?,
                        linear1: Lin::load(&mut w, &format!("{p}.band_transformer.linear1"), true)?,
                        linear2: Lin::load(&mut w, &format!("{p}.band_transformer.linear2"), true)?,
                        norm2: Ln::load(&mut w, &format!("{p}.band_transformer.norm2"))?,
                    },
                },
                "mamba" => Block::Mamba {
                    time: BiMamba::load(&mut w, &format!("{p}.time_mamba"), &cfg)?,
                    freq: BiMamba::load(&mut w, &format!("{p}.freq_mamba"), &cfg)?,
                },
                other => bail!("unknown block type {other}"),
            });
        }
        // The decoder's conditioning input is the constant [1, 0].
        let cw = w.take("denoise_decoder.condition.weight")?;
        let cb = w.take("denoise_decoder.condition.bias")?;
        let cond = cw.narrow(1, 0, 1)?.squeeze(1)?.add(&cb)?;
        let cond_scale = cond.narrow(0, 0, cfg.width)?.contiguous()?;
        let cond_shift = cond.narrow(0, cfg.width, cfg.width)?.contiguous()?;
        let heads = (0..edges.len())
            .map(|i| {
                Ok(Head {
                    ln: Ln::load(&mut w, &format!("denoise_decoder.heads.{i}.0"))?,
                    l1: Lin::load(&mut w, &format!("denoise_decoder.heads.{i}.1"), true)?,
                    l2: Lin::load(&mut w, &format!("denoise_decoder.heads.{i}.3"), true)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let stft = Stft::new(cfg.n_fft, cfg.hop_length);
        Ok(Self { cfg, edges, encoders, band_embedding, blocks, cond_scale, cond_shift, heads, stft })
    }

    fn run_impl(&self, audio: &[f32]) -> Result<Vec<f32>> {
        const MIN: usize = 4096;
        if audio.len() < MIN {
            let mut padded = audio.to_vec();
            padded.resize(MIN, 0.0);
            let mut y = self.run_impl(&padded)?;
            y.truncate(audio.len());
            return Ok(y);
        }
        let cfg = &self.cfg;
        let dev = Device::Cpu;
        let frames = self.stft.num_frames(audio.len());
        let bins = self.stft.bins;
        let spec = self.stft.forward(audio);
        let spec_t = Tensor::from_vec(spec.clone(), (frames, bins, 2), &dev)?;

        // Encode: one token per band and frame.
        let mut toks = Vec::with_capacity(self.edges.len());
        for ((s, e), (ln, lin)) in self.edges.iter().zip(&self.encoders) {
            let comp = spec_t.narrow(1, *s, e - s)?.contiguous()?.reshape((frames, 2 * (e - s)))?;
            toks.push(lin.fwd(&ln.fwd(&comp)?)?);
        }
        let mut x = Tensor::stack(&toks, 0)?.broadcast_add(&self.band_embedding)?;
        drop(toks);

        let rope = Rope::new(frames, cfg.width / cfg.num_heads);
        for block in &self.blocks {
            x = match block {
                Block::Attn { time, band } => {
                    let x = time.fwd(&x, cfg.num_heads, &rope)?;
                    let x = band.fwd(&fast::transpose01(&x)?, cfg.num_heads)?;
                    fast::transpose01(&x)?
                }
                Block::Mamba { time, freq } => {
                    let x = add(&x, &time.fwd(&x)?)?;
                    let f = fast::transpose01(&x)?;
                    let f = add(&f, &freq.fwd(&f)?)?;
                    fast::transpose01(&f)?
                }
            };
        }

        // Decode: per-band heads give a complex mask and a residual per bin.
        let x = x.broadcast_mul(&(&self.cond_scale + 1.0)?)?.broadcast_add(&self.cond_shift)?;
        let mut outs = Vec::with_capacity(self.heads.len());
        for (i, ((s, e), h)) in self.edges.iter().zip(&self.heads).enumerate() {
            let f = x.narrow(0, i, 1)?.squeeze(0)?;
            let o = h.l2.fwd(&gelu(&h.l1.fwd(&h.ln.fwd(&f)?)?)?)?;
            outs.push(o.reshape((frames, e - s, 4))?);
        }
        let raw = host(&Tensor::cat(&outs, 1)?)?; // [frames, bins, 4]

        // Bounded complex mask plus bounded residual, applied to the spectrum.
        let bound = cfg.correction_bound;
        let mut corrected = vec![0f32; spec.len()];
        corrected.par_chunks_mut(2).zip(spec.par_chunks(2)).zip(raw.par_chunks(4)).for_each(|((out, s), r)| {
            let (mr, mi) = (2.0 * r[0].tanh(), 2.0 * r[1].tanh());
            let (cr, ci) = (r[2].tanh(), r[3].tanh());
            let norm = (cr * cr + ci * ci).sqrt().max(1.0);
            let mag = (s[0] * s[0] + s[1] * s[1]).sqrt();
            let k = bound * mag / norm;
            out[0] = s[0] * mr - s[1] * mi + cr * k;
            out[1] = s[0] * mi + s[1] * mr + ci * k;
        });
        let y = self.stft.inverse(&corrected, audio.len());
        Ok(audio.iter().zip(y).map(|(a, b)| a + b).collect())
    }
}

impl Denoiser for Model {
    fn cfg(&self) -> &Config {
        &self.cfg
    }
    fn run(&self, audio: &[f32]) -> Result<Vec<f32>> {
        self.run_impl(audio)
    }
}
