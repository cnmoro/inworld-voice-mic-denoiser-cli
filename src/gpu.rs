//! GPU backend: the whole network on wgpu (Vulkan / DX12 / Metal) with hand-written WGSL kernels.

use crate::config::{band_edges, read_config, read_safetensors, Config, Denoiser, RawTensor};
use crate::dsp::Stft;
use crate::kernels as k;
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferUsages, ComputePass};

struct Lin {
    w: Buffer, // [out, in]
    b: Option<Buffer>,
    out: usize,
    inp: usize,
}

struct Ln {
    w: Buffer,
    b: Buffer,
}

struct Enc {
    ln: Ln,
    w: Buffer,
    bias: Buffer, // linear bias + band embedding
    cols: usize,
}

struct TimeW {
    norm1: Ln,
    qkv: Lin,
    proj: Lin,
    norm2: Ln,
    ffn0: Lin,
    ffn3: Lin,
}

struct BandW {
    norm1: Ln,
    in_proj: Lin,
    out_proj: Lin,
    linear1: Lin,
    linear2: Lin,
    norm2: Ln,
}

struct MambaW {
    in_proj: Lin,
    conv_w: Buffer,
    conv_b: Buffer,
    x_proj: Lin,
    dt_proj: Lin,
    a: Buffer, // -exp(A_log) [inner, 16]
    d: Buffer,
    out_proj: Lin,
}

struct BiW {
    fwd: MambaW,
    bwd: MambaW,
    out_w: Buffer,    // [W, 2W]
    out_b: Buffer,
    sum_w: Buffer,    // W1 + W2, [W, W]
    norm: Ln,
}

enum GBlock {
    Attn(TimeW, BandW),
    Mamba(BiW, BiW),
}

struct Head {
    ln: Ln,
    l1: Lin,
    l2: Lin,
}

struct Prof {
    qs: wgpu::QuerySet,
    cap: u32,
    next: std::cell::Cell<u32>,
    names: std::cell::RefCell<Vec<&'static str>>,
}

pub struct GpuModel {
    scratch_cache: std::cell::RefCell<Option<Scratch>>,
    prof: Option<Prof>,
    cfg: Config,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipes: HashMap<&'static str, wgpu::ComputePipeline>,
    dummy: Buffer,
    edges: Vec<(usize, usize)>,
    encs: Vec<Enc>,
    blocks: Vec<GBlock>,
    cond_scale: Buffer,
    cond_shift: Buffer,
    heads: Vec<Head>,
    stft: Stft,
}

/// Mapping of tokens (rows) to (group, sequence position): row = g*gs + t*ts.
#[derive(Clone, Copy)]
struct Map {
    g: u32,
    t: u32,
    gs: u32,
    ts: u32,
}

struct Scratch {
    frames: usize,
    spec: Buffer,
    x: Buffer,
    h: Buffer,
    mo: Buffer,
    qkv: Buffer,
    big: Buffer,
    q: Buffer,
    kk: Buffer,
    v: Buffer,
    o: Buffer,
    scores: Buffer,
    score_group: usize,
    u: Buffer,
    pre: Buffer,
    y: Buffer,
    proj: Buffer,
    cos: Buffer,
    sin: Buffer,
    raw: Buffer,
    lnbuf: Buffer,
}

fn groups1d(n: usize, per: usize) -> [u32; 3] {
    let g = n.div_ceil(per).max(1);
    [g.min(65535) as u32, g.div_ceil(65535) as u32, 1]
}

fn host_tensor<'a>(w: &'a HashMap<String, RawTensor>, name: &str) -> Result<&'a RawTensor> {
    w.get(name).with_context(|| format!("checkpoint is missing tensor '{name}'"))
}

pub fn list_adapters() -> Vec<wgpu::Adapter> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let mut v: Vec<wgpu::Adapter> = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::PRIMARY))
        .into_iter()
        .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
        .collect();
    let rank = |a: &wgpu::Adapter| match a.get_info().device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        _ => 2,
    };
    v.sort_by_key(rank);
    v
}

impl GpuModel {
    pub fn new(path: &Path, index: usize) -> Result<(Self, String)> {
        let cfg = read_config(path)?;
        let adapters = list_adapters();
        let adapter = adapters.into_iter().nth(index).with_context(|| {
            format!("no GPU with index {index} found (need a Vulkan, DirectX 12 or Metal capable GPU and driver)")
        })?;
        let info = adapter.get_info();
        let name = format!("{} ({:?})", info.name, info.backend);
        let limits = adapter.limits();
        let want_prof = std::env::var_os("INWORLD_PROFILE").is_some();
        let prof_feats = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        let want_prof = want_prof && adapter.features().contains(prof_feats);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("inworldvoice"),
            required_features: if want_prof { prof_feats } else { wgpu::Features::empty() },
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .context("could not open the GPU")?;
        // Largest buffer we bind: FFN hidden / Mamba xz = tokens * 4 * width floats.
        let need = 50_000usize * 4 * cfg.width * 4;
        if (limits.max_storage_buffer_binding_size as usize) < need || (limits.max_buffer_size as usize) < need {
            bail!("this GPU's buffer limits are too small for the model (need ~{} MB per buffer)", need >> 20);
        }

        let mut pipes = HashMap::new();
        for (name, src) in [
            ("matmul", k::MATMUL),
            ("layernorm", k::LAYERNORM),
            ("softmax", k::SOFTMAX),
            ("qkv_split", k::QKV_SPLIT),
            ("merge", k::MERGE),
            ("add", k::ADD),
            ("conv_silu", k::CONV_SILU),
            ("scan", k::SCAN),
        ] {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(format!("{}{}", k::PRELUDE, src))),
            });
            let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            pipes.insert(name, pipe);
        }

        let raw = read_safetensors(path)?;
        let up = |data: &[f32]| -> Buffer {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            })
        };
        let get = |n: &str| -> Result<&RawTensor> { host_tensor(&raw, n) };
        let ln = |p: &str| -> Result<Ln> {
            Ok(Ln { w: up(&get(&format!("{p}.weight"))?.data), b: up(&get(&format!("{p}.bias"))?.data) })
        };
        let lin = |p: &str, bias: bool| -> Result<Lin> {
            let w = get(&format!("{p}.weight"))?;
            Ok(Lin {
                w: up(&w.data),
                b: if bias { Some(up(&get(&format!("{p}.bias"))?.data)) } else { None },
                out: w.shape[0],
                inp: w.shape[1],
            })
        };
        let mamba = |p: &str| -> Result<MambaW> {
            let a: Vec<f32> = get(&format!("{p}.A_log"))?.data.iter().map(|v| -v.exp()).collect();
            Ok(MambaW {
                in_proj: lin(&format!("{p}.in_proj"), false)?,
                conv_w: up(&get(&format!("{p}.conv1d.weight"))?.data),
                conv_b: up(&get(&format!("{p}.conv1d.bias"))?.data),
                x_proj: lin(&format!("{p}.x_proj"), false)?,
                dt_proj: lin(&format!("{p}.dt_proj"), true)?,
                a: up(&a),
                d: up(&get(&format!("{p}.D"))?.data),
                out_proj: lin(&format!("{p}.out_proj"), false)?,
            })
        };
        let bimamba = |p: &str| -> Result<BiW> {
            let w = get(&format!("{p}.output_proj.weight"))?;
            let wd = cfg.width;
            let mut sum = vec![0f32; wd * wd];
            for r in 0..wd {
                for c in 0..wd {
                    sum[r * wd + c] = w.data[r * 2 * wd + c] + w.data[r * 2 * wd + wd + c];
                }
            }
            Ok(BiW {
                fwd: mamba(&format!("{p}.forward_blocks"))?,
                bwd: mamba(&format!("{p}.backward_blocks"))?,
                out_w: up(&w.data),
                out_b: up(&get(&format!("{p}.output_proj.bias"))?.data),
                sum_w: up(&sum),
                norm: ln(&format!("{p}.norm"))?,
            })
        };

        let edges = band_edges(&cfg);
        let emb = get("band_embedding")?;
        let mut encs = Vec::new();
        for (i, (s, e)) in edges.iter().enumerate() {
            let wl = get(&format!("encoders.{i}.1.weight"))?;
            let bl = get(&format!("encoders.{i}.1.bias"))?;
            let total: Vec<f32> = (0..cfg.width).map(|j| bl.data[j] + emb.data[i * cfg.width + j]).collect();
            encs.push(Enc { ln: ln(&format!("encoders.{i}.0"))?, w: up(&wl.data), bias: up(&total), cols: 2 * (e - s) });
        }
        let mut blocks = Vec::new();
        for (i, kind) in cfg.block_types.iter().enumerate() {
            let p = format!("blocks.{i}");
            blocks.push(match kind.as_str() {
                "transformer" => GBlock::Attn(
                    TimeW {
                        norm1: ln(&format!("{p}.time_transformer.norm1"))?,
                        qkv: lin(&format!("{p}.time_transformer.attn.qkv"), true)?,
                        proj: lin(&format!("{p}.time_transformer.attn.proj"), true)?,
                        norm2: ln(&format!("{p}.time_transformer.norm2"))?,
                        ffn0: lin(&format!("{p}.time_transformer.ffn.0"), true)?,
                        ffn3: lin(&format!("{p}.time_transformer.ffn.3"), true)?,
                    },
                    BandW {
                        norm1: ln(&format!("{p}.band_transformer.norm1"))?,
                        in_proj: {
                            let w = get(&format!("{p}.band_transformer.self_attn.in_proj_weight"))?;
                            Lin {
                                w: up(&w.data),
                                b: Some(up(&get(&format!("{p}.band_transformer.self_attn.in_proj_bias"))?.data)),
                                out: w.shape[0],
                                inp: w.shape[1],
                            }
                        },
                        out_proj: lin(&format!("{p}.band_transformer.self_attn.out_proj"), true)?,
                        linear1: lin(&format!("{p}.band_transformer.linear1"), true)?,
                        linear2: lin(&format!("{p}.band_transformer.linear2"), true)?,
                        norm2: ln(&format!("{p}.band_transformer.norm2"))?,
                    },
                ),
                _ => GBlock::Mamba(bimamba(&format!("{p}.time_mamba"))?, bimamba(&format!("{p}.freq_mamba"))?),
            });
        }
        let cw = get("denoise_decoder.condition.weight")?;
        let cb = get("denoise_decoder.condition.bias")?;
        let wd = cfg.width;
        let cond: Vec<f32> = (0..2 * wd).map(|j| cw.data[j * 2] + cb.data[j]).collect();
        let heads = (0..edges.len())
            .map(|i| {
                Ok(Head {
                    ln: ln(&format!("denoise_decoder.heads.{i}.0"))?,
                    l1: lin(&format!("denoise_decoder.heads.{i}.1"), true)?,
                    l2: lin(&format!("denoise_decoder.heads.{i}.3"), true)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let stft = Stft::new(cfg.n_fft, cfg.hop_length);
        let dummy = up(&[0f32; 4]);
        let cond_scale = up(&cond[..wd]);
        let cond_shift = up(&cond[wd..]);
        let prof = if want_prof {
            let cap = 4096u32;
            Some(Prof {
                qs: device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: cap }),
                cap,
                next: std::cell::Cell::new(0),
                names: std::cell::RefCell::new(Vec::new()),
            })
        } else {
            None
        };
        Ok((Self { scratch_cache: std::cell::RefCell::new(None), prof, cfg, device, queue, pipes, dummy, edges, encs, blocks, cond_scale, cond_shift, heads, stft }, name))
    }

    fn storage(&self, floats: usize) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (floats.max(4) * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn scratch(&self, frames: usize) -> Scratch {
        let cfg = &self.cfg;
        let (b, w, h) = (self.edges.len(), cfg.width, cfg.num_heads);
        let r = b * frames;
        let inner = cfg.mamba_expand * w;
        // Scores for a group of bands, capped near 128 MB.
        let per_band = h * frames * frames;
        let score_group = ((32usize << 20) / per_band.max(1)).clamp(1, b);
        let band_scores = frames * h * b * b;
        let scores_len = (score_group * per_band).max(band_scores);
        let bins = self.stft.bins;
        let half = w / h / 2;
        let rope = |f: &dyn Fn(f32) -> f32| -> Vec<f32> {
            let inv: Vec<f32> = (0..half).map(|i| 10000f32.powf(-(i as f32) / half as f32)).collect();
            (0..frames * half).map(|j| f((j / half) as f32 * inv[j % half])).collect()
        };
        let up = |d: &[f32]| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(d),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            })
        };
        Scratch {
            frames,
            spec: self.storage(frames * bins * 2),
            x: self.storage(r * w),
            h: self.storage(r * w),
            mo: self.storage(r * w),
            qkv: self.storage(r * 3 * w),
            big: self.storage(r * 4 * w),
            q: self.storage(r * w),
            kk: self.storage(r * w),
            v: self.storage(r * w),
            o: self.storage(r * w),
            scores: self.storage(scores_len),
            score_group,
            u: self.storage(r * inner),
            pre: self.storage(r * inner),
            y: self.storage(r * inner),
            proj: self.storage(r * 64),
            cos: up(&rope(&|a| a.cos())),
            sin: up(&rope(&|a| a.sin())),
            raw: self.storage(frames * bins * 4),
            lnbuf: self.storage(frames * 2 * 400),
        }
    }

    fn dispatch(&self, pass: &mut ComputePass<'_>, name: &'static str, bufs: &[&Buffer], params: &[u32], groups: [u32; 3]) {
        let pipe = &self.pipes[name];
        let pbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(params),
            usage: BufferUsages::STORAGE,
        });
        let layout = pipe.get_bind_group_layout(0);
        let mut entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        entries.push(wgpu::BindGroupEntry { binding: 7, resource: pbuf.as_entire_binding() });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bg, &[]);
        let slot = self.prof.as_ref().and_then(|p| {
            let i = p.next.get();
            (i + 2 <= p.cap).then_some((p, i))
        });
        if let Some((p, i)) = slot {
            pass.write_timestamp(&p.qs, i);
        }
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        if let Some((p, i)) = slot {
            pass.write_timestamp(&p.qs, i + 1);
            p.next.set(i + 2);
            p.names.borrow_mut().push(name);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mm(
        &self, pass: &mut ComputePass<'_>, a: &Buffer, b: &Buffer, c: &Buffer, bias: Option<&Buffer>, m: usize, n: usize, kdim: usize,
        a_off: usize, a_ld: usize, a_bs: usize, b_off: usize, b_ld: usize, b_bs: usize, b_trans: bool, c_off: usize, c_ld: usize,
        c_bs: usize, batch: usize, accum: bool, gelu: bool,
    ) {
        let flags = bias.is_some() as u32 | ((accum as u32) << 1);
        let p = [
            m as u32, n as u32, kdim as u32, flags, a_off as u32, a_ld as u32, a_bs as u32, b_off as u32, b_ld as u32, b_bs as u32,
            b_trans as u32, c_off as u32, c_ld as u32, c_bs as u32, 1.0f32.to_bits(), gelu as u32,
        ];
        let bias = bias.unwrap_or(&self.dummy);
        self.dispatch(pass, "matmul", &[a, b, c, bias], &p, [n.div_ceil(64) as u32, m.div_ceil(64) as u32, batch as u32]);
    }

    /// out[rows, lin.out] (+)= x[rows, lin.inp] @ W^T (+ bias)
    fn linear(&self, pass: &mut ComputePass<'_>, x: &Buffer, lin: &Lin, out: &Buffer, rows: usize, accum: bool, gelu: bool) {
        self.mm(
            pass, x, &lin.w, out, lin.b.as_ref(), rows, lin.out, lin.inp, 0, lin.inp, 0, 0, lin.inp, 0, true, 0, lin.out, 0, 1, accum, gelu,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn layernorm(
        &self, pass: &mut ComputePass<'_>, inp: &Buffer, out: &Buffer, ln: &Ln, rows: usize, cols: usize, in_off: usize, in_ld: usize,
        out_off: usize, out_ld: usize, pre: Option<(&Buffer, &Buffer)>,
    ) {
        let (ps, pt) = pre.unwrap_or((&self.dummy, &self.dummy));
        let p = [rows as u32, cols as u32, in_off as u32, in_ld as u32, out_off as u32, out_ld as u32, pre.is_some() as u32, 1e-5f32.to_bits()];
        self.dispatch(pass, "layernorm", &[inp, out, &ln.w, &ln.b, ps, pt], &p, [rows.min(65535) as u32, rows.div_ceil(65535) as u32, 1]);
    }

    fn add(&self, pass: &mut ComputePass<'_>, a: &Buffer, b: &Buffer, n: usize) {
        self.dispatch(pass, "add", &[a, b], &[n as u32], groups1d(n, 256));
    }

    /// Multi-head attention over the sequence axis of the token map; result rows go to \`out\`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self, pass: &mut ComputePass<'_>, s: &Scratch, qkv_in: &Buffer, out: &Buffer, map: Map, rope: bool, excl: bool,
    ) {
        let (h, w) = (self.cfg.num_heads, self.cfg.width);
        let hd = w / h;
        let t = map.t as usize;
        let g_total = map.g as usize;
        let scale = 1.0 / (hd as f32).sqrt();
        let p = [map.g, map.t, h as u32, hd as u32, map.gs, map.ts, w as u32, rope as u32, scale.to_bits()];
        self.dispatch(pass, "qkv_split", &[qkv_in, &s.q, &s.kk, &s.v, &s.cos, &s.sin], &p, groups1d(g_total * h * t * hd / 2, 256));
        let group = if rope { s.score_group } else { (g_total).min(((32usize << 20) / (h * t * t).max(1)).max(1)) };
        let mut g0 = 0;
        while g0 < g_total {
            let n = group.min(g_total - g0);
            let batch = n * h;
            let off = g0 * h * t * hd;
            self.mm(pass, &s.q, &s.kk, &s.scores, None, t, t, hd, off, hd, t * hd, off, hd, t * hd, true, 0, t, t * t, batch, false, false);
            self.dispatch(pass, "softmax", &[&s.scores], &[(batch * t) as u32, t as u32], [(batch * t).min(65535) as u32, (batch * t).div_ceil(65535) as u32, 1]);
            self.mm(pass, &s.scores, &s.v, &s.o, None, t, hd, t, 0, t, t * t, off, hd, t * hd, false, off, hd, t * hd, batch, false, false);
            g0 += n;
        }
        let p = [map.g, map.t, h as u32, hd as u32, map.gs, map.ts, w as u32, excl as u32];
        self.dispatch(pass, "merge", &[&s.o, &s.v, out], &p, groups1d(g_total * t * h, 256));
    }

    fn mamba_dir(&self, pass: &mut ComputePass<'_>, s: &Scratch, mw: &MambaW, x: &Buffer, out: &Buffer, map: Map, rev: bool, rows: usize) {
        let inner = self.cfg.mamba_expand * self.cfg.width;
        let kernel = self.cfg.mamba_conv;
        let dt_rank = self.cfg.width.div_ceil(16);
        let n = self.cfg.mamba_state;
        self.linear(pass, x, &mw.in_proj, &s.big, rows, false, false); // xz [rows, 2*inner]
        let p = [map.g, map.t, inner as u32, map.gs, map.ts, rev as u32, kernel as u32];
        self.dispatch(pass, "conv_silu", &[&s.big, &s.u, &mw.conv_w, &mw.conv_b], &p, groups1d(rows * inner, 256));
        self.linear(pass, &s.u, &mw.x_proj, &s.proj, rows, false, false); // [rows, dt_rank + 2n]
        let pld = dt_rank + 2 * n;
        self.mm(pass, &s.proj, &mw.dt_proj.w, &s.pre, mw.dt_proj.b.as_ref(), rows, inner, dt_rank, 0, pld, 0, 0, dt_rank, 0, true, 0, inner, 0, 1, false, false);
        let p = [map.g, map.t, inner as u32, map.gs, map.ts, rev as u32, pld as u32, dt_rank as u32];
        self.dispatch(pass, "scan", &[&s.u, &s.pre, &s.big, &s.proj, &mw.a, &mw.d, &s.y], &p, groups1d(map.g as usize * inner, 64));
        self.linear(pass, &s.y, &mw.out_proj, out, rows, false, false);
    }

    /// x [rows, W] -> mo = LayerNorm(output_proj([fwd + x, bwd + x]))
    fn bimamba(&self, pass: &mut ComputePass<'_>, s: &Scratch, bw: &BiW, map: Map, rows: usize) {
        let wd = self.cfg.width;
        self.mamba_dir(pass, s, &bw.fwd, &s.x, &s.q, map, false, rows);
        self.mamba_dir(pass, s, &bw.bwd, &s.x, &s.kk, map, true, rows);
        // C = fwd @ W[:, :W]^T + bias ; C += bwd @ W[:, W:]^T ; C += x @ (W1 + W2)^T   (the "+ x" terms)
        self.mm(pass, &s.q, &bw.out_w, &s.h, Some(&bw.out_b), rows, wd, wd, 0, wd, 0, 0, 2 * wd, 0, true, 0, wd, 0, 1, false, false);
        self.mm(pass, &s.kk, &bw.out_w, &s.h, None, rows, wd, wd, 0, wd, 0, wd, 2 * wd, 0, true, 0, wd, 0, 1, true, false);
        self.mm(pass, &s.x, &bw.sum_w, &s.h, None, rows, wd, wd, 0, wd, 0, 0, wd, 0, true, 0, wd, 0, 1, true, false);
        self.layernorm(pass, &s.h, &s.mo, &bw.norm, rows, wd, 0, wd, 0, wd, None);
        self.add(pass, &s.x, &s.mo, rows * wd);
    }

    fn run_gpu(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let frames = self.stft.num_frames(audio.len());
        let bins = self.stft.bins;
        let (nb, wd) = (self.edges.len(), self.cfg.width);
        let rows = nb * frames;
        let mut cache = self.scratch_cache.borrow_mut();
        if cache.as_ref().map(|c| c.frames) != Some(frames) {
            *cache = None; // free the old buffers before allocating new ones
            *cache = Some(self.scratch(frames));
        }
        let s = cache.as_ref().unwrap();
        let spec = self.stft.forward(audio);
        self.queue.write_buffer(&s.spec, 0, bytemuck::cast_slice(&spec));

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            let pass = &mut pass;
            // Encoders: LayerNorm + Linear (+ band embedding in the bias) per band.
            for (i, ((st, _), e)) in self.edges.iter().zip(&self.encs).enumerate() {
                self.layernorm(pass, &s.spec, &s.lnbuf, &e.ln, frames, e.cols, st * 2, bins * 2, 0, e.cols, None);
                self.mm(pass, &s.lnbuf, &e.w, &s.x, Some(&e.bias), frames, wd, e.cols, 0, e.cols, 0, 0, e.cols, 0, true, i * frames * wd, wd, 0, 1, false, false);
            }
            let time_map = Map { g: nb as u32, t: frames as u32, gs: frames as u32, ts: 1 };
            let band_map = Map { g: frames as u32, t: nb as u32, gs: 1, ts: frames as u32 };
            for block in &self.blocks {
                match block {
                    GBlock::Attn(tw, bw) => {
                        self.layernorm(pass, &s.x, &s.h, &tw.norm1, rows, wd, 0, wd, 0, wd, None);
                        self.linear(pass, &s.h, &tw.qkv, &s.qkv, rows, false, false);
                        self.attention(pass, &s, &s.qkv, &s.h, time_map, true, true);
                        self.linear(pass, &s.h, &tw.proj, &s.x, rows, true, false);
                        self.layernorm(pass, &s.x, &s.h, &tw.norm2, rows, wd, 0, wd, 0, wd, None);
                        self.linear(pass, &s.h, &tw.ffn0, &s.big, rows, false, true);
                        self.linear(pass, &s.big, &tw.ffn3, &s.x, rows, true, false);

                        self.layernorm(pass, &s.x, &s.h, &bw.norm1, rows, wd, 0, wd, 0, wd, None);
                        self.linear(pass, &s.h, &bw.in_proj, &s.qkv, rows, false, false);
                        self.attention(pass, &s, &s.qkv, &s.h, band_map, false, false);
                        self.linear(pass, &s.h, &bw.out_proj, &s.x, rows, true, false);
                        self.layernorm(pass, &s.x, &s.h, &bw.norm2, rows, wd, 0, wd, 0, wd, None);
                        self.linear(pass, &s.h, &bw.linear1, &s.big, rows, false, true);
                        self.linear(pass, &s.big, &bw.linear2, &s.x, rows, true, false);
                    }
                    GBlock::Mamba(time, freq) => {
                        self.bimamba(pass, &s, time, time_map, rows);
                        self.bimamba(pass, &s, freq, band_map, rows);
                    }
                }
            }
            // Decoder heads.
            for (i, ((st, en), hd)) in self.edges.iter().zip(&self.heads).enumerate() {
                self.layernorm(pass, &s.x, &s.h, &hd.ln, frames, wd, i * frames * wd, wd, 0, wd, Some((&self.cond_scale, &self.cond_shift)));
                self.linear(pass, &s.h, &hd.l1, &s.mo, frames, false, true);
                let out_w = 4 * (en - st);
                self.mm(pass, &s.mo, &hd.l2.w, &s.raw, hd.l2.b.as_ref(), frames, out_w, wd, 0, wd, 0, 0, wd, 0, true, st * 4, bins * 4, 0, 1, false, false);
            }
        }
        let size = (frames * bins * 4 * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&s.raw, 0, &staging, 0, size);
        let prof_buf = self.prof.as_ref().map(|p| {
            let n = p.next.get();
            let rb = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (p.cap as u64) * 8,
                usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let st = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (p.cap as u64) * 8,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            enc.resolve_query_set(&p.qs, 0..n, &rb, 0);
            enc.copy_buffer_to_buffer(&rb, 0, &st, 0, (p.cap as u64) * 8);
            st
        });
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).context("GPU wait failed")?;
        rx.recv()??;
        let raw: Vec<f32> = {
            let data = slice.get_mapped_range()?;
            bytemuck::cast_slice::<u8, f32>(&data).to_vec()
        };
        staging.unmap();
        if let (Some(p), Some(st)) = (&self.prof, &prof_buf) {
            let sl = st.slice(..);
            let (tx2, rx2) = std::sync::mpsc::channel();
            sl.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx2.send(r);
            });
            self.device.poll(wgpu::PollType::wait_indefinitely())?;
            rx2.recv()??;
            let data = sl.get_mapped_range()?;
            let ts: &[u64] = bytemuck::cast_slice(&data);
            let period = self.queue.get_timestamp_period() as f64;
            let mut acc: HashMap<&str, (f64, usize)> = HashMap::new();
            for (i, name) in p.names.borrow().iter().enumerate() {
                let e = acc.entry(name).or_insert((0.0, 0));
                e.0 += (ts[2 * i + 1].wrapping_sub(ts[2 * i])) as f64 * period / 1e9;
                e.1 += 1;
            }
            let mut v: Vec<_> = acc.into_iter().collect();
            v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
            for (n, (t, c)) in v {
                eprintln!("  gpu {n:10} {t:7.3}s over {c} dispatches");
            }
            drop(data);
            p.next.set(0);
            p.names.borrow_mut().clear();
        }

        let bound = self.cfg.correction_bound;
        let mut corrected = vec![0f32; spec.len()];
        corrected.par_chunks_mut(2).zip(spec.par_chunks(2)).zip(raw.par_chunks(4)).for_each(|((out, sp), r)| {
            let (mr, mi) = (2.0 * r[0].tanh(), 2.0 * r[1].tanh());
            let (cr, ci) = (r[2].tanh(), r[3].tanh());
            let norm = (cr * cr + ci * ci).sqrt().max(1.0);
            let mag = (sp[0] * sp[0] + sp[1] * sp[1]).sqrt();
            let kk = bound * mag / norm;
            out[0] = sp[0] * mr - sp[1] * mi + cr * kk;
            out[1] = sp[0] * mi + sp[1] * mr + ci * kk;
        });
        let y = self.stft.inverse(&corrected, audio.len());
        Ok(audio.iter().zip(y).map(|(a, b)| a + b).collect())
    }
}

impl Denoiser for GpuModel {
    fn cfg(&self) -> &Config {
        &self.cfg
    }
    fn run(&self, audio: &[f32]) -> Result<Vec<f32>> {
        const MIN: usize = 4096;
        if audio.len() < MIN {
            let mut padded = audio.to_vec();
            padded.resize(MIN, 0.0);
            let mut y = self.run_gpu(&padded)?;
            y.truncate(audio.len());
            return Ok(y);
        }
        self.run_gpu(audio)
    }
}
