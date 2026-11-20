//! CPU backend: Rust port of the offline Inworld Voice Isolator network (inference only).
//!
//! A band-split dual-path network over the 48 kHz STFT: one token per band and
//! frame, blocks of time/band attention and bidirectional Mamba, per-band
//! decoder heads that predict a bounded complex mask plus a bounded residual.
//!
//! The 4 TFLOP of a window are almost all matrix products against weights, so
//! the schedule is built to keep the cores computing instead of waiting for
//! memory: consecutive row-local layers run as one chain over a few dozen
//! tokens at a time (their intermediates never leave L2), attention is computed
//! per sequence and head in cache-sized slabs, and a Mamba direction streams
//! each sequence through projection, convolution, scan and output projection
//! chunk by chunk. Only the token matrix itself goes through main memory.

use crate::config::{band_edges, compose, read_config, read_safetensors, Config, Denoiser, RawTensor};
use crate::dsp::Stft;
use crate::gemm::{gemm, gemm_par, kern, Act, Epi, PackedB};
use crate::simd::{self, prof};
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::cell::RefCell;
use std::path::Path;

struct Lin {
    w: PackedB,
    b: Option<Vec<f32>>,
    inp: usize,
    out: usize,
}

struct Ln {
    w: Vec<f32>,
    b: Vec<f32>,
}

struct Enc {
    ln: Ln,
    lin: Lin, // bias includes the band embedding
    start: usize,
    cols: usize,
}

/// One attention sub-block (over time or over bands): pre-norm attention and feed-forward.
struct AttnW {
    norm1: Ln,
    qkv: Lin,
    proj: Lin,
    norm2: Ln,
    ff0: Lin,
    ff1: Lin,
}

struct MambaW {
    in_proj: Lin,
    conv_w: Vec<f32>, // [kernel][inner]
    conv_b: Vec<f32>,
    x_proj: Lin,
    dt_proj: Lin,
    a: Vec<f32>, // -exp(A_log), in the scan's blocked layout
    d: Vec<f32>,
    out: PackedB, // output_proj half @ out_proj: inner -> W
}

struct BiW {
    fwd: MambaW,
    bwd: MambaW,
    out_b: Vec<f32>,
    sum_w: PackedB, // W1 + W2: the "+ x" terms of both directions
    norm: Ln,
}

enum Block {
    Attn(AttnW, AttnW),
    Mamba(BiW, BiW),
}

struct Head {
    ln: Ln,
    l1: Lin,
    l2: Lin,
    start: usize,
}

struct Net {
    cfg: Config,
    encs: Vec<Enc>,
    blocks: Vec<Block>,
    cond_scale: Vec<f32>, // 1 + scale
    cond_shift: Vec<f32>,
    heads: Vec<Head>,
    inner: usize,
    dt_rank: usize,
    pld: usize,
}

struct Scratch {
    frames: usize,
    x: Vec<f32>,
    h: Vec<f32>,
    qkv: Vec<f32>,
    cos: Vec<f32>, // [frames][head_dim / 2]
    sin: Vec<f32>,
}

pub struct Model {
    net: Net,
    stft: Stft,
    scratch: RefCell<Option<Scratch>>,
}

/// A matrix shared between the tasks of a parallel loop; every task touches its own rows
/// (or its own columns of a row).
#[derive(Clone, Copy)]
struct Rows {
    p: *mut f32,
    ld: usize,
}

// SAFETY: see the type's comment; the accessors are unsafe and each use states its claim.
unsafe impl Send for Rows {}
unsafe impl Sync for Rows {}

impl Rows {
    fn new(v: &mut [f32], ld: usize) -> Self {
        Self { p: v.as_mut_ptr(), ld }
    }
    /// # Safety
    /// The rows must exist and nobody may be writing them.
    unsafe fn get<'a>(self, r0: usize, n: usize) -> &'a [f32] {
        std::slice::from_raw_parts(self.p.add(r0 * self.ld), n * self.ld)
    }
    /// # Safety
    /// The rows must exist and the caller must be their only user.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut<'a>(self, r0: usize, n: usize) -> &'a mut [f32] {
        std::slice::from_raw_parts_mut(self.p.add(r0 * self.ld), n * self.ld)
    }
    /// # Safety
    /// The columns must exist and the caller must be their only user.
    #[allow(clippy::mut_from_ref)]
    unsafe fn cols_mut<'a>(self, row: usize, c0: usize, n: usize) -> &'a mut [f32] {
        std::slice::from_raw_parts_mut(self.p.add(row * self.ld + c0), n)
    }
}

/// Mapping of tokens (rows) to (sequence, position): row = g * gs + t * ts.
#[derive(Clone, Copy)]
struct Map {
    g: usize,
    t: usize,
    gs: usize,
    ts: usize,
}

#[derive(Default)]
struct Tl {
    a: Vec<f32>,
    b: Vec<f32>,
}

#[derive(Default)]
struct AttWs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    s: Vec<f32>,
    o: Vec<f32>,
    kt: PackedB,
    vp: PackedB,
}

/// Buffers of one Mamba sequence sweep.
#[derive(Default)]
struct Sweep {
    xin: Vec<f32>,
    xz: Vec<f32>,
    u: Vec<f32>,
    step: Vec<f32>,
    proj: Vec<f32>,
    out: Vec<f32>,
    state: Vec<f32>,
    idx: Vec<usize>,
}

thread_local! {
    static TL: RefCell<Tl> = RefCell::new(Tl::default());
    static ATT: RefCell<AttWs> = RefCell::new(AttWs::default());
    static SWEEP: RefCell<Sweep> = RefCell::new(Sweep::default());
}

fn buf(v: &mut Vec<f32>, n: usize) -> &mut [f32] {
    if v.len() < n {
        v.resize(n, 0.0);
    }
    &mut v[..n]
}

/// Blocks of at most `blk` consecutive rows that do not straddle a band.
fn row_blocks(bands: usize, frames: usize, blk: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for b in 0..bands {
        let mut r = 0;
        while r < frames {
            let n = blk.min(frames - r);
            v.push((b * frames + r, n));
            r += n;
        }
    }
    v
}

fn each_block(blocks: &[(usize, usize)], f: impl Fn(usize, usize, &mut Tl) + Sync) {
    blocks.par_iter().for_each(|&(r0, n)| TL.with(|tl| f(r0, n, &mut tl.borrow_mut())));
}

#[derive(Clone, Copy)]
struct Ptr(*mut f32);
// SAFETY: used only by scan_split, whose tasks work on disjoint channels.
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

impl Net {
    /// qkv[r] = LayerNorm(x[r]) @ W + b
    fn qkv_chain(&self, blocks: &[(usize, usize)], x: Rows, ln: &Ln, lin: &Lin, qkv: Rows) {
        let w = self.cfg.width;
        each_block(blocks, |r0, n, tl| {
            // SAFETY: blocks are disjoint; x is only read during this loop.
            let (xb, qb) = unsafe { (x.get(r0, n), qkv.get_mut(r0, n)) };
            let t = buf(&mut tl.a, n * w);
            simd::layer_norm(xb, w, n, w, &ln.w, &ln.b, t, w, false);
            gemm(t, w, n, &lin.w, qb, lin.out, Epi { accum: false, bias: lin.b.as_deref(), act: Act::None });
        });
    }

    /// x += h @ proj; x += ff1(gelu(ff0(LayerNorm(x)))); and, if asked, the next attention's qkv from the new x.
    fn attn_tail(&self, blocks: &[(usize, usize)], x: Rows, h: Rows, aw: &AttnW, next: Option<(&Ln, &Lin)>, qkv: Rows) {
        let w = self.cfg.width;
        let ff = aw.ff0.out;
        each_block(blocks, |r0, n, tl| {
            // SAFETY: blocks are disjoint; h is only read during this loop.
            let (xb, hb) = unsafe { (x.get_mut(r0, n), h.get(r0, n)) };
            gemm(hb, w, n, &aw.proj.w, xb, w, Epi { accum: true, bias: aw.proj.b.as_deref(), act: Act::None });
            let t = buf(&mut tl.a, n * w);
            simd::layer_norm(xb, w, n, w, &aw.norm2.w, &aw.norm2.b, t, w, false);
            let mid = buf(&mut tl.b, n * ff);
            gemm(t, w, n, &aw.ff0.w, mid, ff, Epi { accum: false, bias: aw.ff0.b.as_deref(), act: Act::Gelu });
            gemm(mid, ff, n, &aw.ff1.w, xb, w, Epi { accum: true, bias: aw.ff1.b.as_deref(), act: Act::None });
            if let Some((ln, lin)) = next {
                // SAFETY: as above.
                let qb = unsafe { qkv.get_mut(r0, n) };
                simd::layer_norm(xb, w, n, w, &ln.w, &ln.b, t, w, false);
                gemm(t, w, n, &lin.w, qb, lin.out, Epi { accum: false, bias: lin.b.as_deref(), act: Act::None });
            }
        });
    }

    /// Multi-head self attention over the sequences of `map`: out[row] = heads merged (before the output projection).
    fn attention(&self, qkv: Rows, out: Rows, map: Map, rope: Option<(&[f32], &[f32])>, excl: bool) {
        let (w, heads) = (self.cfg.width, self.cfg.num_heads);
        let hd = w / heads;
        let half = hd / 2;
        let scale = 1.0 / (hd as f32).sqrt();
        let t = map.t;
        // Query rows per slab: the scores (rb x t) stay in L2.
        let mr = kern().mr;
        let rb = ((48_000 / t.max(1)).clamp(mr, 96) / mr * mr).min(t.max(1));
        (0..map.g * heads).into_par_iter().for_each(|i| {
            let (g, head) = (i / heads, i % heads);
            ATT.with(|ws| {
                let t_start = std::time::Instant::now();
                let ws = &mut *ws.borrow_mut();
                let (q, k, v) = (buf(&mut ws.q, t * hd), buf(&mut ws.k, t * hd), buf(&mut ws.v, t * hd));
                for tt in 0..t {
                    // SAFETY: qkv is only read during this loop.
                    let src = unsafe { qkv.get(g * map.gs + tt * map.ts, 1) };
                    let (sq, sk, sv) = (&src[head * hd..][..hd], &src[w + head * hd..][..hd], &src[2 * w + head * hd..][..hd]);
                    let (qr, kr) = (&mut q[tt * hd..(tt + 1) * hd], &mut k[tt * hd..(tt + 1) * hd]);
                    match rope {
                        Some((cos, sin)) => {
                            let (c, s) = (&cos[tt * half..(tt + 1) * half], &sin[tt * half..(tt + 1) * half]);
                            for j in 0..half {
                                qr[j] = (sq[j] * c[j] - sq[j + half] * s[j]) * scale;
                                qr[j + half] = (sq[j] * s[j] + sq[j + half] * c[j]) * scale;
                                kr[j] = sk[j] * c[j] - sk[j + half] * s[j];
                                kr[j + half] = sk[j] * s[j] + sk[j + half] * c[j];
                            }
                        }
                        None => {
                            for j in 0..hd {
                                qr[j] = sq[j] * scale;
                            }
                            kr.copy_from_slice(sk);
                        }
                    }
                    v[tt * hd..(tt + 1) * hd].copy_from_slice(sv);
                }
                ws.kt.pack(k, hd, hd, t, true); // K^T: [hd][t]
                ws.vp.pack(v, hd, t, hd, false);
                if prof::on() {
                    prof::add(5, t_start.elapsed());
                }
                let (s, o) = (buf(&mut ws.s, rb * t), buf(&mut ws.o, rb * hd));
                for r0 in (0..t).step_by(rb) {
                    let n = rb.min(t - r0);
                    prof::part(6, || gemm(&q[r0 * hd..], hd, n, &ws.kt, s, t, Epi::STORE));
                    prof::part(7, || simd::softmax_rows(s, n, t));
                    prof::part(6, || gemm(s, t, n, &ws.vp, o, hd, Epi::STORE));
                    for i in 0..n {
                        let (orow, vrow) = (&o[i * hd..(i + 1) * hd], &v[(r0 + i) * hd..(r0 + i + 1) * hd]);
                        // SAFETY: this task is the only one writing head `head` of sequence g.
                        let dst = unsafe { out.cols_mut(g * map.gs + (r0 + i) * map.ts, head * hd, hd) };
                        if excl {
                            // Exclusive self attention: drop the component along the token's own value.
                            let (mut dot, mut vv) = (0f32, 0f32);
                            for j in 0..hd {
                                dot += orow[j] * vrow[j];
                                vv += vrow[j] * vrow[j];
                            }
                            let coef = dot / vv.max(1e-6);
                            for j in 0..hd {
                                dst[j] = orow[j] - coef * vrow[j];
                            }
                        } else {
                            dst.copy_from_slice(orow);
                        }
                    }
                }
            });
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_split(&self, mw: &MambaW, n: usize, c0: usize, c1: usize, u: Ptr, sw_step: &[f32], xz: &[f32], proj: &[f32], state: Ptr, nested: bool) {
        let inner = self.inner;
        if nested && c1 - c0 > 4 * simd::SCAN_CB {
            let mid = c0 + ((c1 - c0) / 2).div_ceil(simd::SCAN_CB) * simd::SCAN_CB;
            rayon::join(
                || self.scan_split(mw, n, c0, mid, u, sw_step, xz, proj, state, nested),
                || self.scan_split(mw, n, mid, c1, u, sw_step, xz, proj, state, nested),
            );
            return;
        }
        // SAFETY: u is [n][inner] and state has scan_len floats; parallel calls get disjoint channel ranges.
        unsafe {
            simd::scan_chunk(
                n, c0, c1, inner, self.cfg.mamba_state, u.0, sw_step, xz, 2 * inner, inner, proj, self.pld, self.dt_rank, &mw.a, &mw.d, state.0,
            )
        };
    }

    /// One direction of a BiMamba over sequence g, streamed in chunks of rows. The forward
    /// direction leaves its part of the output projection (plus bias) in h; the backward one
    /// adds its own and the "+ x" terms, normalises and adds the result to x.
    #[allow(clippy::too_many_arguments)]
    fn sweep(&self, bw: &BiW, rev: bool, x: Rows, h: Rows, map: Map, g: usize, sw: &mut Sweep, nested: bool) {
        let mw = if rev { &bw.bwd } else { &bw.fwd };
        let (w, inner, pld) = (self.cfg.width, self.inner, self.pld);
        let (ns, kernel) = (self.cfg.mamba_state, self.cfg.mamba_conv);
        let hist = kernel - 1;
        let ld = 2 * inner;
        let ch = kern().blk.max(hist).min(map.t.max(hist));
        let min_rows = kern().mr * 2;
        let mm = |a: &[f32], a_ld: usize, m: usize, pb: &PackedB, c: &mut [f32], c_ld: usize, epi: Epi| {
            if nested {
                gemm_par(a, a_ld, m, pb, c, c_ld, epi, min_rows)
            } else {
                gemm(a, a_ld, m, pb, c, c_ld, epi)
            }
        };
        let xin = buf(&mut sw.xin, ch * w);
        let xz = buf(&mut sw.xz, (hist + ch) * ld);
        let u = buf(&mut sw.u, ch * inner);
        let step = buf(&mut sw.step, ch * inner);
        let proj = buf(&mut sw.proj, ch * pld);
        let out = buf(&mut sw.out, ch * w);
        let state = buf(&mut sw.state, simd::scan_len(ns, inner));
        sw.idx.resize(ch, 0);
        let idx = &mut sw.idx[..];
        state.fill(0.0);
        xz[..hist * ld].fill(0.0);
        let mut prev = 0;
        for c0 in (0..map.t).step_by(ch) {
            let n = ch.min(map.t - c0);
            for i in 0..n {
                let tt = if rev { map.t - 1 - (c0 + i) } else { c0 + i };
                idx[i] = g * map.gs + tt * map.ts;
                // SAFETY: x rows of this sequence are written only by this task, further down.
                xin[i * w..(i + 1) * w].copy_from_slice(unsafe { x.get(idx[i], 1) });
            }
            if c0 > 0 {
                // The convolution's history: the last rows of the previous chunk.
                for j in 0..hist {
                    xz.copy_within((prev + j) * ld..(prev + j) * ld + inner, j * ld);
                }
            }
            prev = n;
            prof::part(0, || mm(xin, w, n, &mw.in_proj.w, &mut xz[hist * ld..], ld, Epi { accum: false, bias: None, act: Act::SiluFrom(inner) }));
            prof::part(1, || simd::conv_silu(xz, ld, n, inner, &mw.conv_w, &mw.conv_b, kernel, u));
            prof::part(2, || {
                mm(u, inner, n, &mw.x_proj.w, proj, pld, Epi::STORE);
                mm(proj, pld, n, &mw.dt_proj.w, step, inner, Epi { accum: false, bias: mw.dt_proj.b.as_deref(), act: Act::Softplus });
            });
            prof::part(3, || self.scan_split(mw, n, 0, inner, Ptr(u.as_mut_ptr()), step, &xz[hist * ld..], proj, Ptr(state.as_mut_ptr()), nested));
            let _t = std::time::Instant::now();
            if !rev {
                mm(u, inner, n, &mw.out, out, w, Epi { accum: false, bias: Some(&bw.out_b), act: Act::None });
                for i in 0..n {
                    // SAFETY: h rows of this sequence belong to this task.
                    unsafe { h.get_mut(idx[i], 1) }.copy_from_slice(&out[i * w..(i + 1) * w]);
                }
            } else {
                for i in 0..n {
                    // SAFETY: as above.
                    out[i * w..(i + 1) * w].copy_from_slice(unsafe { h.get(idx[i], 1) });
                }
                mm(u, inner, n, &mw.out, out, w, Epi::ACCUM);
                mm(xin, w, n, &bw.sum_w, out, w, Epi::ACCUM);
                for i in 0..n {
                    // SAFETY: x rows of this sequence belong to this task, and this row is not read again.
                    let xr = unsafe { x.get_mut(idx[i], 1) };
                    simd::layer_norm(&out[i * w..(i + 1) * w], w, 1, w, &bw.norm.w, &bw.norm.b, xr, w, true);
                }
            }
            if prof::on() {
                prof::add(4, _t.elapsed());
            }
        }
    }

    /// x += LayerNorm(output_proj([fwd(x) + x, bwd(x) + x])) over the sequences of `map`.
    fn bimamba(&self, bw: &BiW, x: Rows, h: Rows, map: Map) {
        // With few long sequences the stages of a sequence are split further, so that no core idles.
        let nested = map.g < 4 * rayon::current_num_threads();
        (0..map.g).into_par_iter().for_each(|g| {
            if nested {
                // Nested joins may run another sequence on this thread: no thread-local buffers.
                let mut sw = Sweep::default();
                self.sweep(bw, false, x, h, map, g, &mut sw, true);
                self.sweep(bw, true, x, h, map, g, &mut sw, true);
            } else {
                SWEEP.with(|sw| {
                    let sw = &mut *sw.borrow_mut();
                    self.sweep(bw, false, x, h, map, g, sw, false);
                    self.sweep(bw, true, x, h, map, g, sw, false);
                });
            }
        });
    }

    /// spec [frames][bins][2] -> raw [frames][bins][4] (mask and residual before their bounds).
    fn forward(&self, spec: &[f32], frames: usize, bins: usize, s: &mut Scratch, raw: &mut [f32]) {
        let cfg = &self.cfg;
        let (w, nb) = (cfg.width, self.encs.len());
        let rows = nb * frames;
        let blocks = row_blocks(nb, frames, kern().blk);
        let x = Rows::new(&mut s.x, w);
        let h = Rows::new(&mut s.h, w);
        let qkv = Rows::new(&mut s.qkv, 3 * w);
        let time = Map { g: nb, t: frames, gs: frames, ts: 1 };
        let band = Map { g: frames, t: nb, gs: 1, ts: frames };
        debug_assert_eq!(s.x.len(), rows * w);

        // Encode: one token per band and frame.
        prof::time("encode", || {
            each_block(&blocks, |r0, n, tl| {
                let (e, f0) = (&self.encs[r0 / frames], r0 % frames);
                let t = buf(&mut tl.a, n * e.cols);
                simd::layer_norm(&spec[f0 * bins * 2 + e.start * 2..], bins * 2, n, e.cols, &e.ln.w, &e.ln.b, t, e.cols, false);
                // SAFETY: blocks are disjoint.
                let xb = unsafe { x.get_mut(r0, n) };
                gemm(t, e.cols, n, &e.lin.w, xb, w, Epi { accum: false, bias: e.lin.b.as_deref(), act: Act::None });
            })
        });

        let rope = (s.cos.as_slice(), s.sin.as_slice());
        let mut qkv_ready = false;
        for (bi, block) in self.blocks.iter().enumerate() {
            match block {
                Block::Attn(tw, bw) => {
                    if !qkv_ready {
                        prof::time("qkv", || self.qkv_chain(&blocks, x, &tw.norm1, &tw.qkv, qkv));
                    }
                    prof::time("time attention", || self.attention(qkv, h, time, Some(rope), cfg.exclusive_self_attention));
                    prof::time("proj+ffn+qkv", || self.attn_tail(&blocks, x, h, tw, Some((&bw.norm1, &bw.qkv)), qkv));
                    prof::time("band attention", || self.attention(qkv, h, band, None, false));
                    let next = match self.blocks.get(bi + 1) {
                        Some(Block::Attn(nt, _)) => Some((&nt.norm1, &nt.qkv)),
                        _ => None,
                    };
                    qkv_ready = next.is_some();
                    prof::time("proj+ffn+qkv", || self.attn_tail(&blocks, x, h, bw, next, qkv));
                }
                Block::Mamba(tm, fm) => {
                    prof::time("time mamba", || self.bimamba(tm, x, h, time));
                    prof::time("band mamba", || self.bimamba(fm, x, h, band));
                    qkv_ready = false;
                }
            }
        }

        // Decode: per-band heads give a complex mask and a residual per bin.
        let rawp = Rows::new(raw, bins * 4);
        prof::time("decode", || {
            each_block(&blocks, |r0, n, tl| {
                let (hd, f0) = (&self.heads[r0 / frames], r0 % frames);
                // SAFETY: x is only read here.
                let xb = unsafe { x.get(r0, n) };
                let t = buf(&mut tl.a, n * w);
                for (tr, xr) in t.chunks_exact_mut(w).zip(xb.chunks_exact(w)) {
                    for j in 0..w {
                        tr[j] = xr[j] * self.cond_scale[j] + self.cond_shift[j];
                    }
                }
                let u = buf(&mut tl.b, n * w);
                simd::layer_norm(t, w, n, w, &hd.ln.w, &hd.ln.b, u, w, false);
                gemm(u, w, n, &hd.l1.w, t, w, Epi { accum: false, bias: hd.l1.b.as_deref(), act: Act::Gelu });
                let ow = hd.l2.out;
                let o = &mut u[..n * ow];
                gemm(t, w, n, &hd.l2.w, o, ow, Epi { accum: false, bias: hd.l2.b.as_deref(), act: Act::None });
                for i in 0..n {
                    // SAFETY: this band's bins of these frames belong to this block.
                    unsafe { rawp.cols_mut(f0 + i, hd.start * 4, ow) }.copy_from_slice(&o[i * ow..(i + 1) * ow]);
                }
            })
        });
    }
}

impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        let cfg = read_config(path)?;
        let raw = read_safetensors(path)?;
        let get = |n: &str| -> Result<&RawTensor> { raw.get(n).with_context(|| format!("checkpoint is missing tensor '{n}'")) };
        let ln = |p: &str| -> Result<Ln> { Ok(Ln { w: get(&format!("{p}.weight"))?.data.clone(), b: get(&format!("{p}.bias"))?.data.clone() }) };
        let lin_from = |w: &RawTensor, b: Option<Vec<f32>>| -> Lin {
            let (out, inp) = (w.shape[0], w.shape[1]);
            Lin { w: PackedB::from_weight(&w.data, inp, out), b, inp, out }
        };
        let lin = |p: &str, bias: bool| -> Result<Lin> {
            let b = if bias { Some(get(&format!("{p}.bias"))?.data.clone()) } else { None };
            Ok(lin_from(get(&format!("{p}.weight"))?, b))
        };
        let w = cfg.width;
        let inner = cfg.mamba_expand * w;
        let ns = cfg.mamba_state;
        let kernel = cfg.mamba_conv;
        if kernel < 2 {
            bail!("unsupported Mamba convolution size {kernel}");
        }
        // `half` selects the columns of the BiMamba output projection that this direction feeds.
        let mamba = |p: &str, outer: &RawTensor, half: usize| -> Result<MambaW> {
            let cw = get(&format!("{p}.conv1d.weight"))?; // [inner][1][kernel]
            let mut conv_w = vec![0f32; kernel * inner];
            for c in 0..inner {
                for k in 0..kernel {
                    conv_w[k * inner + c] = cw.data[c * kernel + k];
                }
            }
            let al = get(&format!("{p}.A_log"))?; // [inner][state]
            let mut a = vec![0f32; simd::scan_len(ns, inner)];
            for c in 0..inner {
                for k in 0..ns {
                    a[simd::scan_index(ns, k, c)] = -al.data[c * ns + k].exp();
                }
            }
            let op = get(&format!("{p}.out_proj.weight"))?; // [W][inner]
            let out = compose(&outer.data, w, 2 * w, half * w, w, &op.data, inner);
            Ok(MambaW {
                in_proj: lin(&format!("{p}.in_proj"), false)?,
                conv_w,
                conv_b: get(&format!("{p}.conv1d.bias"))?.data.clone(),
                x_proj: lin(&format!("{p}.x_proj"), false)?,
                dt_proj: lin(&format!("{p}.dt_proj"), true)?,
                a,
                d: get(&format!("{p}.D"))?.data.clone(),
                out: PackedB::from_weight(&out, inner, w),
            })
        };
        let bimamba = |p: &str| -> Result<BiW> {
            let ow = get(&format!("{p}.output_proj.weight"))?; // [W][2W]
            let mut sum = vec![0f32; w * w];
            for r in 0..w {
                for c in 0..w {
                    sum[r * w + c] = ow.data[r * 2 * w + c] + ow.data[r * 2 * w + w + c];
                }
            }
            Ok(BiW {
                fwd: mamba(&format!("{p}.forward_blocks"), ow, 0)?,
                bwd: mamba(&format!("{p}.backward_blocks"), ow, 1)?,
                out_b: get(&format!("{p}.output_proj.bias"))?.data.clone(),
                sum_w: PackedB::from_weight(&sum, w, w),
                norm: ln(&format!("{p}.norm"))?,
            })
        };

        let edges = band_edges(&cfg);
        let emb = get("band_embedding")?;
        let mut encs = Vec::new();
        for (i, (s, e)) in edges.iter().enumerate() {
            let bl = get(&format!("encoders.{i}.1.bias"))?;
            let total: Vec<f32> = (0..w).map(|j| bl.data[j] + emb.data[i * w + j]).collect();
            encs.push(Enc {
                ln: ln(&format!("encoders.{i}.0"))?,
                lin: lin_from(get(&format!("encoders.{i}.1.weight"))?, Some(total)),
                start: *s,
                cols: 2 * (e - s),
            });
        }
        let mut blocks = Vec::new();
        for (i, kind) in cfg.block_types.iter().enumerate() {
            let p = format!("blocks.{i}");
            blocks.push(match kind.as_str() {
                "transformer" => Block::Attn(
                    AttnW {
                        norm1: ln(&format!("{p}.time_transformer.norm1"))?,
                        qkv: lin(&format!("{p}.time_transformer.attn.qkv"), true)?,
                        proj: lin(&format!("{p}.time_transformer.attn.proj"), true)?,
                        norm2: ln(&format!("{p}.time_transformer.norm2"))?,
                        ff0: lin(&format!("{p}.time_transformer.ffn.0"), true)?,
                        ff1: lin(&format!("{p}.time_transformer.ffn.3"), true)?,
                    },
                    AttnW {
                        norm1: ln(&format!("{p}.band_transformer.norm1"))?,
                        qkv: lin_from(
                            get(&format!("{p}.band_transformer.self_attn.in_proj_weight"))?,
                            Some(get(&format!("{p}.band_transformer.self_attn.in_proj_bias"))?.data.clone()),
                        ),
                        proj: lin(&format!("{p}.band_transformer.self_attn.out_proj"), true)?,
                        norm2: ln(&format!("{p}.band_transformer.norm2"))?,
                        ff0: lin(&format!("{p}.band_transformer.linear1"), true)?,
                        ff1: lin(&format!("{p}.band_transformer.linear2"), true)?,
                    },
                ),
                "mamba" => Block::Mamba(bimamba(&format!("{p}.time_mamba"))?, bimamba(&format!("{p}.freq_mamba"))?),
                other => bail!("unknown block type {other}"),
            });
        }
        // The decoder's conditioning input is the constant [1, 0].
        let cw = get("denoise_decoder.condition.weight")?;
        let cb = get("denoise_decoder.condition.bias")?;
        let cond: Vec<f32> = (0..2 * w).map(|j| cw.data[j * 2] + cb.data[j]).collect();
        let heads = edges
            .iter()
            .enumerate()
            .map(|(i, (s, _))| {
                Ok(Head {
                    ln: ln(&format!("denoise_decoder.heads.{i}.0"))?,
                    l1: lin(&format!("denoise_decoder.heads.{i}.1"), true)?,
                    l2: lin(&format!("denoise_decoder.heads.{i}.3"), true)?,
                    start: *s,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for b in &blocks {
            if let Block::Attn(t, bd) = b {
                if t.qkv.out != 3 * w || bd.qkv.out != 3 * w || t.qkv.inp != w || t.ff1.inp != t.ff0.out || bd.ff1.inp != bd.ff0.out {
                    bail!("unexpected attention layer shapes in the checkpoint");
                }
            }
        }
        let dt_rank = w.div_ceil(16);
        let stft = Stft::new(cfg.n_fft, cfg.hop_length);
        let net = Net {
            encs,
            blocks,
            cond_scale: cond[..w].iter().map(|v| 1.0 + v).collect(),
            cond_shift: cond[w..].to_vec(),
            heads,
            inner,
            dt_rank,
            pld: dt_rank + 2 * ns,
            cfg,
        };
        Ok(Self { net, stft, scratch: RefCell::new(None) })
    }

    fn scratch(&self, frames: usize) -> Scratch {
        let cfg = &self.net.cfg;
        let rows = self.net.encs.len() * frames;
        let half = cfg.width / cfg.num_heads / 2;
        let inv: Vec<f32> = (0..half).map(|i| 10000f32.powf(-(i as f32) / half as f32)).collect();
        let angle = |j: usize| (j / half) as f32 * inv[j % half];
        Scratch {
            frames,
            x: vec![0.0; rows * cfg.width],
            h: vec![0.0; rows * cfg.width],
            qkv: vec![0.0; rows * 3 * cfg.width],
            cos: (0..frames * half).map(|j| angle(j).cos()).collect(),
            sin: (0..frames * half).map(|j| angle(j).sin()).collect(),
        }
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
        let frames = self.stft.num_frames(audio.len());
        let bins = self.stft.bins;
        let spec = self.stft.forward(audio);
        let mut cache = self.scratch.borrow_mut();
        if cache.as_ref().map(|c| c.frames) != Some(frames) {
            *cache = None; // free the old buffers first
            *cache = Some(self.scratch(frames));
        }
        let mut raw = vec![0f32; frames * bins * 4];
        self.net.forward(&spec, frames, bins, cache.as_mut().unwrap(), &mut raw);

        // Bounded complex mask plus bounded residual, applied to the spectrum.
        let bound = self.net.cfg.correction_bound;
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
        &self.net.cfg
    }
    fn run(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let t0 = std::time::Instant::now();
        let y = self.run_impl(audio);
        if prof::on() {
            eprintln!("  cpu window {:.3}s ({:?})", t0.elapsed().as_secs_f64(), kern().isa);
            prof::report();
        }
        y
    }
}

#[allow(dead_code)]
fn _assert_sync(n: &Net) -> &(dyn Sync + '_) {
    n
}

/// `--bench`: what this machine can do, for judging how close the model runs to it.
/// Prints the matrix-product rate of one core and of all cores on cache-resident data,
/// the scan rate, and the memory bandwidth.
pub fn bench() {
    use std::time::Instant;
    let kn = kern();
    let threads = rayon::current_num_threads();
    println!("kernel {:?} {}x{}, kc {} mc {} blk {}, {} threads", kn.isa, kn.mr, kn.nr, kn.kc, kn.mc, kn.blk, threads);
    let lcg = |n: usize| -> Vec<f32> { (0..n).map(|i| ((i * 2654435761usize) % 1000) as f32 / 1000.0 - 0.5).collect() };
    for &(m, k, n) in &[(kn.blk, 512usize, 2048usize), (kn.blk, 2048, 512), (kn.blk, 512, 512), (kn.blk, 1024, 64), (kn.blk, 64, 801), (kn.blk, 801, 64)] {
        let pb = PackedB::from_weight(&lcg(n * k), k, n);
        let a = lcg(m * k);
        let run = |secs: f64| -> f64 {
            let mut c = vec![0f32; m * n];
            let (t0, mut it) = (Instant::now(), 0u64);
            while t0.elapsed().as_secs_f64() < secs {
                gemm(&a, k, m, &pb, &mut c, n, Epi::STORE);
                it += 1;
            }
            it as f64 * 2.0 * (m * n * k) as f64 / t0.elapsed().as_secs_f64() / 1e9
        };
        let one = run(0.4);
        let all: f64 = (0..threads).into_par_iter().map(|_| run(0.8)).sum();
        println!("matmul {m:3}x{k:4}x{n:4}: {one:6.1} GFLOP/s on one thread, {all:7.1} GFLOP/s on all");
    }
    {
        let (rows, inner, ns, pld) = (kn.blk, 1024usize, 16usize, 64usize);
        let run = |secs: f64| -> f64 {
            let u0 = lcg(rows * inner);
            let mut u = u0.clone();
            let step: Vec<f32> = lcg(rows * inner).iter().map(|v| v + 0.6).collect();
            let (gate, proj, d) = (lcg(rows * inner), lcg(rows * pld), lcg(inner));
            let a: Vec<f32> = lcg(simd::scan_len(ns, inner)).iter().map(|v| -(v + 0.6) * 4.0).collect();
            let mut state = vec![0f32; simd::scan_len(ns, inner)];
            let (t0, mut it) = (Instant::now(), 0u64);
            while t0.elapsed().as_secs_f64() < secs {
                u.copy_from_slice(&u0);
                // SAFETY: the buffers have the shapes scan_chunk expects and are used by this thread only.
                unsafe { simd::scan_chunk(rows, 0, inner, inner, ns, u.as_mut_ptr(), &step, &gate, inner, 0, &proj, pld, 32, &a, &d, state.as_mut_ptr()) };
                it += 1;
            }
            it as f64 * (rows * inner * ns) as f64 / t0.elapsed().as_secs_f64() / 1e9
        };
        let one = run(0.4);
        let all: f64 = (0..threads).into_par_iter().map(|_| run(0.8)).sum();
        println!("scan: {one:.3} G state updates/s on one thread, {all:.2} on all");
        let run = |secs: f64, act: Act| -> f64 {
            let mut c = lcg(rows * inner);
            let (t0, mut it) = (Instant::now(), 0u64);
            while t0.elapsed().as_secs_f64() < secs {
                for v in c.iter_mut() {
                    *v = (*v).clamp(-3.0, 3.0) + 0.1;
                }
                simd::epilogue(&mut c, None, act);
                it += 1;
            }
            it as f64 * (rows * inner) as f64 / t0.elapsed().as_secs_f64() / 1e9
        };
        println!("gelu {:.3}, softplus {:.3}, silu {:.3} G elements/s on one thread", run(0.3, Act::Gelu), run(0.3, Act::Softplus), run(0.3, Act::SiluFrom(0)));
    }
    {
        let n = 1usize << 27; // 512 MB per array
        let src = vec![1f32; n];
        let mut dst = vec![0f32; n];
        let mut best = (0f64, 0f64);
        for _ in 0..3 {
            let t0 = Instant::now();
            dst.par_chunks_mut(1 << 16).zip(src.par_chunks(1 << 16)).for_each(|(d, s)| d.copy_from_slice(s));
            best.0 = best.0.max(2.0 * (n * 4) as f64 / t0.elapsed().as_secs_f64() / 1e9);
            let t0 = Instant::now();
            let s: f64 = src.par_chunks(1 << 16).map(|c| c.iter().map(|v| *v as f64).sum::<f64>()).sum();
            std::hint::black_box(s);
            best.1 = best.1.max((n * 4) as f64 / t0.elapsed().as_secs_f64() / 1e9);
        }
        println!("memory: copy {:.1} GB/s (read + write), read {:.1} GB/s", best.0, best.1);
    }
}
