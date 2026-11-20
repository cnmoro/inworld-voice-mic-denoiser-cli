//! Parallel fused f32 kernels for the CPU backend.
//!
//! candle's CPU elementwise ops run on one thread; at the sizes of this model
//! (tens of millions of elements per tensor) they dominate the run time, so the
//! hot spots are re-implemented here with rayon. Every function takes and
//! returns contiguous candle tensors on the CPU device.

use anyhow::{bail, Result};
use candle_core::{Shape, Storage, Tensor};
use rayon::prelude::*;
use std::sync::RwLockReadGuard;

pub struct Slice<'a> {
    guard: RwLockReadGuard<'a, Storage>,
    off: usize,
}

impl Slice<'_> {
    pub fn get(&self) -> &[f32] {
        match &*self.guard {
            Storage::Cpu(c) => &c.as_slice::<f32>().expect("f32 tensor")[self.off..],
            _ => unreachable!("fast kernels only run on CPU tensors"),
        }
    }
}

pub fn view(t: &Tensor) -> Result<Slice<'_>> {
    let (guard, layout) = t.storage_and_layout();
    if !layout.is_contiguous() {
        bail!("fast kernel needs a contiguous tensor");
    }
    Ok(Slice { guard, off: layout.start_offset() })
}

fn out(v: Vec<f32>, shape: impl Into<Shape>) -> Result<Tensor> {
    Ok(Tensor::from_vec(v, shape, &candle_core::Device::Cpu)?)
}

/// Uninitialised-looking output buffer; every element is overwritten by the caller.
fn buf(n: usize) -> Vec<f32> {
    vec![0f32; n]
}

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (sa, sb) = (view(a)?, view(b)?);
    let mut o = buf(a.elem_count());
    o.par_chunks_mut(1 << 14).zip(sa.get().par_chunks(1 << 14)).zip(sb.get().par_chunks(1 << 14)).for_each(
        |((o, a), b)| {
            for i in 0..o.len() {
                o[i] = a[i] + b[i];
            }
        },
    );
    out(o, a.shape().clone())
}

/// y [N, C] + bias [C]
pub fn bias_add(y: &Tensor, b: &Tensor) -> Result<Tensor> {
    let cols = *y.dims().last().unwrap();
    let (sy, sb) = (view(y)?, view(b)?);
    let b = sb.get();
    let mut o = buf(y.elem_count());
    o.par_chunks_mut(cols).zip(sy.get().par_chunks(cols)).for_each(|(o, y)| {
        for i in 0..cols {
            o[i] = y[i] + b[i];
        }
    });
    out(o, y.shape().clone())
}

pub fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let cols = *x.dims().last().unwrap();
    let (sx, sw, sb) = (view(x)?, view(w)?, view(b)?);
    let (w, b) = (sw.get(), sb.get());
    let mut o = buf(x.elem_count());
    o.par_chunks_mut(cols).zip(sx.get().par_chunks(cols)).for_each(|(o, x)| {
        let mean = x.iter().sum::<f32>() / cols as f32;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for i in 0..cols {
            o[i] = (x[i] - mean) * inv * w[i] + b[i];
        }
    });
    out(o, x.shape().clone())
}

pub fn gelu(x: &Tensor) -> Result<Tensor> {
    let sx = view(x)?;
    let mut o = buf(x.elem_count());
    o.par_chunks_mut(1 << 14).zip(sx.get().par_chunks(1 << 14)).for_each(|(o, x)| {
        for i in 0..o.len() {
            o[i] = 0.5 * x[i] * (1.0 + libm::erff(x[i] * std::f32::consts::FRAC_1_SQRT_2));
        }
    });
    out(o, x.shape().clone())
}

/// [A, B, C] -> [B, A, C]
pub fn transpose01(x: &Tensor) -> Result<Tensor> {
    let (a, b, c) = x.dims3()?;
    let sx = view(x)?;
    let xs = sx.get();
    let mut o = buf(a * b * c);
    o.par_chunks_mut(a * c).enumerate().for_each(|(j, dst)| {
        for i in 0..a {
            dst[i * c..(i + 1) * c].copy_from_slice(&xs[(i * b + j) * c..(i * b + j) * c + c]);
        }
    });
    out(o, (b, a, c))
}

/// Causal depthwise convolution over time followed by SiLU, on the first `inner`
/// channels of xz [B, T, 2*inner]. `rev` runs it on the time-reversed sequence
/// (the backward Mamba) without materialising the flip.
pub fn conv_silu(xz: &Tensor, inner: usize, w: &[f32], bias: &[f32], kernel: usize, rev: bool) -> Result<Tensor> {
    let (b, t, c2) = xz.dims3()?;
    let sx = view(xz)?;
    let xs = sx.get();
    let mut o = buf(b * t * inner);
    o.par_chunks_mut(inner).enumerate().for_each(|(row, dst)| {
        let (bi, tt) = (row / t, row % t);
        for ch in 0..inner {
            let mut acc = bias[ch];
            for k in 0..kernel {
                // forward: u[t - (kernel-1) + k]; reversed: u[t + (kernel-1) - k]
                let src = if rev { tt as isize + (kernel - 1 - k) as isize } else { tt as isize - (kernel - 1 - k) as isize };
                if src >= 0 && (src as usize) < t {
                    acc += w[ch * kernel + k] * xs[(bi * t + src as usize) * c2 + ch];
                }
            }
            dst[ch] = acc / (1.0 + (-acc).exp());
        }
    });
    out(o, (b, t, inner))
}

/// Cephes-style expf: branch-free so loops over it vectorise (rel. error ~1e-7).
#[inline(always)]
fn exp_fast(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E + 0.5).floor();
    let r = x - n * 0.693359375 - n * -2.12194440e-4;
    let p = ((((1.987_569_1e-4 * r + 1.398_199_9e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r + 1.666_666_5e-1) * r + 5.000_000_1e-1;
    let y = p * r * r + r + 1.0;
    f32::from_bits(((n as i32 + 127) as u32) << 23) * y
}

struct ScanCtx<'a> {
    u: &'a [f32],
    pre: &'a [f32],
    xz: &'a [f32],
    proj: &'a [f32],
    a: &'a [f32],
    d: &'a [f32],
    t: usize,
    ch: usize,
    p: usize,
    b_off: usize,
    n: usize,
    rev: bool,
}

const CB: usize = 16;

#[inline(always)]
fn scan_task_impl(cx: &ScanCtx, bi: usize, cb: usize) -> Vec<f32> {
    let (t, ch, p, n) = (cx.t, cx.ch, cx.p, cx.n);
    let c0 = cb * CB;
    let w = CB.min(ch - c0);
    let mut h = vec![0f32; w * n];
    let mut o = vec![0f32; t * w];
    for step_i in 0..t {
        let tt = if cx.rev { t - 1 - step_i } else { step_i };
        let row = bi * t + tt;
        let bt = &cx.proj[row * p + cx.b_off..row * p + cx.b_off + n];
        let ct = &cx.proj[row * p + cx.b_off + n..row * p + cx.b_off + 2 * n];
        for j in 0..w {
            let at = row * ch + c0 + j;
            let v = cx.pre[at];
            let step = if v > 20.0 { v } else { exp_fast(v).ln_1p() };
            let x = cx.u[at];
            let drive = step * x;
            let hj = &mut h[j * n..j * n + n];
            let aj = &cx.a[(c0 + j) * n..(c0 + j) * n + n];
            let mut acc;
            if n == 16 {
                // Fixed-size arrays let the compiler vectorise; the reduction is a tree.
                let hj: &mut [f32; 16] = (&mut hj[..16]).try_into().unwrap();
                let (aj, bt, ct): (&[f32; 16], &[f32; 16], &[f32; 16]) =
                    (aj[..16].try_into().unwrap(), bt[..16].try_into().unwrap(), ct[..16].try_into().unwrap());
                let mut e = [0f32; 16];
                for k in 0..16 {
                    e[k] = exp_fast(step * aj[k]);
                }
                let mut prod = [0f32; 16];
                for k in 0..16 {
                    hj[k] = e[k] * hj[k] + drive * bt[k];
                    prod[k] = hj[k] * ct[k];
                }
                let s8: [f32; 8] = std::array::from_fn(|i| prod[i] + prod[i + 8]);
                acc = ((s8[0] + s8[4]) + (s8[1] + s8[5])) + ((s8[2] + s8[6]) + (s8[3] + s8[7]));
            } else {
                acc = 0.0;
                for k in 0..n {
                    hj[k] = exp_fast(step * aj[k]) * hj[k] + drive * bt[k];
                    acc += hj[k] * ct[k];
                }
            }
            acc += x * cx.d[c0 + j];
            let g = cx.xz[row * 2 * ch + ch + c0 + j];
            o[tt * w + j] = acc * (g / (1.0 + exp_fast(-g)));
        }
    }
    o
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_task_avx2(cx: &ScanCtx, bi: usize, cb: usize) -> Vec<f32> {
    scan_task_impl(cx, bi, cb)
}

fn scan_task(cx: &ScanCtx, bi: usize, cb: usize) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: the CPU features were just checked.
            return unsafe { scan_task_avx2(cx, bi, cb) };
        }
    }
    scan_task_impl(cx, bi, cb)
}

/// Selective scan. u, pre: [B, T, C]; xz: [B, T, 2C] (gate z at offset C);
/// proj: [B, T, P] holding B at `b_off` and C at `b_off + n`. `pre` is the
/// pre-softplus step size. Returns y [B, T, C] (gated, with skip).
#[allow(clippy::too_many_arguments)]
pub fn scan(
    u: &Tensor, pre: &Tensor, xz: &Tensor, proj: &Tensor, a: &[f32], d: &[f32], b_off: usize, n: usize, rev: bool,
) -> Result<Tensor> {
    let (batch, t, ch) = u.dims3()?;
    let p = proj.dim(2)?;
    let (su, sp, sz, spj) = (view(u)?, view(pre)?, view(xz)?, view(proj)?);
    let cx = ScanCtx { u: su.get(), pre: sp.get(), xz: sz.get(), proj: spj.get(), a, d, t, ch, p, b_off, n, rev };
    let blocks = ch.div_ceil(CB);
    let tasks: Vec<(usize, usize)> = (0..batch).flat_map(|bi| (0..blocks).map(move |cb| (bi, cb))).collect();
    let parts: Vec<Vec<f32>> = tasks.par_iter().map(|&(bi, cb)| scan_task(&cx, bi, cb)).collect();
    let mut y = buf(batch * t * ch);
    for (&(bi, cb), part) in tasks.iter().zip(parts.iter()) {
        let c0 = cb * CB;
        let w = CB.min(ch - c0);
        for tt in 0..t {
            let dst = (bi * t + tt) * ch + c0;
            y[dst..dst + w].copy_from_slice(&part[tt * w..tt * w + w]);
        }
    }
    out(y, (batch, t, ch))
}

/// Splits qkv [G, T, 3*H*hd] into q [G, H, T, hd] (pre-scaled by `scale`),
/// kt [G, H, hd, T] (transposed, ready for q @ kt) and v [G, H, T, hd],
/// optionally applying rotary embeddings (cos, sin: [T, hd/2]) to q and k.
pub fn split_qkv(qkv: &Tensor, heads: usize, scale: f32, rope: Option<(&[f32], &[f32])>) -> Result<(Tensor, Tensor, Tensor)> {
    let (g, t, w3) = qkv.dims3()?;
    let w = w3 / 3;
    let hd = w / heads;
    let half = hd / 2;
    let sq = view(qkv)?;
    let src = sq.get();
    let n = g * heads * t * hd;
    let (mut q, mut v) = (buf(n), buf(n));
    q.par_chunks_mut(hd).zip(v.par_chunks_mut(hd)).enumerate().for_each(|(idx, (q, v))| {
        let (gi, rest) = (idx / (heads * t), idx % (heads * t));
        let (h, tt) = (rest / t, rest % t);
        let base = (gi * t + tt) * w3 + h * hd;
        let (sq_, sv_) = (&src[base..base + hd], &src[base + 2 * w..base + 2 * w + hd]);
        v.copy_from_slice(sv_);
        match rope {
            Some((cos, sin)) => {
                for i in 0..half {
                    let (c, s) = (cos[tt * half + i], sin[tt * half + i]);
                    q[i] = (sq_[i] * c - sq_[i + half] * s) * scale;
                    q[i + half] = (sq_[i] * s + sq_[i + half] * c) * scale;
                }
            }
            None => {
                for i in 0..hd {
                    q[i] = sq_[i] * scale;
                }
            }
        }
    });
    // k transposed: [G*H, hd, T]; parallel over (g, h) slabs.
    let mut kt = buf(n);
    kt.par_chunks_mut(hd * t).enumerate().for_each(|(slab, dst)| {
        let (gi, h) = (slab / heads, slab % heads);
        for tt in 0..t {
            let base = (gi * t + tt) * w3 + w + h * hd;
            let sk_ = &src[base..base + hd];
            match rope {
                Some((cos, sin)) => {
                    for i in 0..half {
                        let (c, s) = (cos[tt * half + i], sin[tt * half + i]);
                        dst[i * t + tt] = sk_[i] * c - sk_[i + half] * s;
                        dst[(i + half) * t + tt] = sk_[i] * s + sk_[i + half] * c;
                    }
                }
                None => {
                    for i in 0..hd {
                        dst[i * t + tt] = sk_[i];
                    }
                }
            }
        }
    });
    Ok((out(q, (g, heads, t, hd))?, out(kt, (g, heads, hd, t))?, out(v, (g, heads, t, hd))?))
}

/// o [G, H, T, hd] -> [G, T, H*hd], optionally removing each token's component
/// along its own value first (exclusive self attention).
pub fn merge_heads(o: &Tensor, v: Option<&Tensor>) -> Result<Tensor> {
    let (g, h, t, hd) = o.dims4()?;
    let so = view(o)?;
    let sv = match v {
        Some(v) => Some(view(v)?),
        None => None,
    };
    let os = so.get();
    let vs = sv.as_ref().map(|s| s.get());
    let w = h * hd;
    let mut res = buf(g * t * w);
    res.par_chunks_mut(w).enumerate().for_each(|(row, dst)| {
        let (gi, tt) = (row / t, row % t);
        for head in 0..h {
            let src = ((gi * h + head) * t + tt) * hd;
            let orow = &os[src..src + hd];
            let d = &mut dst[head * hd..(head + 1) * hd];
            match vs {
                Some(vs) => {
                    let vrow = &vs[src..src + hd];
                    let (mut dot, mut vv) = (0f32, 0f32);
                    for i in 0..hd {
                        dot += orow[i] * vrow[i];
                        vv += vrow[i] * vrow[i];
                    }
                    let coef = dot / vv.max(1e-6);
                    for i in 0..hd {
                        d[i] = orow[i] - coef * vrow[i];
                    }
                }
                None => d.copy_from_slice(orow),
            }
        }
    });
    out(res, (g, t, w))
}
