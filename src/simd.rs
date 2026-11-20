//! Elementwise and reduction kernels of the CPU backend.
//!
//! Each kernel is written once as plain loops that the compiler vectorises and
//! is compiled for the baseline, for AVX2+FMA and for AVX-512; the widest
//! version the CPU supports is picked at run time (on ARM, NEON is the baseline).

#![allow(clippy::needless_range_loop)] // index loops are what the compiler vectorises here

use crate::gemm::Act;

/// Per-stage wall-clock profile, enabled with INWORLD_PROFILE.
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    type Table = Vec<(String, f64, f64, usize)>;

    fn table() -> Option<&'static Mutex<Table>> {
        static T: OnceLock<Option<Mutex<Table>>> = OnceLock::new();
        T.get_or_init(|| std::env::var_os("INWORLD_PROFILE").map(|_| Mutex::new(Vec::new()))).as_ref()
    }

    pub fn on() -> bool {
        table().is_some()
    }

    /// Runs `f`, charging its time and the matrix-product flops it did to `name`.
    pub fn time<R>(name: &str, f: impl FnOnce() -> R) -> R {
        let Some(t) = table() else { return f() };
        let f0 = crate::gemm::FLOPS.load(Ordering::Relaxed);
        let t0 = Instant::now();
        let r = f();
        let dt = t0.elapsed().as_secs_f64();
        let fl = (crate::gemm::FLOPS.load(Ordering::Relaxed) - f0) as f64;
        let mut g = t.lock().unwrap();
        match g.iter_mut().find(|e| e.0 == name) {
            Some(e) => {
                e.1 += dt;
                e.2 += fl;
                e.3 += 1;
            }
            None => g.push((name.to_string(), dt, fl, 1)),
        }
        r
    }

    /// Core-seconds spent in the parts of the parallel stages, summed over threads.
    pub const PARTS: [&str; 8] = ["in_proj", "conv", "x_proj+dt", "scan", "out+norm", "attn gather+pack", "attn matmul", "attn softmax"];
    static CORE_NS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

    /// Runs `f`, charging its time on this thread to PARTS[part].
    #[inline]
    pub fn part<R>(part: usize, f: impl FnOnce() -> R) -> R {
        if !on() {
            return f();
        }
        let t0 = Instant::now();
        let r = f();
        CORE_NS[part].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    }

    pub fn add(part: usize, d: std::time::Duration) {
        CORE_NS[part].fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Prints and clears the table.
    pub fn report() {
        let Some(t) = table() else { return };
        for (n, c) in PARTS.iter().zip(&CORE_NS) {
            eprintln!("  cpu   {n:20} {:7.3} core-s", c.swap(0, Ordering::Relaxed) as f64 / 1e9);
        }
        let mut g = t.lock().unwrap();
        g.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let (mut tt, mut tf) = (0.0, 0.0);
        for (n, s, f, c) in g.iter() {
            tt += s;
            tf += f;
            eprintln!("  cpu {n:22} {s:7.3}s over {c:3} calls {:7.1} GFLOP in matmul {:6.0} GFLOP/s", f / 1e9, f / s / 1e9);
        }
        eprintln!("  cpu total {tt:.3}s, {:.0} GFLOP in matmul, {:.0} GFLOP/s overall", tf / 1e9, tf / tt.max(1e-9) / 1e9);
        g.clear();
    }
}

/// Defines `pub fn name(args)` from a body generic over `const F: bool` (whether a fused
/// multiply-add is cheap), instantiated per instruction set.
macro_rules! mv {
    ($(#[$m:meta])* pub $(($u:tt))? fn $name:ident<const $f:ident: bool>($($arg:ident: $ty:ty),* $(,)?) $body:block) => {
        $(#[$m])*
        #[allow(clippy::too_many_arguments)]
        pub $($u)? fn $name($($arg: $ty),*) {
            #[inline(always)]
            #[allow(clippy::too_many_arguments, unused_unsafe)]
            $($u)? fn imp<const $f: bool>($($arg: $ty),*) $body
            #[cfg(target_arch = "x86_64")]
            #[allow(unused_unsafe)]
            {
                #[target_feature(enable = "avx512f,avx512vl,avx512dq,avx512bw,avx2,fma")]
                #[allow(clippy::too_many_arguments)]
                unsafe fn v4($($arg: $ty),*) {
                    unsafe { imp::<true>($($arg),*) }
                }
                #[target_feature(enable = "avx2,fma")]
                #[allow(clippy::too_many_arguments)]
                unsafe fn v3($($arg: $ty),*) {
                    unsafe { imp::<true>($($arg),*) }
                }
                // SAFETY: isa() reports only what the CPU supports.
                match crate::gemm::isa() {
                    crate::gemm::Isa::Avx512 => unsafe { v4($($arg),*) },
                    crate::gemm::Isa::Avx2 => unsafe { v3($($arg),*) },
                    _ => unsafe { imp::<false>($($arg),*) },
                }
            }
            #[cfg(target_arch = "aarch64")]
            #[allow(unused_unsafe)]
            unsafe {
                imp::<true>($($arg),*)
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            #[allow(unused_unsafe)]
            unsafe {
                imp::<false>($($arg),*)
            }
        }
    };
}

#[inline(always)]
fn madd<const F: bool>(a: f32, b: f32, c: f32) -> f32 {
    if F {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// Cephes-style expf, branch-free (relative error about 1e-7).
#[inline(always)]
fn exp_f<const F: bool>(x: f32) -> f32 {
    let x = x.max(-87.0).min(88.0);
    // Adding and subtracting 1.5 * 2^23 rounds to the nearest integer.
    let n = (x * std::f32::consts::LOG2_E + 12582912.0) - 12582912.0;
    let r = madd::<F>(n, -0.693359375, x) + n * 2.12194440e-4;
    // Estrin's scheme: the short dependency chain matters more than the operation count,
    // because callers run many of these side by side.
    let r2 = r * r;
    let p01 = madd::<F>(1.666_666_5e-1, r, 5.000_000_1e-1);
    let p23 = madd::<F>(8.333_452e-3, r, 4.166_579_6e-2);
    let p45 = madd::<F>(1.987_569_1e-4, r, 1.398_199_9e-3);
    let p = madd::<F>(r2 * r2, p45, madd::<F>(r2, p23, p01));
    let y = madd::<F>(p, r2, r) + 1.0;
    // SAFETY: n is an integer in [-126, 127] after the clamp.
    let e = unsafe { n.to_int_unchecked::<i32>() };
    f32::from_bits(((e + 127) as u32) << 23) * y
}

/// 2^z, branch-free (relative error about 1e-7): one rounding and a degree-6 polynomial,
/// a shorter dependency chain than exp_f for the scan, which does little else.
#[inline(always)]
fn exp2_f<const F: bool>(z: f32) -> f32 {
    let z = z.max(-126.0).min(127.0);
    let n = if F { z.round_ties_even() } else { (z + 12582912.0) - 12582912.0 };
    let f = z - n;
    let f2 = f * f;
    let p01 = madd::<F>(0.693_147_2, f, 1.0);
    let p23 = madd::<F>(0.055_503_29, f, 0.240_226_46);
    let p45 = madd::<F>(0.001_339_989_5, f, 0.009_618_489);
    let lo = madd::<F>(f2, p23, p01);
    let hi = madd::<F>(f2, 0.000_153_456_97, p45);
    let y = madd::<F>(f2 * f2, hi, lo);
    // SAFETY: n is an integer in [-126, 127] after the clamp.
    let e = unsafe { n.to_int_unchecked::<i32>() };
    f32::from_bits(((e + 127) as u32) << 23) * y
}

/// Cephes-style logf for positive normal x, branch-free.
#[inline(always)]
fn log_f<const F: bool>(x: f32) -> f32 {
    let bits = x.to_bits();
    let mut e = ((bits >> 23) as i32 - 127) as f32;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    let big = m > std::f32::consts::SQRT_2;
    m = if big { m * 0.5 } else { m };
    e = if big { e + 1.0 } else { e };
    let f = m - 1.0;
    let z = f * f;
    let mut p = 7.037_683_6e-2f32;
    p = madd::<F>(p, f, -1.151_461e-1);
    p = madd::<F>(p, f, 1.167_699_9e-1);
    p = madd::<F>(p, f, -1.242_014_1e-1);
    p = madd::<F>(p, f, 1.424_932_3e-1);
    p = madd::<F>(p, f, -1.666_805_8e-1);
    p = madd::<F>(p, f, 2.000_071_5e-1);
    p = madd::<F>(p, f, -2.499_999_4e-1);
    p = madd::<F>(p, f, 3.333_333_1e-1);
    let y = p * f * z + e * -2.121_944_4e-4 - 0.5 * z;
    f + y + e * 0.693_359_4
}

/// The erf of the GPU kernels (Abramowitz and Stegun 7.1.26, absolute error 1.5e-7).
#[inline(always)]
fn gelu_f<const F: bool>(x: f32) -> f32 {
    let v = x * std::f32::consts::FRAC_1_SQRT_2;
    let a = v.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let mut p = 1.061_405_4f32;
    p = madd::<F>(p, t, -1.453_152_1);
    p = madd::<F>(p, t, 1.421_413_8);
    p = madd::<F>(p, t, -0.284_496_72);
    p = madd::<F>(p, t, 0.254_829_6);
    let y = 1.0 - p * t * exp_f::<F>(-a * a);
    0.5 * x * (1.0 + y.copysign(v))
}

#[inline(always)]
fn silu_f<const F: bool>(x: f32) -> f32 {
    x / (1.0 + exp_f::<F>(-x))
}

/// log(1 + exp(x)).
#[inline(always)]
fn softplus_f<const F: bool>(x: f32) -> f32 {
    let e = exp_f::<F>(-x.abs());
    let u = 1.0 + e;
    let d = u - 1.0;
    // log1p(e) = log(u) * e / (u - 1) corrects the rounding of 1 + e.
    let l = log_f::<F>(u) * (e / d);
    x.max(0.0) + if d == 0.0 { e } else { l }
}

#[inline(always)]
fn sum16(a: &[f32; 16]) -> f32 {
    let s8: [f32; 8] = std::array::from_fn(|i| a[i] + a[i + 8]);
    ((s8[0] + s8[4]) + (s8[1] + s8[5])) + ((s8[2] + s8[6]) + (s8[3] + s8[7]))
}

mv! {
    /// c = act(c + bias) over one row.
    pub fn epilogue<const F: bool>(c: &mut [f32], bias: Option<&[f32]>, act: Act) {
        if let Some(b) = bias {
            for (v, b) in c.iter_mut().zip(b) {
                *v += *b;
            }
        }
        match act {
            Act::None => {}
            Act::Gelu => {
                for v in c.iter_mut() {
                    *v = gelu_f::<F>(*v);
                }
            }
            Act::Softplus => {
                for v in c.iter_mut() {
                    *v = softplus_f::<F>(*v);
                }
            }
            Act::SiluFrom(s) => {
                for v in c[s..].iter_mut() {
                    *v = silu_f::<F>(*v);
                }
            }
        }
    }
}

mv! {
    /// dst[r] (+)= LayerNorm(src[r]) * w + b for `rows` rows of `cols` (eps 1e-5).
    pub fn layer_norm<const F: bool>(
        src: &[f32], src_ld: usize, rows: usize, cols: usize, w: &[f32], b: &[f32], dst: &mut [f32], dst_ld: usize, add: bool,
    ) {
        for r in 0..rows {
            let x = &src[r * src_ld..r * src_ld + cols];
            let mut acc = [0f32; 16];
            let ch = x.chunks_exact(16);
            let rem = ch.remainder();
            for c in ch {
                for i in 0..16 {
                    acc[i] += c[i];
                }
            }
            let mean = (sum16(&acc) + rem.iter().sum::<f32>()) / cols as f32;
            let mut acc = [0f32; 16];
            for c in x.chunks_exact(16) {
                for i in 0..16 {
                    let d = c[i] - mean;
                    acc[i] = madd::<F>(d, d, acc[i]);
                }
            }
            let var = (sum16(&acc) + rem.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()) / cols as f32;
            let inv = 1.0 / (var + 1e-5).sqrt();
            let d = &mut dst[r * dst_ld..r * dst_ld + cols];
            if add {
                for i in 0..cols {
                    d[i] += (x[i] - mean) * inv * w[i] + b[i];
                }
            } else {
                for i in 0..cols {
                    d[i] = (x[i] - mean) * inv * w[i] + b[i];
                }
            }
        }
    }
}

mv! {
    /// In-place softmax over each of the `rows` rows of `cols`.
    pub fn softmax_rows<const F: bool>(s: &mut [f32], rows: usize, cols: usize) {
        for r in 0..rows {
            let x = &mut s[r * cols..(r + 1) * cols];
            let mut m = [f32::MIN; 16];
            let ch = x.chunks_exact(16);
            let mut mx = ch.remainder().iter().fold(f32::MIN, |a, b| a.max(*b));
            for c in ch {
                for i in 0..16 {
                    m[i] = if c[i] > m[i] { c[i] } else { m[i] };
                }
            }
            for v in m {
                mx = mx.max(v);
            }
            let mut acc = [0f32; 16];
            let mut ch = x.chunks_exact_mut(16);
            for c in &mut ch {
                for i in 0..16 {
                    c[i] = exp_f::<F>(c[i] - mx);
                    acc[i] += c[i];
                }
            }
            let mut sum = sum16(&acc);
            for v in ch.into_remainder() {
                *v = exp_f::<F>(*v - mx);
                sum += *v;
            }
            let inv = 1.0 / sum;
            for v in x.iter_mut() {
                *v *= inv;
            }
        }
    }
}

mv! {
    /// Causal depthwise convolution over the sequence followed by SiLU.
    /// xz holds `kernel - 1` rows of history and then `rows` rows (leading dimension ld, the first
    /// `inner` columns are convolved); w is [kernel][inner]; u gets [rows][inner].
    pub fn conv_silu<const F: bool>(xz: &[f32], ld: usize, rows: usize, inner: usize, w: &[f32], bias: &[f32], kernel: usize, u: &mut [f32]) {
        for i in 0..rows {
            let o = &mut u[i * inner..(i + 1) * inner];
            o.copy_from_slice(&bias[..inner]);
            for k in 0..kernel {
                let x = &xz[(i + k) * ld..(i + k) * ld + inner];
                let wk = &w[k * inner..(k + 1) * inner];
                for c in 0..inner {
                    o[c] = madd::<F>(wk[c], x[c], o[c]);
                }
            }
            for v in o.iter_mut() {
                *v = silu_f::<F>(*v);
            }
        }
    }
}

/// Channels the scan advances together; their states (ns * SCAN_CB floats) stay in L1.
pub const SCAN_CB: usize = 64;

/// Position of state k of channel c in the scan's `a` and `state` arrays.
#[inline(always)]
pub fn scan_index(ns: usize, k: usize, c: usize) -> usize {
    (c / SCAN_CB * ns + k) * SCAN_CB + c % SCAN_CB
}

/// Length of the scan's `a` and `state` arrays.
pub fn scan_len(ns: usize, inner: usize) -> usize {
    inner.div_ceil(SCAN_CB) * ns * SCAN_CB
}

mv! {
    /// Selective scan over `rows` consecutive steps for the channels c0..c1, in place:
    /// u [rows][inner] holds the conv output on entry and the gated scan output on return.
    /// step [rows][inner] is the (softplus'd) step size, gate [rows][gate_ld] the SiLU'd gate at
    /// column gate_off, proj [rows][pld] holds B at b_off and C at b_off + ns. a is A * log2(e) (the
    /// decay is 2^(step * a)); a and state (h, carried
    /// between calls) are laid out per block of SCAN_CB channels as [block][ns][SCAN_CB] (see
    /// scan_index): a block's states sit together in L1 and its rows are not 4 KB apart.
    ///
    /// # Safety
    /// u and state must be valid for the stated shapes, and no other thread may touch the
    /// columns c0..c1 of either while this runs.
    pub (unsafe) fn scan_chunk<const F: bool>(
        rows: usize, c0: usize, c1: usize, inner: usize, ns: usize, u: *mut f32, step: &[f32], gate: &[f32], gate_ld: usize, gate_off: usize,
        proj: &[f32], pld: usize, b_off: usize, a: &[f32], d: &[f32], state: *mut f32,
    ) {
        let mut b0 = c0;
        while b0 < c1 {
            let cw = SCAN_CB.min(c1 - b0);
            for i in 0..rows {
                let st = &step[i * inner + b0..i * inner + b0 + cw];
                // SAFETY: see the function contract.
                let ur = unsafe { std::slice::from_raw_parts_mut(u.add(i * inner + b0), cw) };
                let g = &gate[i * gate_ld + gate_off + b0..i * gate_ld + gate_off + b0 + cw];
                let pr = &proj[i * pld + b_off..i * pld + b_off + 2 * ns];
                let dd = &d[b0..b0 + cw];
                let mut drive = [0f32; SCAN_CB];
                let mut acc = [0f32; SCAN_CB];
                for j in 0..cw {
                    drive[j] = st[j] * ur[j];
                    acc[j] = ur[j] * dd[j];
                }
                for k in 0..ns {
                    // SAFETY: see the function contract.
                    let hk = unsafe { std::slice::from_raw_parts_mut(state.add(scan_index(ns, k, b0)), cw) };
                    let ak = &a[scan_index(ns, k, b0)..scan_index(ns, k, b0) + cw];
                    let (bk, ck) = (pr[k], pr[ns + k]);
                    // Fixed-size lanes through local arrays: straight-line vector code, no alias checks.
                    const L: usize = 16;
                    let mut j = 0;
                    while j + L <= cw {
                        let s: [f32; L] = st[j..j + L].try_into().unwrap();
                        let al: [f32; L] = ak[j..j + L].try_into().unwrap();
                        let h0: [f32; L] = hk[j..j + L].try_into().unwrap();
                        let dr: [f32; L] = drive[j..j + L].try_into().unwrap();
                        let a0: [f32; L] = acc[j..j + L].try_into().unwrap();
                        let (mut h1, mut a1) = ([0f32; L], [0f32; L]);
                        for q in 0..L {
                            let e = exp2_f::<F>(s[q] * al[q]);
                            let h = madd::<F>(e, h0[q], dr[q] * bk);
                            h1[q] = h;
                            a1[q] = madd::<F>(h, ck, a0[q]);
                        }
                        hk[j..j + L].copy_from_slice(&h1);
                        acc[j..j + L].copy_from_slice(&a1);
                        j += L;
                    }
                    while j < cw {
                        let e = exp2_f::<F>(st[j] * ak[j]);
                        let h = madd::<F>(e, hk[j], drive[j] * bk);
                        hk[j] = h;
                        acc[j] = madd::<F>(h, ck, acc[j]);
                        j += 1;
                    }
                }
                for j in 0..cw {
                    ur[j] = acc[j] * g[j];
                }
            }
            b0 += cw;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn math_kernels_match_libm() {
        let mut worst = [0f32; 4];
        for i in -4000..=4000 {
            let x = i as f32 * 0.01;
            let mut c = [x, x, x, x];
            epilogue(&mut c[0..1], None, Act::Gelu);
            epilogue(&mut c[1..2], None, Act::Softplus);
            epilogue(&mut c[2..4], None, Act::SiluFrom(1));
            let xd = x as f64;
            let gelu = 0.5 * xd * (1.0 + libm::erf(xd / std::f64::consts::SQRT_2));
            let sp = if xd > 20.0 { xd } else { xd.exp().ln_1p() };
            let silu = xd / (1.0 + (-xd).exp());
            for (w, (got, want)) in worst.iter_mut().zip([(c[0], gelu), (c[1], sp), (c[3], silu), (c[2], xd)]) {
                *w = w.max(((got as f64 - want) / (1.0 + want.abs())).abs() as f32);
            }
        }
        assert!(worst[0] < 1e-6 && worst[1] < 1e-6 && worst[2] < 1e-6 && worst[3] == 0.0, "{worst:?}");
    }

    #[test]
    fn softmax_and_layer_norm() {
        let cols = 37;
        let x: Vec<f32> = (0..2 * cols).map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0).collect();
        let mut s = x.clone();
        softmax_rows(&mut s, 2, cols);
        for r in 0..2 {
            let row = &x[r * cols..(r + 1) * cols];
            let m = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
            let z: f64 = row.iter().map(|v| (*v as f64 - m).exp()).sum();
            for c in 0..cols {
                let want = (row[c] as f64 - m).exp() / z;
                assert!((s[r * cols + c] as f64 - want).abs() < 1e-6);
            }
        }
        let (w, b) = (vec![1.5f32; cols], vec![0.25f32; cols]);
        let mut y = vec![1f32; 2 * cols];
        layer_norm(&x, cols, 2, cols, &w, &b, &mut y, cols, true);
        for r in 0..2 {
            let row = &x[r * cols..(r + 1) * cols];
            let mean = row.iter().map(|v| *v as f64).sum::<f64>() / cols as f64;
            let var = row.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / cols as f64;
            for c in 0..cols {
                let want = 1.0 + (row[c] as f64 - mean) / (var + 1e-5).sqrt() * 1.5 + 0.25;
                assert!((y[r * cols + c] as f64 - want).abs() < 1e-5);
            }
        }
    }
}
