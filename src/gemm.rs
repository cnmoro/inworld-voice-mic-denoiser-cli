//! Single-precision matrix product for the CPU backend.
//!
//! A packed, cache-blocked product in the BLIS style: B (almost always a weight
//! matrix, packed once at load time) is stored in NR-wide strips, A is packed per
//! block of rows into MR-wide strips that stay in L2, and a register-tiled
//! microkernel (AVX-512, AVX2+FMA or NEON, chosen at run time) computes MR x NR
//! tiles of C. Callers parallelise over blocks of rows.

#![allow(clippy::needless_range_loop)] // index loops are what the compiler vectorises here

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(dead_code)] // no architecture has every variant
pub enum Isa {
    Generic,
    Avx2,
    Avx512,
    Neon,
}

type Micro = unsafe fn(usize, *const f32, *const f32, *mut f32);

pub struct Kern {
    pub isa: Isa,
    pub mr: usize,
    pub nr: usize,
    /// Columns of K per packed block: one B strip (kc x nr) has to fit in L1.
    pub kc: usize,
    /// Rows per packed block of A (mc x kc floats, in L2).
    pub mc: usize,
    /// Rows a fused chain of layers works on at a time, so its intermediates stay in L2.
    pub blk: usize,
    /// Products go to the system BLAS instead of the microkernel (Apple silicon, see `detect`).
    pub blas: bool,
    micro: Micro,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32, trans_a: i32, trans_b: i32, m: i32, n: i32, k: i32, alpha: f32, a: *const f32, lda: i32, b: *const f32, ldb: i32, beta: f32,
        c: *mut f32, ldc: i32,
    );
}

const MAX_TILE: usize = 12 * 32;

/// Portable fallback, 4 x 8.
unsafe fn micro_generic(kc: usize, a: *const f32, b: *const f32, out: *mut f32) {
    let mut c = [[0f32; 8]; 4];
    for p in 0..kc {
        let ap = &*(a.add(p * 4) as *const [f32; 4]);
        let bp = &*(b.add(p * 8) as *const [f32; 8]);
        for i in 0..4 {
            for j in 0..8 {
                c[i][j] += ap[i] * bp[j];
            }
        }
    }
    std::ptr::copy_nonoverlapping(c.as_ptr() as *const f32, out, 32);
}

/// 6 x 16: twelve ymm accumulators, two FMAs per broadcast of A.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_avx2(kc: usize, mut a: *const f32, mut b: *const f32, out: *mut f32) {
    use std::arch::x86_64::*;
    let mut c = [[_mm256_setzero_ps(); 2]; 6];
    for _ in 0..kc {
        let b0 = _mm256_loadu_ps(b);
        let b1 = _mm256_loadu_ps(b.add(8));
        for i in 0..6 {
            let ai = _mm256_broadcast_ss(&*a.add(i));
            c[i][0] = _mm256_fmadd_ps(ai, b0, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(ai, b1, c[i][1]);
        }
        a = a.add(6);
        b = b.add(16);
    }
    for i in 0..6 {
        _mm256_storeu_ps(out.add(i * 16), c[i][0]);
        _mm256_storeu_ps(out.add(i * 16 + 8), c[i][1]);
    }
}

/// 12 x 32: twenty-four zmm accumulators.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn micro_avx512(kc: usize, mut a: *const f32, mut b: *const f32, out: *mut f32) {
    use std::arch::x86_64::*;
    let mut c = [[_mm512_setzero_ps(); 2]; 12];
    for _ in 0..kc {
        let b0 = _mm512_loadu_ps(b);
        let b1 = _mm512_loadu_ps(b.add(16));
        for i in 0..12 {
            let ai = _mm512_set1_ps(*a.add(i));
            c[i][0] = _mm512_fmadd_ps(ai, b0, c[i][0]);
            c[i][1] = _mm512_fmadd_ps(ai, b1, c[i][1]);
        }
        a = a.add(12);
        b = b.add(32);
    }
    for i in 0..12 {
        _mm512_storeu_ps(out.add(i * 32), c[i][0]);
        _mm512_storeu_ps(out.add(i * 32 + 16), c[i][1]);
    }
}

/// 8 x 12: twenty-four q accumulators.
#[cfg(target_arch = "aarch64")]
unsafe fn micro_neon(kc: usize, mut a: *const f32, mut b: *const f32, out: *mut f32) {
    use std::arch::aarch64::*;
    let mut c = [[vdupq_n_f32(0.0); 3]; 8];
    for _ in 0..kc {
        let b0 = vld1q_f32(b);
        let b1 = vld1q_f32(b.add(4));
        let b2 = vld1q_f32(b.add(8));
        let a0 = vld1q_f32(a);
        let a1 = vld1q_f32(a.add(4));
        macro_rules! row {
            ($i:expr, $av:ident, $lane:literal) => {
                c[$i][0] = vfmaq_laneq_f32::<$lane>(c[$i][0], b0, $av);
                c[$i][1] = vfmaq_laneq_f32::<$lane>(c[$i][1], b1, $av);
                c[$i][2] = vfmaq_laneq_f32::<$lane>(c[$i][2], b2, $av);
            };
        }
        row!(0, a0, 0);
        row!(1, a0, 1);
        row!(2, a0, 2);
        row!(3, a0, 3);
        row!(4, a1, 0);
        row!(5, a1, 1);
        row!(6, a1, 2);
        row!(7, a1, 3);
        a = a.add(8);
        b = b.add(12);
    }
    for i in 0..8 {
        vst1q_f32(out.add(i * 12), c[i][0]);
        vst1q_f32(out.add(i * 12 + 4), c[i][1]);
        vst1q_f32(out.add(i * 12 + 8), c[i][2]);
    }
}

fn detect() -> Kern {
    let force = std::env::var("INWORLD_ISA").unwrap_or_default();
    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        let avx512 = avx2
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512bw");
        if avx512 && force != "avx2" && force != "generic" {
            return Kern { isa: Isa::Avx512, mr: 12, nr: 32, kc: 192, mc: 144, blk: 96, blas: false, micro: micro_avx512 };
        }
        if avx2 && force != "generic" {
            return Kern { isa: Isa::Avx2, mr: 6, nr: 16, kc: 256, mc: 120, blk: 48, blas: false, micro: micro_avx2 };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if force != "generic" {
            // Apple cores have 128 KB of L1 and several MB of L2 per core; other ARM cores much less.
            let (kc, mc, blk) = if cfg!(target_os = "macos") { (512, 192, 96) } else { (256, 96, 48) };
            // Apple silicon has a matrix coprocessor (AMX / SME) that only Accelerate's BLAS drives;
            // on an M4 it multiplies several times faster than all NEON cores together, and
            // letting every thread call it beats keeping some of them on NEON.
            // INWORLD_ISA=neon keeps the products on the NEON microkernel.
            let blas = cfg!(target_os = "macos") && force != "neon";
            return Kern { isa: Isa::Neon, mr: 8, nr: 12, kc, mc, blk, blas, micro: micro_neon };
        }
    }
    Kern { isa: Isa::Generic, mr: 4, nr: 8, kc: 256, mc: 96, blk: 48, blas: false, micro: micro_generic }
}

pub fn kern() -> &'static Kern {
    static K: OnceLock<Kern> = OnceLock::new();
    K.get_or_init(|| {
        let mut k = detect();
        // INWORLD_GEMM="kc,mc,blk" overrides the cache blocking (for tuning).
        if let Ok(s) = std::env::var("INWORLD_GEMM") {
            let v: Vec<usize> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            if v.len() == 3 && v.iter().all(|&x| x > 0) {
                (k.kc, k.mc, k.blk) = (v[0], v[1].div_ceil(k.mr) * k.mr, v[2]);
            }
        }
        k
    })
}

#[allow(dead_code)] // only the x86 kernels dispatch on it
pub fn isa() -> Isa {
    kern().isa
}

/// Flops done by `gemm` so far (counted only while profiling).
pub static FLOPS: AtomicU64 = AtomicU64::new(0);

/// B [k][n] packed for the microkernel: per block of K, per NR-wide strip, [kc][nr].
/// (Kept as a plain row-major [k][n] matrix when the products go to the system BLAS.)
#[derive(Default)]
pub struct PackedB {
    pub k: usize,
    pub n: usize,
    blocks: Vec<(usize, usize, usize)>, // k0, kc, offset of the block's first strip
    buf: Vec<f32>,
    off: usize, // start of the 64-byte aligned data in buf
    plain: Vec<f32>, // row-major [k][n], used instead of buf when the products go to the system BLAS
}

impl PackedB {
    /// Packs B[p][j] = src[p * ld + j], or src[j * ld + p] when `trans` (a weight stored [out][in]).
    pub fn pack(&mut self, src: &[f32], ld: usize, k: usize, n: usize, trans: bool) {
        let kn = kern();
        self.k = k;
        self.n = n;
        self.blocks.clear();
        if kn.blas {
            if self.plain.len() < k * n {
                self.plain = vec![0f32; k * n];
            }
            if trans {
                for j in 0..n {
                    for (p, v) in src[j * ld..j * ld + k].iter().enumerate() {
                        self.plain[p * n + j] = *v;
                    }
                }
            } else {
                for p in 0..k {
                    self.plain[p * n..(p + 1) * n].copy_from_slice(&src[p * ld..p * ld + n]);
                }
            }
            return;
        }
        let nr = kn.nr;
        let strips = n.div_ceil(nr);
        let nb = k.div_ceil(kn.kc).max(1);
        let (base, rem) = (k / nb, k % nb);
        let need = k * strips * nr;
        if self.buf.len() < need + 16 {
            self.buf = vec![0f32; need + 16];
        }
        self.off = self.buf.as_ptr().align_offset(64).min(15);
        let dst = &mut self.buf[self.off..self.off + need];
        let (mut k0, mut off) = (0, 0);
        for b in 0..nb {
            let kc = base + (b < rem) as usize;
            self.blocks.push((k0, kc, off));
            for s in 0..strips {
                let d = &mut dst[off + s * kc * nr..off + (s + 1) * kc * nr];
                let j0 = s * nr;
                let w = nr.min(n - j0);
                if trans {
                    for jj in 0..w {
                        let row = &src[(j0 + jj) * ld + k0..(j0 + jj) * ld + k0 + kc];
                        for (p, v) in row.iter().enumerate() {
                            d[p * nr + jj] = *v;
                        }
                    }
                    if w < nr {
                        for p in 0..kc {
                            d[p * nr + w..(p + 1) * nr].fill(0.0);
                        }
                    }
                } else {
                    for p in 0..kc {
                        let row = &src[(k0 + p) * ld + j0..(k0 + p) * ld + j0 + w];
                        d[p * nr..p * nr + w].copy_from_slice(row);
                        d[p * nr + w..(p + 1) * nr].fill(0.0);
                    }
                }
            }
            k0 += kc;
            off += kc * strips * nr;
        }
    }

    /// Packs a weight matrix stored [n][k] (PyTorch's [out][in]).
    pub fn from_weight(w: &[f32], k: usize, n: usize) -> Self {
        let mut p = Self::default();
        p.pack(w, k, k, n, true);
        p
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Act {
    None,
    Gelu,
    Softplus,
    /// SiLU on the columns from this index on.
    SiluFrom(usize),
}

#[derive(Clone, Copy)]
pub struct Epi<'a> {
    /// C += A B instead of C = A B.
    pub accum: bool,
    pub bias: Option<&'a [f32]>,
    pub act: Act,
}

impl Epi<'_> {
    pub const STORE: Epi<'static> = Epi { accum: false, bias: None, act: Act::None };
    pub const ACCUM: Epi<'static> = Epi { accum: true, bias: None, act: Act::None };
}

thread_local! {
    static PACK: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// C[m][n] (+)= A[m][k] B, then bias and activation, on the calling thread.
/// Row i of A is a[i * a_ld..][..k], row i of C is c[i * c_ld..][..n].
pub fn gemm(a: &[f32], a_ld: usize, m: usize, pb: &PackedB, c: &mut [f32], c_ld: usize, epi: Epi) {
    if m == 0 {
        return;
    }
    let kn = kern();
    let (mr, nr, k, n) = (kn.mr, kn.nr, pb.k, pb.n);
    assert!(a.len() >= (m - 1) * a_ld + k && c.len() >= (m - 1) * c_ld + n && a_ld >= k && c_ld >= n);
    if crate::simd::prof::on() {
        FLOPS.fetch_add(2 * (m * n * k) as u64, Ordering::Relaxed);
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if kn.blas {
        // SAFETY: the slices cover the m x k, k x n and m x n matrices (asserted above).
        unsafe {
            cblas_sgemm(
                101, // row major
                111, // no transpose
                111,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                a_ld as i32,
                pb.plain.as_ptr(),
                n as i32,
                if epi.accum { 1.0 } else { 0.0 },
                c.as_mut_ptr(),
                c_ld as i32,
            )
        };
        if epi.bias.is_some() || epi.act != Act::None {
            for i in 0..m {
                crate::simd::epilogue(&mut c[i * c_ld..i * c_ld + n], epi.bias, epi.act);
            }
        }
        return;
    }
    let strips = n.div_ceil(nr);
    let bdata = &pb.buf[pb.off..];
    PACK.with(|pk| {
        let mut pk = pk.borrow_mut();
        let need = kn.mc * kn.kc.max(pb.blocks.iter().map(|b| b.1).max().unwrap_or(0));
        if pk.len() < need {
            pk.resize(need, 0.0);
        }
        let mut tile = [0f32; MAX_TILE];
        for ic in (0..m).step_by(kn.mc) {
            let mcur = kn.mc.min(m - ic);
            let astrips = mcur.div_ceil(mr);
            for (bi, &(k0, kc, boff)) in pb.blocks.iter().enumerate() {
                // Pack A[ic.., k0..] into MR-wide strips: [strip][kc][mr], zero padded.
                for s in 0..astrips {
                    let d = &mut pk[s * kc * mr..(s + 1) * kc * mr];
                    for i in 0..mr {
                        let r = s * mr + i;
                        if r < mcur {
                            let row = &a[(ic + r) * a_ld + k0..(ic + r) * a_ld + k0 + kc];
                            for (p, v) in row.iter().enumerate() {
                                d[p * mr + i] = *v;
                            }
                        } else {
                            for p in 0..kc {
                                d[p * mr + i] = 0.0;
                            }
                        }
                    }
                }
                let add = epi.accum || bi > 0;
                for js in 0..strips {
                    let bp = bdata[boff + js * kc * nr..].as_ptr();
                    let j0 = js * nr;
                    let w = nr.min(n - j0);
                    for s in 0..astrips {
                        // SAFETY: the strips hold kc * mr and kc * nr floats; the tile is mr * nr.
                        unsafe { (kn.micro)(kc, pk[s * kc * mr..].as_ptr(), bp, tile.as_mut_ptr()) };
                        let h = mr.min(mcur - s * mr);
                        for i in 0..h {
                            let dst = &mut c[(ic + s * mr + i) * c_ld + j0..][..w];
                            let src = &tile[i * nr..i * nr + w];
                            if add {
                                for (d, v) in dst.iter_mut().zip(src) {
                                    *d += *v;
                                }
                            } else {
                                dst.copy_from_slice(src);
                            }
                        }
                    }
                }
            }
            if epi.bias.is_some() || epi.act != Act::None {
                for i in ic..ic + mcur {
                    crate::simd::epilogue(&mut c[i * c_ld..i * c_ld + n], epi.bias, epi.act);
                }
            }
        }
    });
}

/// `gemm` split over rows with rayon::join, for callers that are themselves one of few parallel tasks.
#[allow(clippy::too_many_arguments)]
pub fn gemm_par(a: &[f32], a_ld: usize, m: usize, pb: &PackedB, c: &mut [f32], c_ld: usize, epi: Epi, min_rows: usize) {
    if m < 2 * min_rows {
        return gemm(a, a_ld, m, pb, c, c_ld, epi);
    }
    let mid = (m / 2).div_ceil(kern().mr) * kern().mr;
    let (c0, c1) = c.split_at_mut(mid * c_ld);
    rayon::join(
        || gemm_par(a, a_ld, mid, pb, c0, c_ld, epi, min_rows),
        || gemm_par(&a[mid * a_ld..], a_ld, m - mid, pb, c1, c_ld, epi, min_rows),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn gemm_matches_reference() {
        for &(m, k, n) in &[(1usize, 1usize, 1usize), (7, 5, 3), (13, 64, 33), (150, 300, 70), (29, 801, 64), (130, 520, 97)] {
            for trans in [false, true] {
                let a = lcg(m * (k + 3), 1);
                let b = lcg(k * n, 2);
                let bias = lcg(n, 3);
                let mut pb = PackedB::default();
                if trans {
                    let mut bt = vec![0f32; n * k];
                    for p in 0..k {
                        for j in 0..n {
                            bt[j * k + p] = b[p * n + j];
                        }
                    }
                    pb.pack(&bt, k, k, n, true);
                } else {
                    pb.pack(&b, n, k, n, false);
                }
                let mut want = vec![0f64; m * n];
                for i in 0..m {
                    for p in 0..k {
                        for j in 0..n {
                            want[i * n + j] += a[i * (k + 3) + p] as f64 * b[p * n + j] as f64;
                        }
                    }
                }
                let mut c = vec![1f32; m * (n + 2)];
                gemm(&a, k + 3, m, &pb, &mut c, n + 2, Epi { accum: false, bias: Some(&bias), act: Act::None });
                gemm_par(&a, k + 3, m, &pb, &mut c, n + 2, Epi::ACCUM, 8);
                for i in 0..m {
                    for j in 0..n {
                        let w = 2.0 * want[i * n + j] + bias[j] as f64;
                        let g = c[i * (n + 2) + j] as f64;
                        assert!((g - w).abs() < 1e-4 * (1.0 + w.abs()), "m={m} k={k} n={n} trans={trans} [{i},{j}]: {g} vs {w}");
                    }
                    assert_eq!(c[i * (n + 2) + n], 1.0, "wrote past the row");
                }
            }
        }
    }
}
