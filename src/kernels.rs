//! WGSL compute kernels. Each kernel is compiled as its own module.
//! Parameters arrive as a read-only u32 array at binding 7 (floats via bitcast).

pub const PRELUDE: &str = r#"
fn erf_approx(x: f32) -> f32 {
    let s = sign(x);
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}
fn gelu_f(x: f32) -> f32 {
    return 0.5 * x * (1.0 + erf_approx(x * 0.70710678118));
}
"#;

pub const MATMUL: &str = r#"
// C[batch][m][n] (+)= alpha * sum_k A[batch][m][k] * B[batch][k][n]   (B read transposed if b_trans)
// P: m n k flags | a_off a_ld a_bs b_off | b_ld b_bs b_trans c_off | c_ld c_bs alpha(bits) act
// flags: 1 = bias, 2 = accumulate into C ; act: 1 = gelu
// 64x64 output tile per workgroup of 16x16 threads; thread (tx, ty) owns rows ty+16i, cols tx+16j.
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(7) var<storage, read> P: array<u32>;

const LDS: u32 = 65u;
var<workgroup> As: array<f32, 1040>; // [k][m], 16 x 65
var<workgroup> Bs: array<f32, 1040>; // [k][n], 16 x 65

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let M = P[0]; let N = P[1]; let K = P[2]; let flags = P[3];
    let a_off = P[4]; let a_ld = P[5]; let a_bs = P[6]; let b_off = P[7];
    let b_ld = P[8]; let b_bs = P[9]; let b_trans = P[10]; let c_off = P[11];
    let c_ld = P[12]; let c_bs = P[13]; let alpha = bitcast<f32>(P[14]); let act = P[15];
    let tx = lid.x; let ty = lid.y;
    let tid = ty * 16u + tx;
    let row0 = wid.y * 64u; let col0 = wid.x * 64u;
    let batch = wid.z;
    let a_base = a_off + batch * a_bs;
    let b_base = b_off + batch * b_bs;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < K; k0 += 16u) {
        // A tile: element e -> (m = e/16, kk = e%16); stored [kk][m]
        for (var i = 0u; i < 4u; i++) {
            let e = tid + i * 256u;
            let m = e / 16u; let kk = e % 16u;
            let gm = row0 + m; let gk = k0 + kk;
            var av = 0.0;
            if (gm < M && gk < K) { av = A[a_base + gm * a_ld + gk]; }
            As[kk * LDS + m] = av;
        }
        if (b_trans == 1u) {
            for (var i = 0u; i < 4u; i++) {
                let e = tid + i * 256u;
                let n = e / 16u; let kb = e % 16u;
                let gn = col0 + n; let gk = k0 + kb;
                var bv = 0.0;
                if (gn < N && gk < K) { bv = B[b_base + gn * b_ld + gk]; }
                Bs[kb * LDS + n] = bv;
            }
        } else {
            for (var i = 0u; i < 4u; i++) {
                let e = tid + i * 256u;
                let kb = e / 64u; let n = e % 64u;
                let gn = col0 + n; let gk = k0 + kb;
                var bv = 0.0;
                if (gn < N && gk < K) { bv = B[b_base + gk * b_ld + gn]; }
                Bs[kb * LDS + n] = bv;
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk++) {
            let o = kk * LDS;
            let bv = vec4<f32>(Bs[o + tx], Bs[o + tx + 16u], Bs[o + tx + 32u], Bs[o + tx + 48u]);
            let a0 = As[o + ty]; let a1 = As[o + ty + 16u];
            let a2 = As[o + ty + 32u]; let a3 = As[o + ty + 48u];
            acc0 += a0 * bv; acc1 += a1 * bv; acc2 += a2 * bv; acc3 += a3 * bv;
        }
        workgroupBarrier();
    }
    var accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
    for (var i = 0u; i < 4u; i++) {
        let gm = row0 + ty + 16u * i;
        if (gm >= M) { continue; }
        for (var j = 0u; j < 4u; j++) {
            let gn = col0 + tx + 16u * j;
            if (gn < N) {
                var v = accs[i][j] * alpha;
                if ((flags & 1u) != 0u) { v += bias[gn]; }
                if (act == 1u) { v = gelu_f(v); }
                let idx = c_off + batch * c_bs + gm * c_ld + gn;
                if ((flags & 2u) != 0u) { v += C[idx]; }
                C[idx] = v;
            }
        }
    }
}
"#;

pub const LAYERNORM: &str = r#"
// P: rows cols in_off in_ld | out_off out_ld flags(1=pre affine) eps(bits)
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read> pre_scale: array<f32>;
@group(0) @binding(5) var<storage, read> pre_shift: array<f32>;
@group(0) @binding(7) var<storage, read> P: array<u32>;
var<workgroup> red: array<f32, 256>;

fn load(row: u32, c: u32) -> f32 {
    var x = inp[P[2] + row * P[3] + c];
    if ((P[6] & 1u) != 0u) { x = x * (1.0 + pre_scale[c]) + pre_shift[c]; }
    return x;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let rows = P[0]; let cols = P[1];
    let row = wid.y * 65535u + wid.x;
    if (row >= rows) { return; }
    var s = 0.0;
    for (var c = lid.x; c < cols; c += 256u) { s += load(row, c); }
    red[lid.x] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (lid.x < st) { red[lid.x] += red[lid.x + st]; }
        workgroupBarrier();
    }
    let mean = red[0] / f32(cols);
    workgroupBarrier();
    var v = 0.0;
    for (var c = lid.x; c < cols; c += 256u) { let d = load(row, c) - mean; v += d * d; }
    red[lid.x] = v;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (lid.x < st) { red[lid.x] += red[lid.x + st]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(red[0] / f32(cols) + bitcast<f32>(P[7]));
    for (var c = lid.x; c < cols; c += 256u) {
        outp[P[4] + row * P[5] + c] = (load(row, c) - mean) * inv * w[c] + b[c];
    }
}
"#;

pub const SOFTMAX: &str = r#"
// in place over rows. P: rows cols
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(7) var<storage, read> P: array<u32>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let rows = P[0]; let cols = P[1];
    let row = wid.y * 65535u + wid.x;
    if (row >= rows) { return; }
    let base = row * cols;
    var m = -3.0e38;
    for (var c = lid.x; c < cols; c += 256u) { m = max(m, data[base + c]); }
    red[lid.x] = m;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (lid.x < st) { red[lid.x] = max(red[lid.x], red[lid.x + st]); }
        workgroupBarrier();
    }
    let mx = red[0];
    workgroupBarrier();
    var s = 0.0;
    for (var c = lid.x; c < cols; c += 256u) {
        let e = exp(data[base + c] - mx);
        data[base + c] = e;
        s += e;
    }
    red[lid.x] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (lid.x < st) { red[lid.x] += red[lid.x + st]; }
        workgroupBarrier();
    }
    let inv = 1.0 / red[0];
    for (var c = lid.x; c < cols; c += 256u) { data[base + c] = data[base + c] * inv; }
}
"#;

pub const QKV_SPLIT: &str = r#"
// qkv rows -> q (scaled, roped), k (roped), v, each [G][H][T][hd].
// P: G T H hd | gs ts W flags(1=rope) | scale(bits)
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read_write> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read> cs: array<f32>;
@group(0) @binding(5) var<storage, read> sn: array<f32>;
@group(0) @binding(7) var<storage, read> P: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let G = P[0]; let T = P[1]; let H = P[2]; let hd = P[3];
    let gs = P[4]; let ts = P[5]; let W = P[6]; let flags = P[7];
    let scale = bitcast<f32>(P[8]);
    let half = hd / 2u;
    let idx = (wid.y * 65535u + wid.x) * 256u + lid.x;
    if (idx >= G * H * T * half) { return; }
    let i = idx % half;
    var r = idx / half;
    let t = r % T; r = r / T;
    let h = r % H; let g = r / H;
    let row = g * gs + t * ts;
    let base = row * 3u * W + h * hd;
    let q1 = qkv[base + i]; let q2 = qkv[base + i + half];
    let k1 = qkv[base + W + i]; let k2 = qkv[base + W + i + half];
    var qa = q1; var qb = q2; var ka = k1; var kb = k2;
    if ((flags & 1u) != 0u) {
        let c = cs[t * half + i]; let s = sn[t * half + i];
        qa = q1 * c - q2 * s; qb = q1 * s + q2 * c;
        ka = k1 * c - k2 * s; kb = k1 * s + k2 * c;
    }
    let o = ((g * H + h) * T + t) * hd;
    q[o + i] = qa * scale; q[o + i + half] = qb * scale;
    k[o + i] = ka; k[o + i + half] = kb;
    v[o + i] = qkv[base + 2u * W + i]; v[o + i + half] = qkv[base + 2u * W + i + half];
}
"#;

pub const MERGE: &str = r#"
// o [G][H][T][hd] -> out rows [W], optionally removing the component along v.
// P: G T H hd | gs ts W flags(1=exclusive)
@group(0) @binding(0) var<storage, read> o: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(7) var<storage, read> P: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let G = P[0]; let T = P[1]; let H = P[2]; let hd = P[3];
    let gs = P[4]; let ts = P[5]; let W = P[6]; let flags = P[7];
    let idx = (wid.y * 65535u + wid.x) * 256u + lid.x;
    if (idx >= G * T * H) { return; }
    let head = idx % H;
    var r = idx / H;
    let t = r % T; let g = r / T;
    let src = ((g * H + head) * T + t) * hd;
    let dst = (g * gs + t * ts) * W + head * hd;
    var coef = 0.0;
    if ((flags & 1u) != 0u) {
        var dotv = 0.0; var vv = 0.0;
        for (var i = 0u; i < hd; i++) { dotv += o[src + i] * v[src + i]; vv += v[src + i] * v[src + i]; }
        coef = dotv / max(vv, 1e-6);
    }
    for (var i = 0u; i < hd; i++) {
        var val = o[src + i];
        if ((flags & 1u) != 0u) { val = val - coef * v[src + i]; }
        outp[dst + i] = val;
    }
}
"#;

pub const ADD: &str = r#"
// a += b. P: n
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(7) var<storage, read> P: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let idx = (wid.y * 65535u + wid.x) * 256u + lid.x;
    if (idx >= P[0]) { return; }
    a[idx] = a[idx] + b[idx];
}
"#;

pub const CONV_SILU: &str = r#"
// causal depthwise conv over the sequence axis + SiLU, on the first I channels of xz [rows][2I].
// P: G Ts I gs | ts rev K
@group(0) @binding(0) var<storage, read> xz: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(7) var<storage, read> P: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let G = P[0]; let Ts = P[1]; let I = P[2]; let gs = P[3];
    let ts = P[4]; let rev = P[5]; let K = P[6];
    let idx = (wid.y * 65535u + wid.x) * 256u + lid.x;
    if (idx >= G * Ts * I) { return; }
    let ch = idx % I;
    var r = idx / I;
    let t = r % Ts; let g = r / Ts;
    let row = g * gs + t * ts;
    var acc = bias[ch];
    for (var k = 0u; k < K; k++) {
        let off = K - 1u - k;
        var src: i32 = i32(t) - i32(off);
        if (rev == 1u) { src = i32(t) + i32(off); }
        if (src >= 0 && src < i32(Ts)) {
            let rowk = g * gs + u32(src) * ts;
            acc += w[ch * K + k] * xz[rowk * 2u * I + ch];
        }
    }
    outp[row * I + ch] = acc / (1.0 + exp(-acc));
}
"#;

pub const SCAN: &str = r#"
// Selective state-space scan (state size 16), one thread per (group, channel).
// P: G Ts I gs | ts rev proj_ld b_off
@group(0) @binding(0) var<storage, read> u: array<f32>;
@group(0) @binding(1) var<storage, read> pre: array<f32>;
@group(0) @binding(2) var<storage, read> xz: array<f32>;
@group(0) @binding(3) var<storage, read> proj: array<f32>;
@group(0) @binding(4) var<storage, read> A: array<f32>;
@group(0) @binding(5) var<storage, read> D: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(7) var<storage, read> P: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let G = P[0]; let Ts = P[1]; let I = P[2]; let gs = P[3];
    let ts = P[4]; let rev = P[5]; let pld = P[6]; let boff = P[7];
    let idx = (wid.y * 65535u + wid.x) * 64u + lid.x;
    if (idx >= G * I) { return; }
    let c = idx % I;
    let g = idx / I;
    var a: array<f32, 16>;
    var h: array<f32, 16>;
    for (var k = 0u; k < 16u; k++) { a[k] = A[c * 16u + k]; h[k] = 0.0; }
    let dd = D[c];
    for (var i = 0u; i < Ts; i++) {
        var t = i;
        if (rev == 1u) { t = Ts - 1u - i; }
        let row = g * gs + t * ts;
        let pv = pre[row * I + c];
        var step = pv;
        if (pv <= 20.0) { step = log(1.0 + exp(pv)); }
        let x = u[row * I + c];
        let drive = step * x;
        var acc = 0.0;
        for (var k = 0u; k < 16u; k++) {
            h[k] = exp(step * a[k]) * h[k] + drive * proj[row * pld + boff + k];
            acc += h[k] * proj[row * pld + boff + 16u + k];
        }
        acc += x * dd;
        let z = xz[row * 2u * I + I + c];
        y[row * I + c] = acc * (z / (1.0 + exp(-z)));
    }
}
"#;
