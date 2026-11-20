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

/// Shape of the matmul kernel. A workgroup of tx*ty threads computes a
/// (ty*ry) x (tx*rx) output tile; every thread keeps an ry x rx block of
/// accumulators in registers. With `shared` the operands are staged through
/// workgroup memory `ks` columns of K at a time; without it every thread reads
/// its rows of A and columns of B straight from the buffers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MmCfg {
    pub kind: char, // 's' shared scalar, 'v' shared vec4, 'd' direct
    pub tx: u32,
    pub ty: u32,
    pub rx: u32, // multiple of 4
    pub ry: u32,
    pub ks: u32,
    pub fma: bool,
}

impl MmCfg {
    pub const DEFAULT: MmCfg = MmCfg { kind: 's', tx: 16, ty: 16, rx: 4, ry: 4, ks: 16, fma: false };
    /// Kernel variants for a GPU vendor (PCI id); the dispatcher picks per matrix shape.
    pub fn defaults(_vendor: u32) -> Vec<MmCfg> {
        let s = |tx, ty| MmCfg { kind: 's', tx, ty, rx: 8, ry: 8, ks: 16, fma: false };
        // 128x128 tiles for the big products, narrower ones for N = 64 and the 46x46 band attention.
        vec![s(16, 16), s(8, 16), s(8, 8)]
    }
    pub fn tile_m(&self) -> u32 {
        self.ty * self.ry
    }
    pub fn tile_n(&self) -> u32 {
        self.tx * self.rx
    }
    /// "s|d,tx,ty,rx,ry,ks[,fma]"
    pub fn parse(s: &str) -> Option<MmCfg> {
        let p: Vec<&str> = s.split(',').collect();
        if p.len() < 6 {
            return None;
        }
        let n = |i: usize| p[i].trim().parse::<u32>().ok();
        let c = MmCfg { kind: p[0].trim().chars().next()?, tx: n(1)?, ty: n(2)?, rx: n(3)?, ry: n(4)?, ks: n(5)?, fma: p.len() > 6 && p[6].trim() == "fma" };
        let ok = c.tx > 0 && c.ty > 0 && c.ry > 0 && c.ks > 0 && c.rx >= 4 && c.rx % 4 == 0 && c.tx * c.ty <= 256
            && "svd".contains(c.kind) && (c.kind != 'v' || c.ry % 4 == 0) && (c.kind == 'd' || c.ks * (c.tile_m() + c.tile_n() + 2) * 4 <= 32768);
        ok.then_some(c)
    }
}

/// C[batch][m][n] (+)= sum_k A[batch][m][k] * B[batch][k][n]   (B read transposed if b_trans)
/// P: m n k flags | a_off a_ld a_bs b_off | b_ld b_bs b_trans c_off | c_ld c_bs - act
/// flags: 1 = bias, 2 = accumulate into C, 4 = scale row m of batch z by bias[z * M + m] ; act: 1 = gelu
pub fn matmul(c: MmCfg) -> String {
    use std::fmt::Write;
    let (tx, ty, rx, ry, ks) = (c.tx, c.ty, c.rx, c.ry, c.ks);
    let (tm, tn, nt) = (c.tile_m(), c.tile_n(), tx * ty);
    let (lda, ldb) = (tm + 1, tn + 1);
    let nv = rx / 4;
    let mut s = String::new();
    s.push_str(
        r#"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(7) var<storage, read> P: array<u32>;
"#,
    );
    if c.kind == 's' {
        let _ = writeln!(s, "var<workgroup> As: array<f32, {}>;\nvar<workgroup> Bs: array<f32, {}>;", ks * lda, ks * ldb);
    } else if c.kind == 'v' {
        let _ = writeln!(s, "var<workgroup> As: array<vec4<f32>, {}>;\nvar<workgroup> Bs: array<vec4<f32>, {}>;", ks * tm / 4, ks * tn / 4);
    }
    let _ = writeln!(s, "@compute @workgroup_size({tx}, {ty})");
    s.push_str(
        r#"fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let M = P[0]; let N = P[1]; let K = P[2]; let flags = P[3];
    let a_off = P[4]; let a_ld = P[5]; let a_bs = P[6]; let b_off = P[7];
    let b_ld = P[8]; let b_bs = P[9]; let b_trans = P[10]; let c_off = P[11];
    let c_ld = P[12]; let c_bs = P[13]; let act = P[15];
    let tx = lid.x; let ty = lid.y;
    let batch = wid.z;
    let a_base = a_off + batch * a_bs;
    let b_base = b_off + batch * b_bs;
"#,
    );
    let _ = writeln!(s, "    let tid = ty * {tx}u + tx;\n    let row0 = wid.y * {tm}u; let col0 = wid.x * {tn}u;");
    for i in 0..ry {
        for j in 0..nv {
            let _ = writeln!(s, "    var acc_{i}_{j} = vec4<f32>(0.0);");
        }
    }
    let upd = |s: &mut String, ind: &str| {
        for i in 0..ry {
            for j in 0..nv {
                if c.fma {
                    let _ = writeln!(s, "{ind}acc_{i}_{j} = fma(vec4<f32>(a{i}), b{j}, acc_{i}_{j});");
                } else {
                    let _ = writeln!(s, "{ind}acc_{i}_{j} += a{i} * b{j};");
                }
            }
        }
    };
    if c.kind == 'v' {
        // Workgroup memory holds vec4s: slot (kk, m4) of As is A[row0 + 4*m4 .. +4][k0 + kk], and
        // likewise for Bs, so a thread fetches its 8 operands of a k step with 4 loads.
        let (ea, eb) = (tm / 4 * ks, tn / 4 * ks);
        let (la, lb) = (ea.div_ceil(nt), eb.div_ceil(nt));
        let (qa, qb) = (tm / 4, tn / 4);
        let ga = if ea % nt != 0 { format!("            if (e >= {ea}u) {{ break; }}\n") } else { String::new() };
        let gb = if eb % nt != 0 { format!("                if (e >= {eb}u) {{ break; }}\n") } else { String::new() };
        let _ = writeln!(s, "    let rmax = M - 1u; let cmax = N - 1u;\n    for (var k0 = 0u; k0 < K; k0 += {ks}u) {{");
        let _ = writeln!(
            s,
            "        for (var i = 0u; i < {la}u; i++) {{\n            let e = tid + i * {nt}u;\n{ga}            let m4 = e / {ks}u; let kk = e % {ks}u;\n            let gm = row0 + m4 * 4u; let gk = k0 + kk;\n            var av = vec4<f32>(0.0);\n            if (gk < K) {{\n                av = vec4<f32>(A[a_base + min(gm, rmax) * a_ld + gk], A[a_base + min(gm + 1u, rmax) * a_ld + gk], A[a_base + min(gm + 2u, rmax) * a_ld + gk], A[a_base + min(gm + 3u, rmax) * a_ld + gk]);\n            }}\n            As[kk * {qa}u + m4] = av;\n        }}"
        );
        let _ = writeln!(
            s,
            "        if (b_trans == 1u) {{\n            for (var i = 0u; i < {lb}u; i++) {{\n                let e = tid + i * {nt}u;\n{gb}                let n4 = e / {ks}u; let kb = e % {ks}u;\n                let gn = col0 + n4 * 4u; let gk = k0 + kb;\n                var bv = vec4<f32>(0.0);\n                if (gk < K) {{\n                    bv = vec4<f32>(B[b_base + min(gn, cmax) * b_ld + gk], B[b_base + min(gn + 1u, cmax) * b_ld + gk], B[b_base + min(gn + 2u, cmax) * b_ld + gk], B[b_base + min(gn + 3u, cmax) * b_ld + gk]);\n                }}\n                Bs[kb * {qb}u + n4] = bv;\n            }}\n        }} else {{\n            for (var i = 0u; i < {lb}u; i++) {{\n                let e = tid + i * {nt}u;\n{gb}                let kb = e / {qb}u; let n4 = e % {qb}u;\n                let gn = col0 + n4 * 4u; let gk = k0 + kb;\n                var bv = vec4<f32>(0.0);\n                if (gk < K) {{\n                    let o = b_base + gk * b_ld;\n                    bv = vec4<f32>(B[o + min(gn, cmax)], B[o + min(gn + 1u, cmax)], B[o + min(gn + 2u, cmax)], B[o + min(gn + 3u, cmax)]);\n                }}\n                Bs[kb * {qb}u + n4] = bv;\n            }}\n        }}\n        workgroupBarrier();"
        );
        let _ = writeln!(s, "        for (var kk = 0u; kk < {ks}u; kk++) {{\n            let oa = kk * {qa}u + ty * {}u; let ob = kk * {qb}u + tx * {}u;", ry / 4, nv);
        for j in 0..nv {
            let _ = writeln!(s, "            let b{j} = Bs[ob + {j}u];");
        }
        for i in 0..ry / 4 {
            let _ = writeln!(s, "            let av{i} = As[oa + {i}u];");
        }
        for i in 0..ry {
            let _ = writeln!(s, "            let a{i} = av{}[{}];", i / 4, i % 4);
        }
        upd(&mut s, "            ");
        s.push_str("        }\n        workgroupBarrier();\n    }\n");
    } else if c.kind == 's' {
        let (ea, eb) = (tm * ks, tn * ks);
        let (la, lb) = (ea.div_ceil(nt), eb.div_ceil(nt));
        let _ = writeln!(s, "    for (var k0 = 0u; k0 < K; k0 += {ks}u) {{");
        let _ = writeln!(s, "        for (var i = 0u; i < {la}u; i++) {{\n            let e = tid + i * {nt}u;");
        if ea % nt != 0 {
            let _ = writeln!(s, "            if (e >= {ea}u) {{ break; }}");
        }
        let _ = writeln!(
            s,
            "            let m = e / {ks}u; let kk = e % {ks}u;\n            let gm = row0 + m; let gk = k0 + kk;\n            var av = 0.0;\n            if (gm < M && gk < K) {{ av = A[a_base + gm * a_ld + gk]; }}\n            As[kk * {lda}u + m] = av;\n        }}"
        );
        let guard = if eb % nt != 0 { format!("                if (e >= {eb}u) {{ break; }}\n") } else { String::new() };
        let _ = writeln!(
            s,
            "        if (b_trans == 1u) {{\n            for (var i = 0u; i < {lb}u; i++) {{\n                let e = tid + i * {nt}u;\n{guard}                let n = e / {ks}u; let kb = e % {ks}u;\n                let gn = col0 + n; let gk = k0 + kb;\n                var bv = 0.0;\n                if (gn < N && gk < K) {{ bv = B[b_base + gn * b_ld + gk]; }}\n                Bs[kb * {ldb}u + n] = bv;\n            }}\n        }} else {{\n            for (var i = 0u; i < {lb}u; i++) {{\n                let e = tid + i * {nt}u;\n{guard}                let kb = e / {tn}u; let n = e % {tn}u;\n                let gn = col0 + n; let gk = k0 + kb;\n                var bv = 0.0;\n                if (gn < N && gk < K) {{ bv = B[b_base + gk * b_ld + gn]; }}\n                Bs[kb * {ldb}u + n] = bv;\n            }}\n        }}\n        workgroupBarrier();"
        );
        let _ = writeln!(s, "        for (var kk = 0u; kk < {ks}u; kk++) {{\n            let oa = kk * {lda}u + ty; let ob = kk * {ldb}u + tx;");
        for j in 0..nv {
            let e: Vec<String> = (0..4).map(|q| format!("Bs[ob + {}u]", (j * 4 + q) * tx)).collect();
            let _ = writeln!(s, "            let b{j} = vec4<f32>({});", e.join(", "));
        }
        for i in 0..ry {
            let _ = writeln!(s, "            let a{i} = As[oa + {}u];", i * ty);
        }
        upd(&mut s, "            ");
        s.push_str("        }\n        workgroupBarrier();\n    }\n");
    } else {
        // Rows/columns past the edge are clamped: they compute values that are never stored.
        for i in 0..ry {
            let _ = writeln!(s, "    let ra{i} = a_base + min(row0 + ty * {ry}u + {i}u, M - 1u) * a_ld;");
        }
        s.push_str("    if (b_trans == 1u) {\n");
        for j in 0..rx {
            let _ = writeln!(s, "        let cb{j} = b_base + min(col0 + tx * {rx}u + {j}u, N - 1u) * b_ld;");
        }
        s.push_str("        for (var k = 0u; k < K; k++) {\n");
        for j in 0..nv {
            let e: Vec<String> = (0..4).map(|q| format!("B[cb{} + k]", j * 4 + q)).collect();
            let _ = writeln!(s, "            let b{j} = vec4<f32>({});", e.join(", "));
        }
        for i in 0..ry {
            let _ = writeln!(s, "            let a{i} = A[ra{i} + k];");
        }
        upd(&mut s, "            ");
        s.push_str("        }\n    } else {\n");
        for j in 0..rx {
            let _ = writeln!(s, "        let cb{j} = b_base + min(col0 + tx * {rx}u + {j}u, N - 1u);");
        }
        s.push_str("        for (var k = 0u; k < K; k++) {\n            let kb = k * b_ld;\n");
        for j in 0..nv {
            let e: Vec<String> = (0..4).map(|q| format!("B[cb{} + kb]", j * 4 + q)).collect();
            let _ = writeln!(s, "            let b{j} = vec4<f32>({});", e.join(", "));
        }
        for i in 0..ry {
            let _ = writeln!(s, "            let a{i} = A[ra{i} + k];");
        }
        upd(&mut s, "            ");
        s.push_str("        }\n    }\n");
    }
    for i in 0..ry {
        let gm = if c.kind == 's' { format!("row0 + ty + {}u", i * ty) } else { format!("row0 + ty * {ry}u + {i}u") };
        let _ = writeln!(s, "    {{\n        let gm = {gm};\n        if (gm < M) {{");
        for j in 0..rx {
            let gn = if c.kind == 's' { format!("col0 + tx + {}u", j * tx) } else { format!("col0 + tx * {rx}u + {j}u") };
            let _ = writeln!(
                s,
                "            {{\n                let gn = {gn};\n                if (gn < N) {{\n                    var v = acc_{i}_{}[{}];\n                    if ((flags & 4u) != 0u) {{ v *= bias[batch * M + gm]; }}\n                    if ((flags & 1u) != 0u) {{ v += bias[gn]; }}\n                    if (act == 1u) {{ v = gelu_f(v); }}\n                    let idx = c_off + batch * c_bs + gm * c_ld + gn;\n                    if ((flags & 2u) != 0u) {{ v += C[idx]; }}\n                    C[idx] = v;\n                }}\n            }}",
                j / 4,
                j % 4
            );
        }
        s.push_str("        }\n    }\n");
    }
    s.push_str("}\n");
    s
}

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
// data <- exp(data - rowmax) in place, inv[row] <- 1 / rowsum; the division is applied by the
// matmul that consumes the scores (flag 4). P: rows cols
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> inv: array<f32>;
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
    if (lid.x == 0u) { inv[row] = 1.0 / red[0]; }
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
