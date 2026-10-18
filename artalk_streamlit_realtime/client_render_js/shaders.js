// WGSL kernels for an N-channel 3D Gaussian tile rasterizer.
//
// Written from the published algorithm (Kerbl et al., "3D Gaussian Splatting
// for Real-Time Radiance Field Rendering", SIGGRAPH 2023, and the EWA
// splatting derivation it builds on): project each Gaussian to a 2D
// covariance, bin it into 16x16 tiles, sort (tile, depth) keys, then
// composite front to back per tile. Two design choices come from permissive
// references:
//  - the portable 4-bit, 8-pass radix sort with a separate block-sum scan
//    follows the structure of PlayCanvas's `ComputeRadixSortMultipass`
//    (MIT, (c) PlayCanvas Ltd.; host side:
//    https://github.com/playcanvas/engine/blob/342ec1a04585928918d60fea1af900fbfec3cb7a/src/scene/graphics/radix-sort/compute-radix-sort-multipass.js,
//    WGSL kernels:
//    https://github.com/playcanvas/engine/tree/342ec1a04585928918d60fea1af900fbfec3cb7a/src/scene/shader-lib/wgsl/chunks/radix-sort),
//    which in turn credits kishimisu's WebGPU-Radix-Sort; the local ranking
//    here uses packed 16-bit per-digit counters instead of its bitmasks.
//  - the per-splat Mahalanobis threshold ln(255 * opacity) and the
//    tile-vs-ellipse intersection test are patterned after Brush
//    (Apache-2.0, by Arthur Brussee,
//    https://github.com/ArthurBrussee/brush/blob/1388f74c6fe0236f68ee4915564bf00e9d2e3747/crates/brush-render/src/kernels/helpers.rs,
//    `compute_bbox_extent` and `will_primitive_contribute`, itself after
//    StopThePop); the edge-minimum test below is a from-scratch 4-edge version.
// Compositing conventions that the CUDA reference also documents in its
// paper (0.3 px covariance dilation, alpha clamp 0.99, 1/255 alpha cutoff,
// transmittance stop at 1e-4) are reproduced so the output matches it.

export const TILE = 16;
export const SCAN_BLOCK = 1024; // 256 threads x 4 counts
export const SORT_THREADS = 256;
export const SORT_KEYS_PER_THREAD = 8;
export const SORT_BLOCK = SORT_THREADS * SORT_KEYS_PER_THREAD;
export const RADIX_PASSES = 8;

export const PARAMS_BYTES = 128;

const common = /* wgsl */ `
struct Params {
  view: mat4x4<f32>,
  fx: f32, fy: f32, cx: f32, cy: f32,
  width: u32, height: u32, tiles_x: u32, tiles_y: u32,
  count: u32, depth_bits: u32, pair_capacity: u32, tile_test: u32,
  tan_fov_x: f32, tan_fov_y: f32, scan_blocks: u32, pad1: u32,
}

struct Proj {
  x: f32, y: f32, ca: f32, cb: f32, cc: f32, op: f32, depth: f32,
  ex: f32, ey: f32,
  x0: u32, y0: u32, x1: u32, y1: u32,
  pad: u32,
}

struct Counters {
  pairs: u32, depth_min: u32, depth_max: u32, overflow: u32,
}

struct AtomicCounters {
  pairs: atomic<u32>, depth_min: atomic<u32>, depth_max: atomic<u32>, overflow: atomic<u32>,
}
`;

// Minimum of 0.5 * d^T C d along the segment a->b, d = p - m. The quadratic
// along the segment is q(s) = Q(u) + 2 s B(u, v) + s^2 Q(v), minimised at
// the clamped vertex.
const tileTest = /* wgsl */ `
fn edge_min(a: vec2f, b: vec2f, m: vec2f, ca: f32, cb: f32, cc: f32) -> f32 {
  let u = a - m;
  let v = b - a;
  let qu = ca * u.x * u.x + 2.0 * cb * u.x * u.y + cc * u.y * u.y;
  let qv = ca * v.x * v.x + 2.0 * cb * v.x * v.y + cc * v.y * v.y;
  let buv = ca * u.x * v.x + cb * (u.x * v.y + u.y * v.x) + cc * u.y * v.y;
  let s = clamp(-buv / max(qv, 1e-12), 0.0, 1.0);
  return 0.5 * (qu + 2.0 * s * buv + s * s * qv);
}

// Does any pixel centre of tile (tx, ty) have 0.5 * d^T C d <= tau?
fn tile_hit(tx: u32, ty: u32, m: vec2f, ca: f32, cb: f32, cc: f32, tau: f32) -> bool {
  let x0 = f32(tx * ${TILE}u);
  let y0 = f32(ty * ${TILE}u);
  let x1 = x0 + ${TILE - 1}.0;
  let y1 = y0 + ${TILE - 1}.0;
  if (m.x >= x0 && m.x <= x1 && m.y >= y0 && m.y <= y1) { return true; }
  let e0 = edge_min(vec2f(x0, y0), vec2f(x1, y0), m, ca, cb, cc);
  let e1 = edge_min(vec2f(x1, y0), vec2f(x1, y1), m, ca, cb, cc);
  let e2 = edge_min(vec2f(x0, y1), vec2f(x1, y1), m, ca, cb, cc);
  let e3 = edge_min(vec2f(x0, y0), vec2f(x0, y1), m, ca, cb, cc);
  return min(min(e0, e1), min(e2, e3)) <= tau;
}
`;

export const projectShader = /* wgsl */ `
${common}
${tileTest}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> cov: array<f32>;
@group(0) @binding(3) var<storage, read> opac: array<f32>;
@group(0) @binding(4) var<storage, read_write> proj: array<Proj>;
@group(0) @binding(5) var<storage, read_write> tile_counts: array<u32>;
@group(0) @binding(6) var<storage, read_write> counters: AtomicCounters;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.count) { return; }
  tile_counts[i] = 0u;
  let p = vec3f(pos[3u * i], pos[3u * i + 1u], pos[3u * i + 2u]);
  let t = (P.view * vec4f(p, 1.0)).xyz;
  if (t.z < 0.2) { return; }
  let op = opac[i];
  if (P.tile_test == 1u && op < 1.0 / 255.0) { return; }

  let limx = 1.3 * P.tan_fov_x;
  let limy = 1.3 * P.tan_fov_y;
  let txtz = clamp(t.x / t.z, -limx, limx) * t.z;
  let tytz = clamp(t.y / t.z, -limy, limy) * t.z;
  let tz2 = t.z * t.z;
  let J = mat3x3f(
    vec3f(P.fx / t.z, 0.0, 0.0),
    vec3f(0.0, P.fy / t.z, 0.0),
    vec3f(-P.fx * txtz / tz2, -P.fy * tytz / tz2, 0.0));
  let W = mat3x3f(P.view[0].xyz, P.view[1].xyz, P.view[2].xyz);
  let T = J * W;
  let c0 = cov[6u * i]; let c1 = cov[6u * i + 1u]; let c2 = cov[6u * i + 2u];
  let c3 = cov[6u * i + 3u]; let c4 = cov[6u * i + 4u]; let c5 = cov[6u * i + 5u];
  let S = mat3x3f(vec3f(c0, c1, c2), vec3f(c1, c3, c4), vec3f(c2, c4, c5));
  let C2 = T * S * transpose(T);
  let a = C2[0][0] + 0.3;
  let b = C2[1][0];
  let c = C2[1][1] + 0.3;
  let det = a * c - b * b;
  if (det == 0.0) { return; }
  let inv = 1.0 / det;
  let ca = c * inv;
  let cb = -b * inv;
  let cc = a * inv;
  let m = vec2f(P.fx * (t.x / t.z) + P.cx, P.fy * (t.y / t.z) + P.cy);

  var x0: u32; var y0: u32; var x1: u32; var y1: u32;
  var count = 0u;
  var ex = 0.0; var ey = 0.0;
  let tx_f = f32(P.tiles_x);
  let ty_f = f32(P.tiles_y);
  if (P.tile_test == 0u) {
    let mid = 0.5 * (a + c);
    let l1 = mid + sqrt(max(0.1, mid * mid - det));
    let r = ceil(3.0 * sqrt(l1));
    ex = r; ey = r;
    x0 = u32(clamp(floor((m.x - r) / ${TILE}.0), 0.0, tx_f));
    x1 = u32(clamp(floor((m.x + r + ${TILE - 1}.0) / ${TILE}.0), 0.0, tx_f));
    y0 = u32(clamp(floor((m.y - r) / ${TILE}.0), 0.0, ty_f));
    y1 = u32(clamp(floor((m.y + r + ${TILE - 1}.0) / ${TILE}.0), 0.0, ty_f));
    count = select(0u, (x1 - x0) * (y1 - y0), x1 > x0 && y1 > y0);
  } else {
    let tau = log(255.0 * op);
    ex = sqrt(2.0 * tau * a);
    ey = sqrt(2.0 * tau * c);
    x0 = u32(clamp(floor((m.x - ex) / ${TILE}.0), 0.0, tx_f));
    x1 = u32(clamp(floor((m.x + ex) / ${TILE}.0) + 1.0, 0.0, tx_f));
    y0 = u32(clamp(floor((m.y - ey) / ${TILE}.0), 0.0, ty_f));
    y1 = u32(clamp(floor((m.y + ey) / ${TILE}.0) + 1.0, 0.0, ty_f));
    for (var ty = y0; ty < y1; ty++) {
      for (var tx = x0; tx < x1; tx++) {
        if (tile_hit(tx, ty, m, ca, cb, cc, tau)) { count++; }
      }
    }
  }
  if (count == 0u) { return; }
  proj[i] = Proj(m.x, m.y, ca, cb, cc, op, t.z, ex, ey, x0, y0, x1, y1, 0u);
  tile_counts[i] = count;
  let dbits = bitcast<u32>(t.z);
  atomicMin(&counters.depth_min, dbits);
  atomicMax(&counters.depth_max, dbits);
}
`;

// Exclusive prefix within blocks of SCAN_BLOCK counts, plus one sum per block.
export const scanLocalShader = /* wgsl */ `
${common}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> tile_counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> prefix: array<u32>;
@group(0) @binding(3) var<storage, read_write> block_sums: array<u32>;
var<workgroup> sh: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3u) {
  let base = wid.x * ${SCAN_BLOCK}u + li * 4u;
  var c: array<u32, 4>;
  var sum = 0u;
  for (var j = 0u; j < 4u; j++) {
    let idx = base + j;
    c[j] = select(0u, tile_counts[idx], idx < P.count);
    sum += c[j];
  }
  sh[li] = sum;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off <<= 1u) {
    var v = 0u;
    if (li >= off) { v = sh[li - off]; }
    workgroupBarrier();
    sh[li] += v;
    workgroupBarrier();
  }
  var run = sh[li] - sum;
  for (var j = 0u; j < 4u; j++) {
    let idx = base + j;
    if (idx < P.count) { prefix[idx] = run; }
    run += c[j];
  }
  if (li == 255u) { block_sums[wid.x] = sh[255u]; }
}
`;

// Single-workgroup exclusive scan of `len` values (each thread owns a
// contiguous segment), used for the block sums and the radix histograms.
function singleScan({ lenExpr, extraBindings, epilogue }) {
  return /* wgsl */ `
${common}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
${extraBindings}
var<workgroup> sh: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  let len = ${lenExpr};
  let per = (len + 255u) / 256u;
  let begin = li * per;
  let stop = min(begin + per, len);
  var sum = 0u;
  for (var k = begin; k < stop; k++) { sum += src[k]; }
  sh[li] = sum;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off <<= 1u) {
    var v = 0u;
    if (li >= off) { v = sh[li - off]; }
    workgroupBarrier();
    sh[li] += v;
    workgroupBarrier();
  }
  var run = sh[li] - sum;
  for (var k = begin; k < stop; k++) {
    let v = src[k];
    dst[k] = run;
    run += v;
  }
  ${epilogue}
}
`;
}

export const scanBlocksShader = singleScan({
  lenExpr: 'P.scan_blocks',
  extraBindings: /* wgsl */ `
@group(0) @binding(3) var<storage, read_write> counters: Counters;
@group(0) @binding(4) var<storage, read_write> indirect: array<u32>;`,
  epilogue: /* wgsl */ `
  if (li == 255u) {
    let total = sh[255u];
    let n = min(total, P.pair_capacity);
    counters.pairs = n;
    counters.overflow = select(0u, total, total > P.pair_capacity);
    indirect[0] = (n + ${SORT_BLOCK - 1}u) / ${SORT_BLOCK}u;
    indirect[1] = 1u;
    indirect[2] = 1u;
    indirect[3] = (n + 255u) / 256u;
    indirect[4] = 1u;
    indirect[5] = 1u;
  }`,
});

export const scanHistShader = singleScan({
  lenExpr: '16u * ((counters.pairs + ' + (SORT_BLOCK - 1) + 'u) / ' + SORT_BLOCK + 'u)',
  extraBindings: /* wgsl */ `
@group(0) @binding(3) var<storage, read> counters: Counters;`,
  epilogue: '',
});

export const emitShader = /* wgsl */ `
${common}
${tileTest}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> proj: array<Proj>;
@group(0) @binding(2) var<storage, read> tile_counts: array<u32>;
@group(0) @binding(3) var<storage, read> prefix: array<u32>;
@group(0) @binding(4) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(5) var<storage, read> counters: Counters;
@group(0) @binding(6) var<storage, read_write> keys: array<u32>;
@group(0) @binding(7) var<storage, read_write> vals: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.count) { return; }
  let count = tile_counts[i];
  if (count == 0u) { return; }
  let base = prefix[i] + block_offsets[i / ${SCAN_BLOCK}u];
  let s = proj[i];
  let zmin = bitcast<f32>(counters.depth_min);
  let zmax = bitcast<f32>(counters.depth_max);
  let qmax = f32((1u << P.depth_bits) - 1u);
  let q = u32(clamp((s.depth - zmin) / max(zmax - zmin, 1e-9), 0.0, 1.0) * qmax);
  let m = vec2f(s.x, s.y);
  let tau = log(255.0 * s.op);
  var n = 0u;
  for (var ty = s.y0; ty < s.y1; ty++) {
    for (var tx = s.x0; tx < s.x1; tx++) {
      if (P.tile_test == 1u && !tile_hit(tx, ty, m, s.ca, s.cb, s.cc, tau)) { continue; }
      let k = base + n;
      n++;
      if (k < P.pair_capacity) {
        keys[k] = ((ty * P.tiles_x + tx) << P.depth_bits) | q;
        vals[k] = i;
      }
    }
  }
}
`;

export function upsweepShader(shift) {
  return /* wgsl */ `
${common}
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> hist: array<u32>;
@group(0) @binding(2) var<storage, read> counters: Counters;
var<workgroup> h: array<atomic<u32>, 16>;

@compute @workgroup_size(${SORT_THREADS})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3u) {
  if (li < 16u) { atomicStore(&h[li], 0u); }
  workgroupBarrier();
  let num = counters.pairs;
  let base = wid.x * ${SORT_BLOCK}u + li * ${SORT_KEYS_PER_THREAD}u;
  for (var j = 0u; j < ${SORT_KEYS_PER_THREAD}u; j++) {
    let idx = base + j;
    if (idx < num) {
      atomicAdd(&h[(keys[idx] >> ${shift}u) & 15u], 1u);
    }
  }
  workgroupBarrier();
  if (li < 16u) {
    let nblocks = (num + ${SORT_BLOCK - 1}u) / ${SORT_BLOCK}u;
    hist[li * nblocks + wid.x] = atomicLoad(&h[li]);
  }
}
`;
}

// Stable scatter: each thread ranks its 8 consecutive keys per digit in
// registers (two 16-bit lanes per u32, so eight words hold 16 digits), the
// workgroup scans those packed words across threads, and the destination is
// the scanned global histogram plus the local rank.
export function downsweepShader(shift) {
  return /* wgsl */ `
${common}
@group(0) @binding(0) var<storage, read> keys_in: array<u32>;
@group(0) @binding(1) var<storage, read> vals_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> vals_out: array<u32>;
@group(0) @binding(4) var<storage, read> hist_scan: array<u32>;
@group(0) @binding(5) var<storage, read> counters: Counters;
var<workgroup> sc: array<array<u32, 8>, ${SORT_THREADS}>;

@compute @workgroup_size(${SORT_THREADS})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3u) {
  let num = counters.pairs;
  let nblocks = (num + ${SORT_BLOCK - 1}u) / ${SORT_BLOCK}u;
  let base = wid.x * ${SORT_BLOCK}u + li * ${SORT_KEYS_PER_THREAD}u;
  var k: array<u32, ${SORT_KEYS_PER_THREAD}>;
  var d: array<u32, ${SORT_KEYS_PER_THREAD}>;
  var rank: array<u32, ${SORT_KEYS_PER_THREAD}>;
  var cnt: array<u32, 8>;
  for (var w = 0u; w < 8u; w++) { cnt[w] = 0u; }
  for (var j = 0u; j < ${SORT_KEYS_PER_THREAD}u; j++) {
    let idx = base + j;
    if (idx < num) {
      k[j] = keys_in[idx];
      d[j] = (k[j] >> ${shift}u) & 15u;
      let w = d[j] >> 1u;
      let sh = (d[j] & 1u) * 16u;
      rank[j] = (cnt[w] >> sh) & 0xffffu;
      cnt[w] += 1u << sh;
    }
  }
  for (var w = 0u; w < 8u; w++) { sc[li][w] = cnt[w]; }
  workgroupBarrier();
  for (var off = 1u; off < ${SORT_THREADS}u; off <<= 1u) {
    var v: array<u32, 8>;
    for (var w = 0u; w < 8u; w++) { v[w] = 0u; }
    if (li >= off) {
      for (var w = 0u; w < 8u; w++) { v[w] = sc[li - off][w]; }
    }
    workgroupBarrier();
    for (var w = 0u; w < 8u; w++) { sc[li][w] += v[w]; }
    workgroupBarrier();
  }
  for (var j = 0u; j < ${SORT_KEYS_PER_THREAD}u; j++) {
    let idx = base + j;
    if (idx < num) {
      let w = d[j] >> 1u;
      let sh = (d[j] & 1u) * 16u;
      let excl = ((sc[li][w] - cnt[w]) >> sh) & 0xffffu;
      let dest = hist_scan[d[j] * nblocks + wid.x] + excl + rank[j];
      keys_out[dest] = k[j];
      vals_out[dest] = vals_in[idx];
    }
  }
}
`;
}

export const rangesShader = /* wgsl */ `
${common}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read> counters: Counters;
@group(0) @binding(3) var<storage, read_write> ranges: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  let num = counters.pairs;
  if (j >= num) { return; }
  let t = keys[j] >> P.depth_bits;
  if (j == 0u || (keys[j - 1u] >> P.depth_bits) != t) { ranges[2u * t] = j; }
  if (j + 1u == num || (keys[j + 1u] >> P.depth_bits) != t) { ranges[2u * t + 1u] = j + 1u; }
}
`;

// channels: feature channels composited; outputF16: pack pixel pairs as f16.
export function rasterizeShader({ channels, outputF16 }) {
  const words = Math.ceil(channels / 2);
  const accum = [];
  for (let w = 0; w < words; w++) {
    accum.push(`        { let v = unpack2x16float(feat[fb + ${w}u]); acc[${2 * w}] += v.x * wgt;${2 * w + 1 < channels ? ` acc[${2 * w + 1}] += v.y * wgt;` : ''} }`);
  }
  const zero = [];
  for (let c = 0; c < channels; c++) zero.push(`  acc[${c}] = 0.0;`);
  let write;
  if (!outputF16) {
    const lines = [];
    for (let c = 0; c < channels; c++) lines.push(`    image[${c}u * plane + pix] = acc[${c}];`);
    write = `  if (inside) {\n${lines.join('\n')}\n  }`;
  } else {
    // Odd columns hand their accumulators to the even neighbour through
    // workgroup memory, 16 channels at a time, so each thread pair writes
    // one packed u32 per channel.
    const chunks = [];
    for (let c0 = 0; c0 < channels; c0 += 16) {
      const n = Math.min(16, channels - c0);
      const put = [], pack = [];
      for (let c = 0; c < n; c++) {
        put.push(`    xfer[(li >> 1u) * 16u + ${c}u] = acc[${c0 + c}];`);
        pack.push(`    image[(${c0 + c}u * plane + pix) >> 1u] = pack2x16float(vec2f(acc[${c0 + c}], xfer[(li >> 1u) * 16u + ${c}u]));`);
      }
      chunks.push(`  workgroupBarrier();\n  if ((li & 1u) == 1u) {\n${put.join('\n')}\n  }\n  workgroupBarrier();\n  if ((li & 1u) == 0u && inside) {\n${pack.join('\n')}\n  }`);
    }
    write = chunks.join('\n');
  }
  return /* wgsl */ `
${common}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> proj: array<Proj>;
@group(0) @binding(2) var<storage, read> vals: array<u32>;
@group(0) @binding(3) var<storage, read> ranges: array<u32>;
@group(0) @binding(4) var<storage, read> feat: array<u32>;
@group(0) @binding(5) var<storage, read_write> image: array<${outputF16 ? 'u32' : 'f32'}>;
var<workgroup> s_a: array<vec4f, 256>;
var<workgroup> s_b: array<vec2f, 256>;
var<workgroup> s_id: array<u32, 256>;
var<workgroup> done_count: atomic<u32>;
var<workgroup> done_flag: u32;
${outputF16 ? 'var<workgroup> xfer: array<f32, 2048>;' : ''}

@compute @workgroup_size(${TILE}, ${TILE}, 1)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3u) {
  let tile = wid.y * P.tiles_x + wid.x;
  let px = wid.x * ${TILE}u + lid.x;
  let py = wid.y * ${TILE}u + lid.y;
  let pixf = vec2f(f32(px), f32(py));
  let inside = px < P.width && py < P.height;
  var done = !inside;
  if (li == 0u) { atomicStore(&done_count, 0u); }
  workgroupBarrier();
  if (done) { atomicAdd(&done_count, 1u); }
  var T = 1.0;
  var acc: array<f32, ${channels}>;
${zero.join('\n')}
  let start = ranges[2u * tile];
  let end = ranges[2u * tile + 1u];
  for (var b = start; b < end; b += 256u) {
    if (li == 0u) { done_flag = atomicLoad(&done_count); }
    if (workgroupUniformLoad(&done_flag) >= 256u) { break; }
    let j = b + li;
    if (j < end) {
      let id = vals[j];
      s_id[li] = id;
      s_a[li] = vec4f(proj[id].x, proj[id].y, proj[id].ca, proj[id].cb);
      s_b[li] = vec2f(proj[id].cc, proj[id].op);
    }
    workgroupBarrier();
    let n = min(256u, end - b);
    if (!done) {
      for (var k = 0u; k < n; k++) {
        let a = s_a[k];
        let bq = s_b[k];
        let dd = a.xy - pixf;
        let power = -0.5 * (a.z * dd.x * dd.x + bq.x * dd.y * dd.y) - a.w * dd.x * dd.y;
        if (power > 0.0) { continue; }
        let alpha = min(0.99, bq.y * exp(power));
        if (alpha < 1.0 / 255.0) { continue; }
        let next_T = T * (1.0 - alpha);
        if (next_T < 1e-4) { done = true; atomicAdd(&done_count, 1u); break; }
        let wgt = alpha * T;
        let fb = s_id[k] * ${words}u;
${accum.join('\n')}
        T = next_T;
      }
    }
    workgroupBarrier();
  }
  let plane = P.width * P.height;
  let pix = py * P.width + px;
${write}
}
`;
}
