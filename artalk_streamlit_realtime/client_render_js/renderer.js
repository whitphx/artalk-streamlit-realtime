// Draws one motion message: the avatar's Gaussians splatted at the streamed
// head vertices into 16 feature channels, then the network on WebGPU through
// ONNX Runtime Web, then a blit to the canvas.
//
// ORT creates its WebGPU device with the first session and keeps it while
// any session lives. Sessions are cached here and never released, so the
// device, the sessions and their graph captures outlive a WebRTC session
// and a restart in the same page is quick.

import { parseBundle } from './bundle.js';
import { createSplatter } from './splatter.js';

const OUT = 512, CHANNELS = 16;
const PAIR_CAPACITY = { 256: 1_500_000, 512: 3_000_000 };
const WASM_PARTS = ['ort-wasm-asyncify-1.30.0.part0.wasm', 'ort-wasm-asyncify-1.30.0.part1.wasm'];
const WASM_SHA256 = '39f9f0894d478800487ed9f7dbe92618498db320cf55c8e3d89adff8dce658da';
export const CAMERA_OFFSET = 48, OFFSETS_OFFSET = 96;

const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
const sha256 = async (bytes) => hex(await crypto.subtle.digest('SHA-256', bytes));
async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.arrayBuffer();
}

// Mirrors hash_fill in the network export: the fixed network-check input.
function hashFill(n, seed, bound) {
  const out = new Float32Array(n), mix = Math.imul(seed + 1, 0x9E3779B9) >>> 0;
  for (let i = 0; i < n; i++) {
    let x = (i ^ mix) >>> 0;
    x ^= x >>> 16; x = Math.imul(x, 0x7FEB352D); x ^= x >>> 15; x = Math.imul(x, 0x846CA68B); x ^= x >>> 16;
    out[i] = ((x >>> 0) / 4294967296 * 2 - 1) * bound;
  }
  return out;
}
const f32b = new Float32Array(1), u32b = new Uint32Array(f32b.buffer);
// Round-to-nearest-even float32 -> float16 bits, matching numpy's astype(float16).
function toHalf(v) {
  f32b[0] = v;
  const x = u32b[0], sign = (x >>> 16) & 0x8000, exp = (x >>> 23) & 0xff;
  let m = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    m |= 0x800000;
    const shift = 14 - e, half = 1 << (shift - 1), rem = m & ((1 << shift) - 1);
    let h = m >>> shift;
    if (rem > half || (rem === half && (h & 1))) h++;
    return sign | h;
  }
  let h = (e << 10) | (m >>> 13);
  const rem = m & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++;
  return sign | h;
}
function fromHalf(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// Same matrix as gagavatar's build_camera_matrices, stored row-major as the splatter's `view` expects:
// rotation in the upper 3x3, translation in the last row, first two columns negated.
function viewFromCamera(cam, out) {
  out.fill(0);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) out[i * 4 + j] = cam[i * 4 + j];
    out[12 + i] = cam[i * 4 + 3];
  }
  out[15] = 1;
  for (let i = 0; i < 4; i++) { out[i * 4] = -out[i * 4]; out[i * 4 + 1] = -out[i * 4 + 1]; }
  return out;
}

const presentCode = (fp16) => /* wgsl */ `
const PLANE = ${OUT * OUT}u;
@group(0) @binding(0) var<storage, read> src: array<${fp16 ? 'u32' : 'f32'}>;
${fp16
    ? 'fn rd(i: u32) -> f32 { let w = unpack2x16float(src[i >> 1u]); return select(w.x, w.y, (i & 1u) == 1u); }'
    : 'fn rd(i: u32) -> f32 { return src[i]; }'}
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let pix = u32(pos.y) * ${OUT}u + u32(pos.x);
  return vec4f(clamp(vec3f(rd(pix), rd(PLANE + pix), rd(2u * PLANE + pix)), vec3f(0.0), vec3f(1.0)), 1.0);
}`;

let ort = null, env = null, device = null, models = null, presentPipeline = null;
const sessions = new Map(), bundles = new Map();

export function environment() { return env; }

// Loads ONNX Runtime Web and picks the precision; safe to call again.
export async function init({ ortUrl, dataBase, log }) {
  if (env) return env;
  const e = { ortUrl };
  ort = await import(ortUrl);
  ort.env.wasm.numThreads = 1;
  try {
    const parts = await Promise.all(WASM_PARTS.map(async (p) => new Uint8Array(await fetchBytes(new URL(`ort/${p}`, dataBase)))));
    const bin = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let o = 0;
    for (const p of parts) { bin.set(p, o); o += p.length; }
    if ((await sha256(bin)) !== WASM_SHA256) throw new Error('wasm checksum mismatch');
    ort.env.wasm.wasmBinary = bin;
    e.wasmSource = 'local';
  } catch (err) {
    e.wasmSource = 'CDN';
    log(`local ORT wasm not used (${err.message})`);
  }
  if (!('gpu' in navigator)) throw new Error('this browser has no WebGPU');
  ort.env.webgpu.powerPreference = 'high-performance';
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const info = adapter.info ?? {};
  e.adapter = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' / ');
  e.softwareAdapter = info.isFallbackAdapter === true || /swiftshader/i.test(e.adapter);
  e.precision = adapter.features.has('shader-f16') ? 'fp16' : 'fp32';
  models = (await (await fetch(new URL('models.json', dataBase))).json()).models;
  env = e;
  return env;
}

async function ensureDevice(log) {
  const d = await ort.env.webgpu.device;
  if (d === device) return;
  if (device) throw new Error('ONNX Runtime replaced its WebGPU device; reload the page');
  device = d;
  device.lost.then((i) => log(`WebGPU device lost: ${i.message || i.reason}`));
  device.addEventListener('uncapturederror', (e) => log(`WebGPU error: ${e.error.message}`));
  const module = device.createShaderModule({ code: presentCode(env.precision === 'fp16'), label: 'present' });
  presentPipeline = device.createRenderPipeline({
    layout: 'auto', vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }] },
  });
}

async function readBuffer(src, bytes) {
  const rb = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, rb, 0, bytes);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const raw = rb.getMappedRange().slice(0);
  rb.destroy();
  return raw;
}

async function sessionFor(key, dataBase, log) {
  if (sessions.has(key)) return sessions.get(key);
  const spec = models.find((m) => m.key === key);
  if (!spec) throw new Error(`no network ${key} in models.json`);
  const prec = env.precision, files = spec.files[prec];
  const weights = new Uint8Array(await fetchBytes(new URL(files.weights_file, dataBase)));
  if ((await sha256(weights)) !== files.weights_sha256) throw new Error(`${files.weights_file}: checksum mismatch`);
  const graph = new Uint8Array(await fetchBytes(new URL(files.file, dataBase)));
  const make = (capture) => ort.InferenceSession.create(graph, {
    executionProviders: [{ name: 'webgpu' }], graphOptimizationLevel: 'all',
    externalData: [{ path: 'weights.bin', data: weights.slice() }], preferredOutputLocation: 'gpu-buffer',
    ...(capture ? { enableGraphCapture: true } : {}),
  });
  const t0 = performance.now();
  let session = await make(true);
  await ensureDevice(log);
  const U = GPUBufferUsage, dt = prec === 'fp16' ? 'float16' : 'float32', bpe = prec === 'fp16' ? 2 : 4;
  const nIn = spec.input.reduce((a, b) => a * b), nOut = spec.output.reduce((a, b) => a * b);
  const s = { key, size: spec.splat_size, capture: true, bpe };
  const bind = () => {
    s.inBuf = device.createBuffer({ size: nIn * bpe, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST, label: `${key}_in` });
    s.outBuf = device.createBuffer({ size: nOut * bpe, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST, label: `${key}_out` });
    const feeds = { [session.inputNames[0]]: ort.Tensor.fromGpuBuffer(s.inBuf, { dataType: dt, dims: spec.input }) };
    const fetches = { [session.outputNames[0]]: ort.Tensor.fromGpuBuffer(s.outBuf, { dataType: dt, dims: spec.output }) };
    s.run = () => session.run(feeds, fetches);
  };
  const probe = hashFill(nIn, spec.check.input_seed, 1.0);
  const writeProbe = () => device.queue.writeBuffer(s.inBuf, 0, prec === 'fp32' ? probe : Uint16Array.from(probe, toHalf));
  bind();
  writeProbe();
  // The first run is also the graph capture, so a capture failure surfaces here.
  try {
    await s.run();
  } catch (e) {
    log(`${key}: graph capture failed (${e.message}); running without it`);
    session = await make(false);
    s.capture = false;
    bind();
    writeProbe();
    await s.run();
  }
  const raw = await readBuffer(s.outBuf, nOut * bpe);
  const out = prec === 'fp32' ? new Float32Array(raw) : Float32Array.from(new Uint16Array(raw), fromHalf);
  const tol = prec === 'fp16' ? 0.05 : 2e-3;
  const maxDiff = Math.max(...spec.check.indices.map((idx, k) => Math.abs(out[idx] - spec.check.values[k])));
  s.check = { ok: out.every(Number.isFinite) && maxDiff < tol, maxSampleDiff: maxDiff, tolerance: tol };
  s.setupMs = Math.round(performance.now() - t0);
  log(`${key}: ready in ${s.setupMs} ms, ${prec}${s.capture ? ' + graph capture' : ''}, network check ${s.check.ok ? 'ok' : 'FAILED'}`);
  sessions.set(key, s);
  return s;
}

function bundleFor(avatar, dataBase) {
  if (!bundles.has(avatar)) {
    bundles.set(avatar, fetchBytes(new URL(`avatars/${encodeURIComponent(avatar)}/avatar.gsb`, dataBase)).then(parseBundle));
  }
  return bundles.get(avatar);
}

// One network and one avatar drawing into one canvas.
export async function createRenderer({ canvas, network, avatar, dataBase, log }) {
  const sess = await sessionFor(network, dataBase, log);
  const bundle = await bundleFor(avatar, dataBase);
  const fp16 = env.precision === 'fp16';
  const sp = await createSplatter(device, bundle, {
    size: sess.size, channels: CHANNELS, tileTest: 'ellipse', outputF16: fp16,
    pairCapacity: PAIR_CAPACITY[sess.size], timestamps: false, imageBuffer: sess.inBuf,
  });
  const ctx = canvas.getContext('webgpu');
  ctx.configure({ device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'opaque' });
  const bg = device.createBindGroup({ layout: presentPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: sess.outBuf } }] });
  const headRef = bundle.arrays.head_ref.data, n = headRef.length;
  const pos = new Float32Array(n), view = new Float32Array(16);
  return {
    network, avatar, precision: env.precision, graphCapture: sess.capture, networkCheck: sess.check,
    // Resolves when the frame is queued for display.
    async draw(message, quantStep) {
      const cam = new Float32Array(message, CAMERA_OFFSET, 12);
      const q = new Int16Array(message, OFFSETS_OFFSET, n);
      for (let i = 0; i < n; i++) pos[i] = headRef[i] + q[i] * quantStep;
      sp.setPose(viewFromCamera(cam, view), pos);
      device.queue.submit([sp.encode(false)]);
      await sess.run();
      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
      pass.setPipeline(presentPipeline); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
      device.queue.submit([enc.finish()]);
    },
    // Resolves when the GPU has finished everything submitted so far.
    done: () => device.queue.onSubmittedWorkDone(),
    destroy() { sp.destroy(); },
  };
}
