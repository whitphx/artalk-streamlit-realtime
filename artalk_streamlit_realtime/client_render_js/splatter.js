// Host side of the N-channel WGSL tile rasterizer: buffers, pipelines and
// the per-frame dispatch sequence. Runs unchanged in browsers and in Node
// via Dawn (node-webgpu).
// `imageBuffer` renders into a buffer the caller owns, so a network bound to
// it keeps its graph capture across avatars.

import {
  TILE, SCAN_BLOCK, SORT_BLOCK, RADIX_PASSES, PARAMS_BYTES,
  projectShader, scanLocalShader, scanBlocksShader, scanHistShader, emitShader,
  upsweepShader, downsweepShader, rangesShader, rasterizeShader,
} from './shaders.js';
import { packFeatures } from './bundle.js';

export const STAGES = ['project', 'scan', 'emit', 'sort', 'ranges', 'rasterize'];

export async function createSplatter(device, bundle, {
  size = 512, channels = 16, tileTest = 'box', outputF16 = true,
  pairCapacity = 3_000_000, timestamps = true, imageUsage = 0,
  imageBuffer = null,
} = {}) {
  const n = bundle.meta.count;
  const tiles = size / TILE;
  const tileBits = Math.ceil(Math.log2(tiles * tiles));
  const depthBits = 32 - tileBits;
  const scanBlocks = Math.ceil(n / SCAN_BLOCK);
  const sortBlocksCap = Math.ceil(pairCapacity / SORT_BLOCK);
  const words = Math.ceil(channels / 2);
  const useTimestamps = timestamps && device.features.has('timestamp-query');

  const mk = (bytes, usage, label) => device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage, label });
  const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
  const buf = {
    params: mk(PARAMS_BYTES, GPUBufferUsage.UNIFORM | CD, 'params'),
    pos: mk(n * 12, S | CD, 'pos'),
    cov: mk(n * 24, S | CD, 'cov'),
    opac: mk(n * 4, S | CD, 'opacity'),
    feat: mk(n * words * 4, S | CD, 'features'),
    proj: mk(n * 56, S | CS, 'proj'),
    tileCounts: mk(n * 4, S, 'tile_counts'),
    prefix: mk(n * 4, S, 'prefix'),
    blockSums: mk(scanBlocks * 4, S, 'block_sums'),
    blockOffsets: mk(scanBlocks * 4, S, 'block_offsets'),
    counters: mk(16, S | CD | CS, 'counters'),
    indirect: mk(24, S | GPUBufferUsage.INDIRECT, 'indirect'),
    keysA: mk(pairCapacity * 4, S, 'keysA'),
    valsA: mk(pairCapacity * 4, S, 'valsA'),
    keysB: mk(pairCapacity * 4, S, 'keysB'),
    valsB: mk(pairCapacity * 4, S, 'valsB'),
    hist: mk(16 * sortBlocksCap * 4, S, 'hist'),
    histScan: mk(16 * sortBlocksCap * 4, S, 'hist_scan'),
    ranges: mk(tiles * tiles * 8, S | CD, 'ranges'),
    image: imageBuffer ?? mk(channels * size * size * (outputF16 ? 2 : 4), S | CS | imageUsage, 'image'),
  };
  device.queue.writeBuffer(buf.pos, 0, bundle.arrays.pos.data);
  device.queue.writeBuffer(buf.cov, 0, bundle.arrays.cov.data);
  device.queue.writeBuffer(buf.opac, 0, bundle.arrays.opacity.data);
  device.queue.writeBuffer(buf.feat, 0, packFeatures(bundle, channels));

  const paramsData = new ArrayBuffer(PARAMS_BYTES);
  const pf = new Float32Array(paramsData), pu = new Uint32Array(paramsData);
  const tanFov = bundle.meta.tan_fov;
  const fx = size / (2 * tanFov);
  pf[16] = fx; pf[17] = fx; pf[18] = (size - 1) / 2; pf[19] = (size - 1) / 2;
  pu[20] = size; pu[21] = size; pu[22] = tiles; pu[23] = tiles;
  pu[24] = n; pu[25] = depthBits; pu[26] = pairCapacity; pu[27] = tileTest === 'ellipse' ? 1 : 0;
  pf[28] = tanFov; pf[29] = tanFov; pu[30] = scanBlocks; pu[31] = 0;

  const pipe = async (code, label) => device.createComputePipelineAsync({ label, layout: 'auto', compute: { module: device.createShaderModule({ code, label }), entryPoint: 'main' } });
  const [pProject, pScanLocal, pScanBlocks, pScanHist, pEmit, pRanges, pRaster] = await Promise.all([
    pipe(projectShader, 'project'), pipe(scanLocalShader, 'scan_local'), pipe(scanBlocksShader, 'scan_blocks'),
    pipe(scanHistShader, 'scan_hist'), pipe(emitShader, 'emit'), pipe(rangesShader, 'ranges'),
    pipe(rasterizeShader({ channels, outputF16 }), 'rasterize'),
  ]);
  const pUp = await Promise.all([...Array(RADIX_PASSES)].map((_, p) => pipe(upsweepShader(4 * p), `upsweep_${p}`)));
  const pDown = await Promise.all([...Array(RADIX_PASSES)].map((_, p) => pipe(downsweepShader(4 * p), `downsweep_${p}`)));

  // A null entry skips a binding the shader does not reference (auto layouts drop those).
  const bg = (pipeline, buffers) => device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((b, i) => b && { binding: i, resource: { buffer: b } }).filter(Boolean),
  });
  const bgProject = bg(pProject, [buf.params, buf.pos, buf.cov, buf.opac, buf.proj, buf.tileCounts, buf.counters]);
  const bgScanLocal = bg(pScanLocal, [buf.params, buf.tileCounts, buf.prefix, buf.blockSums]);
  const bgScanBlocks = bg(pScanBlocks, [buf.params, buf.blockSums, buf.blockOffsets, buf.counters, buf.indirect]);
  const bgScanHist = bg(pScanHist, [null, buf.hist, buf.histScan, buf.counters]);
  const bgEmit = bg(pEmit, [buf.params, buf.proj, buf.tileCounts, buf.prefix, buf.blockOffsets, buf.counters, buf.keysA, buf.valsA]);
  const pingpong = [[buf.keysA, buf.valsA, buf.keysB, buf.valsB], [buf.keysB, buf.valsB, buf.keysA, buf.valsA]];
  const bgUp = pUp.map((p, i) => bg(p, [pingpong[i % 2][0], buf.hist, buf.counters]));
  const bgDown = pDown.map((p, i) => bg(p, [...pingpong[i % 2], buf.histScan, buf.counters]));
  // Eight passes leave the sorted keys back in the A buffers.
  const bgRanges = bg(pRanges, [buf.params, buf.keysA, buf.counters, buf.ranges]);
  const bgRaster = bg(pRaster, [buf.params, buf.proj, buf.valsA, buf.ranges, buf.feat, buf.image]);

  let querySet = null, queryBuf = null, queryRead = null;
  if (useTimestamps) {
    querySet = device.createQuerySet({ type: 'timestamp', count: 2 * STAGES.length });
    queryBuf = device.createBuffer({ size: 16 * STAGES.length, usage: GPUBufferUsage.QUERY_RESOLVE | CS });
    queryRead = device.createBuffer({ size: 16 * STAGES.length, usage: GPUBufferUsage.MAP_READ | CD });
  }
  const countersRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | CD });
  const countersInit = new Uint32Array([0, 0xffffffff, 0, 0]);

  // view: 16 floats, row-vector convention (built from the server's 3x4 camera); headPos: headCount x 3 f32.
  function setPose(view, headPos) {
    pf.set(view, 0);
    device.queue.writeBuffer(buf.params, 0, paramsData);
    device.queue.writeBuffer(buf.pos, 0, headPos);
  }

  // Records one frame; `readback` adds the copies needed by readStats/readTimings.
  function encode(readback = true) {
    device.queue.writeBuffer(buf.counters, 0, countersInit);
    const enc = device.createCommandEncoder();
    enc.clearBuffer(buf.ranges);
    let stage = 0;
    const pass = () => {
      const desc = {};
      if (useTimestamps) desc.timestampWrites = { querySet, beginningOfPassWriteIndex: 2 * stage, endOfPassWriteIndex: 2 * stage + 1 };
      stage++;
      return enc.beginComputePass(desc);
    };
    let p = pass();
    p.setPipeline(pProject); p.setBindGroup(0, bgProject); p.dispatchWorkgroups(Math.ceil(n / 256));
    p.end();
    p = pass();
    p.setPipeline(pScanLocal); p.setBindGroup(0, bgScanLocal); p.dispatchWorkgroups(scanBlocks);
    p.setPipeline(pScanBlocks); p.setBindGroup(0, bgScanBlocks); p.dispatchWorkgroups(1);
    p.end();
    p = pass();
    p.setPipeline(pEmit); p.setBindGroup(0, bgEmit); p.dispatchWorkgroups(Math.ceil(n / 256));
    p.end();
    p = pass();
    for (let i = 0; i < RADIX_PASSES; i++) {
      p.setPipeline(pUp[i]); p.setBindGroup(0, bgUp[i]); p.dispatchWorkgroupsIndirect(buf.indirect, 0);
      p.setPipeline(pScanHist); p.setBindGroup(0, bgScanHist); p.dispatchWorkgroups(1);
      p.setPipeline(pDown[i]); p.setBindGroup(0, bgDown[i]); p.dispatchWorkgroupsIndirect(buf.indirect, 0);
    }
    p.end();
    p = pass();
    p.setPipeline(pRanges); p.setBindGroup(0, bgRanges); p.dispatchWorkgroupsIndirect(buf.indirect, 12);
    p.end();
    p = pass();
    p.setPipeline(pRaster); p.setBindGroup(0, bgRaster); p.dispatchWorkgroups(tiles, tiles);
    p.end();
    if (readback) {
      if (useTimestamps) {
        enc.resolveQuerySet(querySet, 0, 2 * STAGES.length, queryBuf, 0);
        enc.copyBufferToBuffer(queryBuf, 0, queryRead, 0, 16 * STAGES.length);
      }
      enc.copyBufferToBuffer(buf.counters, 0, countersRead, 0, 16);
    }
    return enc.finish();
  }

  async function readStats() {
    await countersRead.mapAsync(GPUMapMode.READ);
    const c = new Uint32Array(countersRead.getMappedRange().slice(0));
    countersRead.unmap();
    const f = new Float32Array(c.buffer);
    return { pairs: c[0], depthMin: f[1], depthMax: f[2], overflow: c[3] };
  }

  // Per-stage GPU time in ms from the timestamp queries of the last frame.
  async function readTimings() {
    if (!useTimestamps) return null;
    await queryRead.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(queryRead.getMappedRange().slice(0));
    queryRead.unmap();
    const out = {};
    STAGES.forEach((s, i) => { out[s] = Number(t[2 * i + 1] - t[2 * i]) / 1e6; });
    out.total = Number(t[2 * STAGES.length - 1] - t[0]) / 1e6;
    return out;
  }

  return {
    size, channels, tileTest, outputF16, depthBits, useTimestamps, count: n, imageBuffer: buf.image,
    setPose, encode, readStats, readTimings,
    submitFrame(readback = true) { device.queue.submit([encode(readback)]); },
    destroy() { for (const [k, b] of Object.entries(buf)) if (k !== 'image' || !imageBuffer) b.destroy(); },
  };
}
