// One WebRTC session of the client-side renderer: receives motion messages
// (layout in client_render.py), renders each as it arrives, shows it when
// the audio it belongs to plays, and measures what happened.

import { init, createRenderer, environment } from './renderer.js';
import { createPacer, slewToward, RING_SLOTS } from './pacer.js';
import { makeBadge } from './badge.js';

const PAGE = 'artalk-client-render/1';
const HEADER_BYTES = 56;
const OFFSET_WINDOW_MS = 2000;
const SAMPLE_CAP = 30000;
const PER_FRAME_CAP = 750;

function parseHeader(buf) {
  const v = new DataView(buf);
  if (v.getUint32(0, true) !== 0x31565241) return null; // "ARV1"
  return {
    index: v.getUint32(4, true), generation: v.getUint32(8, true), jaw: v.getFloat32(12, true),
    mediaMs: v.getFloat64(16, true) * 1000, sentWallMs: v.getFloat64(24, true), leadMs: v.getFloat32(32, true),
    rms: v.getFloat32(36, true), audioFrameMs: v.getFloat64(40, true) * 1000,
    rtp: v.getUint32(52, true) ? v.getUint32(48, true) : null,
  };
}

// On the app's URL, ?raw=1 adds the time series to the results for offline
// analysis.
const PAGE_PARAMS = (() => { try { return new URLSearchParams(window.parent.location.search); } catch { return new URLSearchParams(); } })();
const RAW = PAGE_PARAMS.has('raw');

function schemeWindows(rows, windowMs = 8000) {
  const out = [];
  const valid = rows.filter((r) => r[3] !== null);
  for (let t0 = valid[0]?.[0]; valid.length && t0 + windowMs <= valid.at(-1)[0]; t0 += windowMs) {
    const d = summary(valid.filter((r) => r[0] >= t0 && r[0] < t0 + windowMs).map((r) => r[3] - r[2]));
    if (d) out.push({ atS: Math.round(t0 / 100) / 10, median: d.median, p10: d.p10, p90: d.p90 });
  }
  return out;
}

function summary(xs) {
  const s = xs.filter(Number.isFinite).sort((a, b) => a - b);
  if (!s.length) return null;
  const q = (p) => { const i = (s.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i); return s[lo] + (s[hi] - s[lo]) * (i - lo); };
  const r = (x) => Math.round(x * 10) / 10;
  return { median: r(q(0.5)), p10: r(q(0.1)), p90: r(q(0.9)), max: r(s[s.length - 1]), n: s.length };
}

function push(list, item) {
  list.push(item);
  if (list.length > SAMPLE_CAP) list.splice(0, list.length - SAMPLE_CAP);
}

// f(t) for a time series [[t, v], ...] sorted by t: linear, or the latest value at or before t ("hold").
function sampler(points, hold) {
  return (t) => {
    let lo = 0, hi = points.length - 1;
    if (hi < 0 || t < points[0][0] || t > points[hi][0]) return NaN;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (points[mid][0] <= t) lo = mid; else hi = mid; }
    const [t0, v0] = points[lo], [t1, v1] = points[hi];
    if (hold || t1 === t0) return t >= t1 ? v1 : v0;
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0);
  };
}

// The lag d (ms) that best lines up mouth(t) with sound(t - d): how far the face trails the audio.
function bestLag(mouth, sound) {
  if (mouth.length < 50 || sound.length < 50) return null;
  const m = sampler(mouth, true), s = sampler(sound, false);
  const t0 = Math.max(mouth[0][0], sound[0][0]) + 400, t1 = Math.min(mouth.at(-1)[0], sound.at(-1)[0]) - 400;
  if (t1 - t0 < 5000) return null;
  let best = null;
  for (let d = -400; d <= 400; d += 5) {
    let n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (let t = t0; t < t1; t += 10) {
      const a = m(t), b = s(t - d);
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      n++; sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
    }
    if (n < 100) continue;
    const cov = sab / n - (sa / n) * (sb / n), va = saa / n - (sa / n) ** 2, vb = sbb / n - (sb / n) ** 2;
    if (va <= 0 || vb <= 0) continue;
    const r = cov / Math.sqrt(va * vb);
    if (!best || r > best.r) best = { lagMs: d, r: Math.round(r * 1000) / 1000 };
  }
  return best;
}

// Per window of shown frames: how far the face trails the audio the page plays. The audio's own
// timing comes from matching the RMS the server stamps on each frame (media clock) against the RMS
// heard on the page, so unlike bestLag it does not depend on how well the jaw follows the voice.
function faceVsAudio(drawn, audioServer, audioShown, windowMs = 8000) {
  if (audioServer.length < 50 || audioShown.length < 50) return [];
  const server = sampler([...audioServer].sort((a, b) => a[0] - b[0]), false);
  const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[s.length >> 1]; };
  const out = [];
  for (let t0 = drawn[0]?.[4]; drawn.length && t0 + windowMs <= drawn.at(-1)[4]; t0 += windowMs) {
    const shownMinusMedia = median(drawn.filter((d) => d[4] >= t0 && d[4] < t0 + windowMs).map((d) => d[4] - d[8]));
    const heard = audioShown.filter((a) => a[0] >= t0 && a[0] < t0 + windowMs);
    let best = null;
    for (let lag = shownMinusMedia - 400; lag <= shownMinusMedia + 400; lag += 2) {
      let n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
      for (const [t, a] of heard) {
        const b = server(t - lag);
        if (!Number.isFinite(b)) continue;
        n++; sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
      }
      if (n < 200) continue;
      const va = saa / n - (sa / n) ** 2, vb = sbb / n - (sb / n) ** 2;
      if (va <= 0 || vb <= 0) continue;
      const r = (sab / n - (sa / n) * (sb / n)) / Math.sqrt(va * vb);
      if (!best || r > best.r) best = { lag, r };
    }
    if (best) out.push({ atS: Math.round(t0 / 100) / 10, faceMinusAudioMs: Math.round(shownMinusMedia - best.lag), r: Math.round(best.r * 1000) / 1000 });
  }
  return out;
}

// RMS of the audio the page plays, per 10 ms, stamped with the audio context's frame position.
const ENVELOPE_WORKLET = `
class Envelope extends AudioWorkletProcessor {
  constructor() { super(); this.acc = 0; this.n = 0; this.start = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      if (this.n === 0) this.start = currentFrame;
      for (let i = 0; i < ch.length; i++) this.acc += ch[i] * ch[i];
      this.n += ch.length;
      if (this.n >= sampleRate / 100) { this.port.postMessage([this.start, this.n, Math.sqrt(this.acc / this.n)]); this.acc = 0; this.n = 0; }
    }
    return true;
  }
}
registerProcessor('envelope', Envelope);`;

async function startAudioProbe(track, onEnvelope) {
  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([ENVELOPE_WORKLET], { type: 'text/javascript' })));
  const src = ctx.createMediaStreamSource(new MediaStream([track]));
  const node = new AudioWorkletNode(ctx, 'envelope');
  const mute = ctx.createGain();
  mute.gain.value = 0;
  src.connect(node).connect(mute).connect(ctx.destination);
  if (ctx.state !== 'running') await ctx.resume().catch(() => {});
  node.port.onmessage = ({ data: [start, n, rms] }) => {
    // When the block's middle reaches the speakers, on the performance.now() clock.
    const ts = ctx.getOutputTimestamp();
    const mid = (start + n / 2) / ctx.sampleRate;
    onEnvelope(ts.performanceTime + (mid - ts.contextTime) * 1000, rms);
  };
  return ctx;
}

export function start({ channels, root, pc, data, createPlayoutClock }) {
  const channel = channels.motion;
  channel.binaryType = 'arraybuffer';
  const dataBase = new URL(`../${data.dirs.data}/`, location.href);
  const log = (msg) => { console.log(`[client-render] ${msg}`); logLines.push(`${(performance.now() / 1000).toFixed(1)} ${msg}`); };
  const logLines = [];

  root.replaceChildren();
  const wrap = document.createElement('div');
  wrap.style.cssText = 'font: 12px/1.4 ui-monospace, monospace; color: inherit; padding-top: 6px';
  // Rendered frames wait as ImageBitmaps, one per ring slot, and showing one
  // hands it to this canvas. Stacking a canvas per slot and swapping opacity
  // would leave it to the browser when a canvas drawn while hidden gets its
  // new frame on screen. Copying into one WebGPU canvas would queue the copy
  // behind the render the GPU is busy with.
  const stage = document.createElement('canvas');
  stage.width = stage.height = 512;
  stage.style.cssText = 'display: block; width: 100%; max-width: 512px; aspect-ratio: 1; background: #000';
  const stageCtx = stage.getContext('bitmaprenderer');
  const frames = Array(RING_SLOTS).fill(null);
  const dropFrames = () => frames.forEach((f, i) => { f?.close(); frames[i] = null; });
  const status = document.createElement('div');
  const readout = document.createElement('div');
  const controls = document.createElement('div');
  controls.style.cssText = 'display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 4px';
  const delayLabel = document.createElement('label');
  delayLabel.textContent = 'Extra face delay (ms) ';
  const delayInput = document.createElement('input');
  delayInput.type = 'number'; delayInput.step = '10'; delayInput.value = '0'; delayInput.style.width = '5em';
  delayLabel.append(delayInput);
  const copyButton = document.createElement('button');
  copyButton.type = 'button';
  copyButton.textContent = 'Copy results as JSON';
  const json = document.createElement('textarea');
  json.readOnly = true;
  json.style.cssText = 'width: 100%; height: 6em; display: none';
  controls.append(delayLabel, copyButton);
  wrap.append(stage, status, readout, controls, json);
  root.append(wrap);

  let config = data, scheme = data.sync, renderer = null, pending = null, rendering = null, raf = 0, stopped = false;
  const offsets = [];
  let offsetMs = null, jitterBufferMs = 0, playoutMs = 0, renderMsAvg = 0, prevStats = null;
  let holdMsNow = null, holdAt = 0, holdSettled = false;
  const counts = {
    received: 0, bytes: 0, malformed: 0, outOfOrder: 0, drawn: 0, animationFrames: 0, skippedLate: 0, skippedOtherAvatar: 0, skippedHidden: 0, queueOverflow: 0,
    rendered: 0, renderedAhead: 0, presented: 0, skippedOverdue: 0, droppedRingFull: 0, requeuedRingFull: 0,
  };
  const pacer = createPacer(counts);
  const series = { arrival: [], audioServer: [], mouthMedia: [], mouthShown: [], audioShown: [], drawn: [], perFrame: [], schemes: [], rtpClock: [], hold: [], margins: [] };
  // ?hold=0 turns off holding the audio back for late frames; ?holdmax=<ms>
  // changes the most it may add.
  const holdForData = PAGE_PARAMS.get('hold') === '0' ? undefined : {
    ...(PAGE_PARAMS.has('holdmax') && { maxMs: Number(PAGE_PARAMS.get('holdmax')) }),
  };
  const playoutClock = createPlayoutClock?.('audio', { holdForData }) ?? null;
  // Due minus media time under the RTP scheme, slewed like the hold.
  let rtpOffsetNow = null, rtpOffsetAt = 0, lastRtpFrame = null, outputLatencyMs = null;
  const startedPerf = performance.now(), startedWall = Date.now();
  let lastIndex = -1, audioProbe = null, audioProbeError = null;

  async function load(cfg) {
    const t0 = performance.now();
    status.textContent = `Loading ${cfg.network} for ${cfg.avatar}...`;
    try {
      let next;
      if (cfg.network === 'none') {
        const canvas = new OffscreenCanvas(512, 512), ctx = canvas.getContext('2d');
        const badge = cfg.aiBadge ? makeBadge(512) : null;
        next = { network: 'none', avatar: cfg.avatar };
        next.render = async (_message, _step, h) => {
          ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 512, 512);
          ctx.fillStyle = '#6cf'; ctx.fillRect(156, 256, 200, Math.max(2, h.jaw * 1200));
          if (badge) ctx.drawImage(badge.image, badge.x, badge.y);
          return canvas.transferToImageBitmap();
        };
        next.done = async () => {};
      } else {
        await init({ ortUrl: cfg.ortUrl ?? new URL('ort/ort.webgpu.bundle.min.mjs', dataBase).href, graphCapture: cfg.graphCapture, dataBase, log });
        next = await createRenderer({ network: cfg.network, avatar: cfg.avatar, aiBadge: cfg.aiBadge, sharpen: cfg.sharpen, dataBase, log });
      }
      next.generation = cfg.generation;
      next.loadMs = Math.round(performance.now() - t0);
      pending = next;
      pump();
      const env = environment();
      status.textContent = `${cfg.avatar}, ${cfg.network}${env ? `, ${env.precision}, ${env.adapter}` : ''}${env?.softwareAdapter ? ' (software WebGPU)' : ''}`;
    } catch (e) {
      status.textContent = `Renderer failed: ${e.message}`;
      log(`load failed: ${e.stack ?? e}`);
    }
  }

  // Swapped in between renders, so a frame never straddles two renderers.
  function adoptPending() {
    if (!pending) return;
    if (renderer && renderer !== pending && renderer.destroy) renderer.destroy();
    renderer = pending;
    pending = null;
    pacer.reset();
    dropFrames();
  }

  channel.addEventListener('message', ({ data: buf }) => {
    const arrival = performance.now();
    const h = buf instanceof ArrayBuffer && buf.byteLength > HEADER_BYTES ? parseHeader(buf) : null;
    if (!h) { counts.malformed++; return; }
    counts.received++;
    counts.bytes += buf.byteLength;
    if (h.index <= lastIndex) counts.outOfOrder++;
    lastIndex = Math.max(lastIndex, h.index);
    h.arrival = arrival;
    h.buf = buf;
    // When this frame's audio reaches the browser, on this page's clock, if audio and data take the same time to
    // arrive. The lower envelope over a window drops the data channel's own jitter.
    const audioArrival = arrival + h.leadMs - h.mediaMs;
    offsets.push([arrival, audioArrival]);
    while (offsets.length && offsets[0][0] < arrival - OFFSET_WINDOW_MS) offsets.shift();
    const low = Math.min(...offsets.map((o) => o[1]));
    // Slew-limited so the drawing clock never jumps by more than 1 ms per frame.
    offsetMs = offsetMs === null ? low : offsetMs + Math.max(-1, Math.min(1, low - offsetMs));
    push(series.arrival, [arrival, h.index, h.sentWallMs, Date.now()]);
    push(series.audioServer, [h.audioFrameMs + 10, h.rms]);
    push(series.mouthMedia, [h.mediaMs, h.jaw]);
    if (h.rtp !== null) {
      lastRtpFrame = h;
      // Only frames the RTP scheme schedules should hold the audio back.
      if (scheme === 'rtp') playoutClock?.noteDataArrival(h.rtp, arrival);
      if (RAW) push(series.margins, [arrival, h.index, h.mediaMs, playoutClock?.playoutTime(h.rtp) ?? null, rtpDue(h)]);
    }
    // Chrome pauses animation frames in a hidden tab, so a queued frame could only come back late.
    if (document.hidden) { counts.skippedHidden++; return; }
    pacer.add(h);
    pump();
  });

  const holdTarget = () => jitterBufferMs + playoutMs + (Number(delayInput.value) || 0);
  const holdMs = () => holdMsNow ?? holdTarget();
  function updateHold(now) {
    // The first estimates land directly: the face is not moving yet.
    holdMsNow = holdSettled ? slewToward(holdMsNow, holdTarget(), now - holdAt) : holdTarget();
    holdSettled = jitterBufferMs > 0;
    holdAt = now;
  }
  // When the latest stamped frame's audio is heard, by the receiver's RTP
  // clock, minus that frame's media time. Where the browser reports no output
  // delay, the audio probe's output latency stands in for it.
  function rtpOffsetTarget() {
    if (!playoutClock || !lastRtpFrame) return null;
    const heard = playoutClock.playoutTime(lastRtpFrame.rtp);
    if (heard === null) return null;
    const tail = playoutClock.outputDelay() === null ? outputLatencyMs ?? 0 : 0;
    return heard + tail - lastRtpFrame.mediaMs + (Number(delayInput.value) || 0);
  }
  function updateRtpOffset(now) {
    const target = rtpOffsetTarget();
    if (target === null) return;
    rtpOffsetNow = rtpOffsetNow === null || !holdSettled ? target : slewToward(rtpOffsetNow, target, now - rtpOffsetAt);
    rtpOffsetAt = now;
  }
  const arrivalDue = (h) => h.mediaMs + offsetMs + holdMs();
  const rtpDue = (h) => (rtpOffsetNow === null ? null : h.mediaMs + rtpOffsetNow);
  // Falls back to the arrival estimate until the RTP clock has a reading.
  const dueAt = (h) => (scheme === 'rtp' && rtpOffsetNow !== null ? rtpDue(h) : arrivalDue(h));

  // At most one render in flight: queuing more on the GPU only delays the
  // frames due soonest.
  async function pump() {
    if (rendering || stopped) return;
    adoptPending();
    if (!renderer || offsetMs === null || document.hidden) return;
    const job = pacer.takeNext(performance.now(), dueAt, renderer.generation);
    if (!job) return;
    rendering = job;
    job.start = performance.now();
    try {
      const frame = await renderer.render(job.h.buf, config.quantStep, job.h);
      frames[job.slot]?.close();
      frames[job.slot] = frame;
      await renderer.done();
      job.end = performance.now();
      renderMsAvg = renderMsAvg ? renderMsAvg * 0.9 + (job.end - job.start) * 0.1 : job.end - job.start;
      if (document.hidden) { counts.skippedHidden++; pacer.cancel(job); } else pacer.finish(job, job.end, dueAt);
    } catch (e) {
      log(`render failed: ${e.message}`);
      pacer.cancel(job);
    } finally {
      rendering = null;
    }
    pump();
  }

  function show(job, shown) {
    stageCtx.transferFromImageBitmap(frames[job.slot]);
    frames[job.slot] = null;
    counts.drawn++;
    const h = job.h, due = dueAt(h);
    push(series.drawn, [h.index, h.arrival, due, job.start, shown, h.sentWallMs, Date.now(), job.end, h.mediaMs]);
    push(series.mouthShown, [shown, h.jaw]);
    push(series.schemes, [shown, h.mediaMs, arrivalDue(h), rtpDue(h), scheme === 'rtp' && rtpOffsetNow !== null ? 1 : 0]);
    series.perFrame.push([h.index, h.mediaMs, h.arrival, job.start, job.end, due, shown].map((x) => Math.round(x * 10) / 10));
    if (series.perFrame.length > PER_FRAME_CAP) series.perFrame.shift();
  }

  // Shown is the animation frame's timestamp, not the time of the swap:
  // every style change made in this callback reaches the screen with that frame.
  function tick(ts) {
    raf = requestAnimationFrame(tick);
    counts.animationFrames++;
    updateHold(ts);
    updateRtpOffset(ts);
    if (renderer && offsetMs !== null && !document.hidden) {
      const job = pacer.present(ts, dueAt);
      if (job) show(job, ts);
    }
    pump();
  }

  const statsTimer = setInterval(async () => {
    const stats = await pc.getStats();
    let audio = null, playout = null;
    stats.forEach((r) => {
      if (r.type === 'inbound-rtp' && r.kind === 'audio') audio = r;
      if (r.type === 'media-playout') playout = r;
    });
    if (audio && prevStats?.audio && audio.jitterBufferEmittedCount > prevStats.audio.jitterBufferEmittedCount) {
      const jb = 1000 * (audio.jitterBufferDelay - prevStats.audio.jitterBufferDelay) / (audio.jitterBufferEmittedCount - prevStats.audio.jitterBufferEmittedCount);
      jitterBufferMs = jitterBufferMs ? jitterBufferMs * 0.7 + jb * 0.3 : jb;
    }
    if (playout && prevStats?.playout && playout.totalSamplesCount > prevStats.playout.totalSamplesCount) {
      const pd = 1000 * (playout.totalPlayoutDelay - prevStats.playout.totalPlayoutDelay) / (playout.totalSamplesCount - prevStats.playout.totalSamplesCount);
      playoutMs = playoutMs ? playoutMs * 0.7 + pd * 0.3 : pd;
    }
    prevStats = { audio, playout };
    const ctx = await Promise.resolve(audioProbe).catch(() => null);
    if (ctx?.outputLatency !== undefined) outputLatencyMs = ctx.outputLatency * 1000;
    const recv = pc.getReceivers().find((r) => r.track.kind === 'audio');
    const src = recv?.getSynchronizationSources?.()[0];
    const hold = playoutClock?.hold();
    if (hold) push(series.hold, [performance.now(), hold.targetMs, hold.jitterBufferMs, hold.marginMs, jitterBufferMs]);
    if (src) push(series.rtpClock, [performance.now(), src.timestamp - performance.timeOrigin, Date.now() - src.timestamp, src.rtpTimestamp ?? null]);
    const s = (performance.now() - startedPerf) / 1000;
    readout.textContent = `received ${counts.received} (${(counts.bytes * 8 / 1000 / Math.max(s, 1)).toFixed(0)} kbit/s), shown ${counts.presented}, ` +
      `rendered ahead ${counts.renderedAhead}, late ${counts.skippedLate + counts.skippedOverdue}, hold ${holdMs().toFixed(0)} ms ` +
      `(jitter buffer ${jitterBufferMs.toFixed(0)} + playout ${playoutMs.toFixed(0)}), render ${renderMsAvg.toFixed(0)} ms, ` +
      `${holdText()}, RTP minus arrival scheme ${rtpOffsetNow === null || offsetMs === null ? 'n/a' : (rtpOffsetNow - offsetMs - holdMs()).toFixed(0)} ms`;
  }, 500);

  function holdText() {
    const hold = playoutClock?.hold();
    if (!hold?.supported) return 'audio hold off';
    const ms = (x) => (x === null ? 'none' : `${x.toFixed(0)} ms`);
    return `audio hold target ${ms(hold.targetMs)}, frame margin p2 ${ms(hold.marginMs)}`;
  }

  function onTrack(track) {
    if (audioProbe || track.kind !== 'audio') return;
    audioProbe = startAudioProbe(track, (t, rms) => push(series.audioShown, [t, rms]))
      .catch((e) => { audioProbeError = e.message; log(`audio probe unavailable: ${e.message}`); });
  }
  pc.addEventListener('track', (e) => onTrack(e.track));

  function results() {
    const durS = (performance.now() - startedPerf) / 1000;
    const drawn = series.drawn;
    const shownGaps = drawn.slice(1).map((d, i) => d[4] - drawn[i][4]);
    const intended = bestLag(series.mouthMedia, series.audioServer);
    const shown = bestLag(series.mouthShown, series.audioShown);
    const arrivals = series.arrival;
    return {
      page: PAGE, userAgent: navigator.userAgent, renderer: environment(), config: { avatar: config.avatar, network: config.network, generation: config.generation },
      networkCheck: renderer?.networkCheck ?? null, graphCapture: renderer?.graphCapture ?? null, loadMs: renderer?.loadMs ?? null,
      startedAt: new Date(startedWall).toISOString(), durationS: Math.round(durS * 10) / 10, counts,
      rates: {
        receivedPerS: Math.round(counts.received / durS * 10) / 10, drawnPerS: Math.round(counts.drawn / durS * 10) / 10,
        kbitPerS: Math.round(counts.bytes * 8 / 1000 / durS), bytesPerMessage: counts.received ? Math.round(counts.bytes / counts.received) : null,
      },
      timingMs: {
        holdNow: Math.round(holdMs()), holdTarget: Math.round(holdTarget()), jitterBuffer: Math.round(jitterBufferMs), playout: Math.round(playoutMs), extraDelay: Number(delayInput.value) || 0,
        interArrival: summary(arrivals.slice(1).map((a, i) => a[0] - arrivals[i][0])),
        interSend: summary(arrivals.slice(1).map((a, i) => a[2] - arrivals[i][2])),
        arrivalToDrawStart: summary(drawn.map((d) => d[3] - d[1])),
        drawStartToShown: summary(drawn.map((d) => d[4] - d[3])),
        renderMs: summary(drawn.map((d) => d[7] - d[3])),
        renderDoneBeforeDue: summary(drawn.map((d) => d[2] - d[7])),
        shownMinusDue: summary(drawn.map((d) => d[4] - d[2])),
        shownInterval: summary(shownGaps),
        // Server send to shown, on the two machines' wall clocks: exact only when they are the same host or synced.
        sentToShownWall: summary(drawn.map((d) => d[6] - d[5])),
        sentToArrivalWall: summary(arrivals.map((a) => a[3] - a[2])),
      },
      shownIntervalsOver60ms: shownGaps.filter((g) => g > 60).length,
      // Positive: the face trails the sound. "shown" is measured on the page (drawn frames against the audio
      // it plays); "intended" is the same estimate on the server's own timeline.
      sync: {
        intended, shown, offsetMs: intended && shown ? shown.lagMs - intended.lagMs : null, audioProbeError, faceVsAudio: faceVsAudio(drawn, series.audioServer, series.audioShown),
        hold: playoutClock?.hold() ?? null,
        // Every 500 ms: [page time, target, the clock's jitter buffer delay, frame margin p2, the session's smoothed jitter buffer delay].
        holdSeries: series.hold.filter((_, i) => i % 4 === 0).map((r) => r.map((x) => (x === null ? null : Math.round(x)))),
        scheme, playoutClock: !!playoutClock, clockOutputDelayMs: playoutClock?.outputDelay() ?? null, rtpStamped: !!lastRtpFrame, outputLatencyMs: outputLatencyMs === null ? null : Math.round(outputLatencyMs),
        // Per window: how much later the RTP scheme would show a frame than the arrival scheme.
        rtpMinusArrival: schemeWindows(series.schemes),
        // getSynchronizationSources: playout time minus now (should be a little negative) and Date.now() minus the timestamp.
        ssrcTimestampMinusNow: summary(series.rtpClock.map((r) => r[1] - r[0])), dateNowMinusSsrcTimestamp: summary(series.rtpClock.map((r) => r[2])),
      },
      raw: RAW ? { margins: series.margins, hold: series.hold, drawn: series.drawn, audioServer: series.audioServer, audioShown: series.audioShown, schemes: series.schemes, arrival: series.arrival } : undefined,
      perFrame: { fields: ['index', 'mediaMs', 'arrival', 'renderStart', 'renderEnd', 'due', 'shown'], rows: series.perFrame },
      log: logLines.slice(-50),
    };
  }
  copyButton.addEventListener('click', async () => {
    // One line per number array, so the per-frame rows stay short enough to paste.
    const text = JSON.stringify(results(), null, 1).replace(/\[\s+([-\d.,\s]+?)\s+\]/g, (_, xs) => `[${xs.replace(/\s+/g, '')}]`);
    json.style.display = 'block';
    json.value = text;
    try { await navigator.clipboard.writeText(text); copyButton.textContent = 'Copied'; } catch { json.select(); copyButton.textContent = 'Select and copy below'; }
    setTimeout(() => { copyButton.textContent = 'Copy results as JSON'; }, 2000);
  });
  window.artalkClientRender = { results, setScheme: (v) => { scheme = v; } };

  const onVisibility = () => {
    if (!document.hidden) return;
    counts.skippedHidden += pacer.clear();
  };
  document.addEventListener('visibilitychange', onVisibility);

  load(config);
  raf = requestAnimationFrame(tick);

  return {
    update(next) {
      const changed = next.avatar !== config.avatar || next.network !== config.network || next.generation !== config.generation;
      config = next;
      scheme = next.sync;
      if (changed && !stopped) load(next);
    },
    cleanup() {
      stopped = true;
      cancelAnimationFrame(raf);
      clearInterval(statsTimer);
      document.removeEventListener('visibilitychange', onVisibility);
      Promise.resolve(audioProbe).then((ctx) => ctx?.close());
      if (renderer?.destroy) renderer.destroy();
      dropFrames();
    },
  };
}
