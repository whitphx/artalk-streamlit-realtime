# Realtime Performance Notes

## 2026-06-18: Single-frame rendering throughput

Context:

- App: Streamlit realtime ARTalk + GAGAvatar development app.
- Related code commits:
  - ARTalk renderer profiling/warm-up: `1bc3b4d`
  - Streamlit diagnostics UI: `4a3e057`
- Target: 25 fps output, or about 40 ms/frame for the full post-ARTalk path.
- ARTalk model behavior: fixed chunked input/output. One motion chunk is 100
  frames for 4 seconds of 16 kHz audio; the smoother emits 96 frames for the
  first chunk because of its causal delay.
- Diagnostic rows are wall-clock timings from the app UI. The profiled renderer
  path synchronizes CUDA at stage boundaries to reduce ambiguous GPU timing
  attribution.

### Mesh-only renderer

Measured stage summary:

| Stage | Count | Avg ms | Max ms |
| --- | ---: | ---: | ---: |
| Resample input | 49 | 0.3 | 3.8 |
| ARTalk streamer feed | 49 | 16.5 | 630.2 |
| Savgol smoother | 3 | 3.7 | 7.3 |
| Avatar prepare frame | 247 | 21.5 | 344.9 |
| Avatar forward model | 247 | 42.7 | 1201.6 |
| Avatar GPU to CPU copy | 247 | 5.7 | 155.4 |
| Avatar render frame | 247 | 70.8 | 1214.7 |
| RGB tensor to ndarray | 246 | 40.0 | 338.4 |
| Render chunk total | 2 | 10392.3 | 13366.2 |

Output counters:

| Counter | Value |
| --- | ---: |
| Motion chunks | 3 |
| Motion frames | 300 |
| Smoothed frames | 296 |
| Rendered frames | 296 |
| Video frames served | 296 |
| Video placeholders | 1424 |
| Audio frames served | 3438 |
| Audio underrun frames | 2838 |

Interpretation:

- Mesh mode is not realtime. The measured post-motion path is roughly
  `70.8 + 40.0 = 110.8 ms/frame`, or about 9 fps before queue/callback effects.
- The max frame time is much larger than the average, especially in
  `Avatar forward model`, which indicates GPU synchronization stalls or
  single-frame render overhead rather than only steady compute.
- WebRTC asks for frames continuously, but the renderer cannot fill the queue,
  so placeholders dominate.

### GAGAvatar renderer

Measured stage summary:

| Stage | Count | Avg ms | Max ms |
| --- | ---: | ---: | ---: |
| Resample input | 49 | 0.4 | 3.6 |
| ARTalk streamer feed | 49 | 18.8 | 434.3 |
| Savgol smoother | 3 | 1.7 | 2.6 |
| Renderer warm-up | 1 | 512.8 | 512.8 |
| Warm-up prepare | 1 | 176.8 | 176.8 |
| Warm-up forward | 1 | 307.2 | 307.2 |
| Warm-up GPU copy | 1 | 2.1 | 2.1 |
| Warm-up RGB convert | 1 | 26.6 | 26.6 |
| Avatar prepare frame | 228 | 18.9 | 717.2 |
| Avatar forward model | 228 | 63.1 | 475.3 |
| Avatar GPU to CPU copy | 228 | 0.5 | 2.9 |
| Avatar render frame | 228 | 82.6 | 1193.3 |
| RGB tensor to ndarray | 227 | 22.5 | 340.2 |
| Render chunk total | 2 | 9937.1 | 10823.5 |

Output counters:

| Counter | Value |
| --- | ---: |
| Motion chunks | 3 |
| Motion frames | 300 |
| Smoothed frames | 296 |
| Rendered frames | 280 |
| Video frames served | 279 |
| Video placeholders | 770 |
| Audio frames served | 2095 |
| Audio underrun frames | 1495 |

Interpretation:

- GAGAvatar warm-up successfully moves one-time lazy setup out of the live
  render path. The warm-up cost was about 513 ms.
- GAGAvatar still is not realtime. The measured post-motion path is roughly
  `82.6 + 22.5 = 105.1 ms/frame`, or about 9.5 fps before queue/callback
  effects.
- Large max times remain after warm-up, so the outliers are not only first-use
  setup. They are likely GPU synchronization stalls, allocator/kernel variance,
  or single-frame render overhead.

### Conclusion

ARTalk model inference is not the throughput bottleneck in these runs. It still
imposes the expected 4-second chunk floor, but its measured feed time is much
smaller than the post-motion rendering path.

Both mesh and GAGAvatar modes are bottlenecked by single-frame avatar rendering
plus RGB conversion. The current worker renders an entire motion chunk
synchronously, so it cannot keep draining input audio while rendering. This
causes queue starvation, video placeholders, and audio underruns.

### Next optimization direction

The next meaningful experiment is mini-batch rendering:

- Render frames in small batches, e.g. 4 or 8 frames, instead of one frame at a
  time.
- Batch mesh FLAME vertices and mesh rendering where possible.
- Batch GAGAvatar `build_forward_batch` and `forward_expression` where GPU
  memory permits.
- Convert rendered RGB tensors to NumPy in batches or otherwise reduce
  per-frame CPU conversion overhead.

The goal is to reduce Python/kernel-launch overhead and improve GPU utilization.
The tradeoff is a small additional mini-batch latency, but current chunk render
times are already far above the realtime budget, so throughput is the immediate
constraint.

## 2026-06-23: Output buffering knobs

The pipeline now exposes two runtime tuning knobs:

- `--output-prebuffer-seconds`: delayed audio required before playback starts.
  Higher values add startup latency but reduce underruns if chunk rendering is
  only slightly slower than realtime.
- `--output-segment-seconds`: minimum rendered segment size published from the
  render worker to WebRTC. Lower values refill the output buffer sooner, but too
  small a value can increase chunk-boundary jitter.

First comparison target:

```bash
scripts/run_app.sh --no-remote -- \
  --device cuda \
  --render-res 512 \
  --render-batch-size 8 \
  --output-prebuffer-seconds 2.0 \
  --output-segment-seconds 0.5
```

Compare this against the previous defaults:

- `--output-prebuffer-seconds 1.0`
- `--output-segment-seconds 1.0`

Use the diagnostics counters to evaluate whether underruns and placeholders drop
enough to justify the added latency.

The direct end-to-end generation latency metric is `Audio to video publish`.
It measures from when the ARTalk pipeline accepts an audio frame or sample chunk
to when the rendered video segment for the matching audio slice is published to
the output queues. The output counters also expose last/min/max values in
seconds as `audio-to-video latency`.

## 2026-07-07: PyTorch Profiler capture and stage-sync toggle

Two measurement knobs were added to chase the reported latency spikes in
GPU↔CPU transfer and inter-stage buffering:

- `--profile-trace-dir DIR`: opt-in PyTorch Profiler capture in the pipeline
  worker. Only worker iterations that cross ARTalk's 4-second chunk boundary
  are profiled (profiler start/stop on every 20 ms audio item would be too
  expensive). Each profiled chunk exports one Chrome trace to a per-run
  subdirectory. Pipeline stages carry `artalk.*` labels
  (`streamer_feed`, `smoother_feed`, `render_batch`, `rgb_batch_to_numpy`,
  `publish_segment`, `resample`). `--profile-skip-chunks` (default 1) and
  `--profile-max-chunks` (default 2) control which chunks are captured.
- `--no-renderer-stage-sync`: disables the `torch.cuda.synchronize()` calls at
  renderer stage boundaries. The syncs make the wall-clock stage timings above
  attributable, but they also serialize GPU work in the production path — a
  possible contributor to the spikes being measured. With syncs off, per-stage
  timings only measure kernel launch; use end-to-end metrics
  (`audio-to-video latency`, underruns, placeholders) for comparison instead.

Both settings surface in diagnostics as `renderer_stage_sync` and
`profiler_enabled`, and trace exports count as `profiler_traces_captured`
(export runs on the worker thread; its cost shows as `profiler_trace_export`,
and the operator-summary build as `profiler_summary_build`).

When profiling is enabled the app renders a "Torch profiler" panel above the
diagnostics column with each captured chunk's `key_averages` operator table
(sorted by self CUDA time) and a download button for the Chrome trace.

Suggested A/B procedure:

1. Run with defaults (syncs on, no profiler) and note baseline diagnostics.
2. Re-run with `--no-renderer-stage-sync` and compare `audio-to-video latency`
   and underrun/placeholder counters to quantify the sync observer effect.
3. Re-run with `--profile-trace-dir profiles` (both sync modes) and inspect
   the traces in Perfetto for cudaMemcpy/cudaStreamSynchronize stalls around
   the `artalk.*` spans.

## 2026-07-09: Worker-thread pauses dominate; GC pause probe

Analysis of the first profiler capture (`chunk-002`, GAGAvatar mode, render
batch size 8) reframed the bottleneck:

- The chunk took 7.2 s wall for 4.0 s of media, but GPU kernels were busy for
  only 1.2 s (17%). GPU compute is not the constraint.
- 69% of the wall time (5.0 s) was 34 gaps in which the pipeline worker thread
  executed nothing — median 151 ms each, recurring every ~160–340 ms, starting
  at arbitrary points in the op stream (including inside `streamer_feed`).
- Stage timings that previously looked like GPU↔CPU transfer spikes were these
  pauses landing inside whichever stage was being timed. Actual DtoH copy time
  for the whole chunk was 66 ms, and the streaming smoother pass was 1.5 ms.
- Without the pauses the chunk would have rendered in ~2.2 s, faster than
  realtime.

The uniform ~150 ms pause duration and allocation-correlated cadence point at
CPython gen-2 GC passes: the collector holds the GIL for the entire pass, so
one pass freezes every pipeline thread at once. To confirm before changing GC
behavior, the app installs a `gc.callbacks` probe
(`artalk_streamlit_realtime/gc_probe.py`) that times every collection, and the
diagnostics column shows a "GC pauses" panel with per-generation stats and
recent ≥10 ms pauses.

If the probe confirms GC as the source, the candidate fix is `gc.freeze()`
after model load plus raised collection thresholds, re-measured with the same
profiler capture procedure.

## 2026-07-10: GC verdict — Streamlit's post-script gc.collect(2)

The probe convicted the garbage collector, with an unexpected shape: **510
gen-2 collections vs only 30 gen-1** over one session, ~151 ms each, arriving
every ~186 ms — 77 s of total pause, roughly 80% of session wall time. That
ratio is impossible for threshold-driven GC (gen-2 fires once per ~10 gen-1
passes); it means something calls `gc.collect()` explicitly and continuously.

The caller is Streamlit: `ScriptRunner._on_script_finished` runs `gc.collect(2)`
after **every script and fragment run** (streamlit 1.58,
`runtime/scriptrunner/script_runner.py:884`). The diagnostics fragments rerun
every 500 ms / 1 s / 2 s, so the app generates ~5 full stop-the-world
collections per second, each traversing the whole torch object graph while
holding the GIL. This is also why heavier diagnostics UI made playback worse,
and why the pauses appeared as "GPU transfer spikes" in wall-clock stage
timings.

Fix: `runner.postScriptGC = false` in `.streamlit/config.toml` (the collect is
config-gated). Organic threshold-driven GC stays enabled; if occasional
threshold-triggered gen-2 passes still show up as ~150 ms hiccups in the GC
panel, the follow-up is `gc.freeze()` after model load to shrink the traversed
object graph.

## 2026-07-13: Post-GC-fix profile and stage-sync A/B

Re-baseline after the GC fix (GAGAvatar, 512, batch 8, syncs on): a 4-second
chunk processes in 1.78 s wall (was 7.2 s), render batches are uniform
106–128 ms (was 87–1276 ms), worker-thread gaps >20 ms dropped from 34 to 0,
GPU is ~74% busy while processing, and the realtime ratio is 0.42x with zero
audio underruns. Organic GC now costs ~1% of wall time and the multi-second
output buffer absorbs individual passes without underruns. Throughput is no
longer the constraint; remaining end-to-end latency is dominated by the 4 s
model chunk floor, chunk fill time (speech pauses), and startup backlog that
strictly-realtime playback never drains (Interactive mode drains it during
silent gaps between responses).

Stage-sync A/B results:

- Mesh mode (headless, 256 px): disabling the syncs halves chunk render time,
  1.61 s → 0.80 s (realtime ratio 0.42x → 0.21x).
- GAGAvatar mode (live app + trace): end-to-end chunk time is roughly
  unchanged (~2.0 s incl. profiler overhead; batch medians slightly faster).
  The path is GPU-bound and each batch ends in a GPU-to-CPU copy that forces
  a sync anyway, so removing the 27 intermediate syncs mostly re-attributes
  time rather than saving it. No underruns or jitter regressions.

Default flipped: `--renderer-stage-sync` is now off by default and becomes an
opt-in for reading per-stage timings. With syncs off, queued GPU work shows up
in whichever stage forces the next sync (typically `avatar_gpu_to_cpu_batch`).

Side observation while reproducing GAGAvatar headless: a mis-prepared
rasterizer batch made `diff_gaussian_rasterization` request an absurd
allocation (131,067 GiB "CUDA out of memory"). If a huge nonsensical OOM like
this appears (as in the 2026-07-02 meeting's A100 attempt), suspect corrupted
rasterizer inputs (avatar/camera setup), not actual memory pressure.

## 2026-07-13: Silence pump flush budget for turn latency

With throughput solved, measured turn latency was dominated by queued
silence: the silence pump kept injecting idle silence up to its 3 s output
buffer cap, so playback always ran ~3 s behind and every new response queued
behind that backlog (observed `Audio out buffer` 3.6–3.8 s and `Serve wait`
~1.7 s even with `--output-prebuffer-seconds 0.5 --output-segment-seconds
0.5`).

The pump now stops after injecting one model chunk's worth (4 s) of silence
per idle period — enough to flush any partial chunk containing trailing real
speech, which is the pump's actual job — and resumes only on new real input.
During idle the output buffer drains to zero, so a new response queues behind
nothing and turn latency approaches the floor (chunk fill + prebuffer +
render). Cost: the avatar freezes on its last frame during long idle instead
of continuously idling. New counters: `silence_pump_samples_since_input`,
`silence_pump_flush_complete_skips`.

## 2026-07-14: Barge-in freeze — bounded video queue vs unbounded audio

Reported symptom: interrupt the assistant mid-response, and when the next
response's audio starts after the gap, the face is frozen and only recovers
later.

Root cause: video frames and audio samples are published in paired amounts,
but the video queue is bounded (200 frames = 8 s, overflow drops the *oldest*
frames) while the audio buffer is unbounded. OpenAI delivers response audio in
bursts and nothing cancelled a superseded response, so a barge-in stacks the
old response's unplayed remainder + pump silence + the new response — easily
past 8 s of unplayed media. The queue then evicts the earliest video frames
(including the new response's start) while their audio still plays; the
video-serving clock (`synced_audio_samples_served`) sits behind the surviving
frame indices until that orphaned audio finishes — the freeze. The same
mechanism fires on any single response whose unplayed backlog exceeds 8 s.
Diagnostic signature: `video frames dropped` > 0.

Fixes (verified headless with simulated playback callbacks):

- `ARTalkPipeline.flush_output()` — drops queued-but-unplayed output and
  credits the flushed samples to the served clock so later frame indices stay
  reachable. The bridge calls it on `input_audio_buffer.speech_started`
  (server VAD barge-in), so the assistant now stops talking when interrupted
  instead of playing out its buffered answer. Staleness after a flush is
  bounded by the model chunk already inside the worker.
- Paired eviction — when the video queue overflows, the matching audio is now
  removed from the buffer front and credited to the clock, so overflow
  degrades to a content skip that stays in sync instead of a frozen face.
  Counter: `audio_samples_skipped_for_dropped_video`.

## 2026-07-14: Turn-start latency metric

Session-average latency metrics mislead in Interactive mode: OpenAI delivers
response audio in bursts, so frames deep inside a response are "accepted"
long before they can possibly play, inflating averages regardless of pipeline
performance. The honest per-turn KPI is the time from a turn's first real
audio to the first video frame served for it.

Implementation: `push_silence` now marks its samples as filler
(`push_audio_samples(..., is_filler=True)`), so the pipeline can detect a
turn start at acceptance time — real audio arriving after ≥1 s (`TURN_GAP_S`)
without real audio. The first published frame whose audio midpoint reaches
the turn start carries a tag; when that frame (or, if it was sync-dropped,
the frame served in its place) is served, the pipeline records
`turn_first_frame_latency`. Surfaced in the diagnostics Overview as
"Turn first frame" (last/avg/max) and "Turns served".

Caveat: in Loopback mode the browser microphone streams continuously
(including room silence), so only the first turn registers; the metric is
meaningful in Interactive mode, where only assistant response audio enters
the pipeline.

## 2026-07-14: Full-path warm-up pre-roll

The first response of a fresh session paid one-time costs (CUDA allocations,
cuDNN autotuning, audio encoder first pass) that later turns did not. The
pipeline now runs one silent model chunk through the full streamer → smoother
→ renderer path at construction and discards the output; the streamer and
smoother are reset afterwards so real inference starts from the same state as
an un-warmed pipeline. These costs are process-global, so warm-up runs once
per (device, renderer mode, resolution, batch size) per process — pipelines
recreated by settings changes skip it.

Measured headless (mesh 256): warm-up costs 1.6 s at init and cuts the first
turn's first-frame latency from 1.26 s to 0.49 s, making turn 1 the fastest
turn instead of the slowest. This replaces the previous single-frame
GAGAvatar-only renderer warm-up; warm-up timings appear in diagnostics as
`pipeline_warmup` / `warmup_*` stages.

Operational note: Streamlit's file watcher hot-reloads app-local modules but
not the editable-installed `artalk` package — after ARTalk-package changes,
restart the app server or the running process keeps executing the old
pipeline code under the new UI.

## 2026-07-14: Truncate interrupted responses in OpenAI conversation state

After a barge-in flush, OpenAI's conversation state still contained the full
assistant answer, so follow-up replies could reference audio the user never
heard. The bridge now tracks the current response item and where its audio
begins on the pipeline's served-samples clock (served + everything queued
ahead when its first delta arrives); on `speech_started` it computes the
actually-heard duration from that clock — read before `flush_output()`, which
credits the discarded samples — and sends `conversation.item.truncate` with
`audio_end_ms` clamped to the received duration. Verified headless with a
fake connection: 10 s pushed, interrupted mid-playback, truncated to the
served duration exactly. Truncation is skipped when everything played
(within 100 ms) and failures are logged, never fatal.

## 2026-08-02: Mid-turn pause before a response's final words — chunk flush

Reported: a ~4 s answer ("...How about you?") paused for 1-2 s right before
its last words, then resumed. The diagnostics snapshot showed the mechanism
exactly:

- The response was 68,784 samples (4.3 s). Chunk 1 (64,000) rendered and
  played; the final 4,784 samples sat in the streamer buffer below the
  model's chunk threshold.
- The only path to flush that tail was the silence pump, whose gates are
  deliberately conservative for microphone input: 1.0 s idle threshold
  (`recent_input_skips` 18), skips while the worker renders
  (`worker_busy_skips` 22 ≈ 2.2 s), and realtime-paced 0.25 s silence
  chunks (~3.7 s to fill the remaining 59,216 samples). Chunk 2 therefore
  landed ~1 s after chunk 1 finished playing: `min_audio_out_buffer` 0.0,
  21 underrun frames — the audible pause.

This is the 4-second chunk floor biting *mid-turn* for any response whose
length is not a multiple of 4 s — which the earlier fixes (eviction,
backpressure) had been masking under larger problems.

Fix: `ARTalkPipeline.request_chunk_flush()` enqueues a `ChunkFlushRequest`
through the same FIFO as audio; when the worker processes it, it pads the
partial chunk to the boundary with exactly the missing silence (so it can
never race ahead of still-queued audio, and the padding is sample-exact).
The OpenAI bridge fires it on `response.output_audio.done` — the upstream's
explicit end-of-audio signal — replacing heuristics with ground truth. The
silence pump remains as the fallback for Loopback (no done event) and for
cancelled responses. Counters: `chunk_flush_requests`,
`chunk_flush_padded_samples`.

Verified headless with the exact failing shape (4.3 s burst + flush): the
tail chunk is ready 1.0 s after response end (previously ~6-7 s), padding
is exact (59,216 samples), zero underruns through full playback. Confirmed
live: the pause before final words is gone.

## 2026-08-02: The recurring full-rerun hang — GC-probe self-deadlock

Since 2026-07-09 the app intermittently froze on full script reruns (stop
clicks, snapshot-button clicks), with the process alive and the pipeline
still logging. The hung process ran on a host without interactive access at
incident time, which drove the forensics tooling now in the app: a SIGUSR1
faulthandler dump, and then the `ScriptHangWatchdog`
(`artalk_streamlit_realtime/hang_watchdog.py`), which auto-dumps every
thread's stack to `diagnostics_snapshots/hang-*.txt` when a script run
exceeds 90 s and re-dumps periodically so an incident's progression is
visible.

The watchdog's dumps caught the cause in the act:

- `GcPauseProbe.snapshot()` copied its stats while holding the probe lock;
  the copies allocate.
- One allocation triggered a garbage collection on that same thread, and
  the collection's stop callback (`_on_gc_event`) blocked acquiring the
  same non-reentrant lock — a permanent self-deadlock (the dump shows
  `_on_gc_event` stacked directly on `snapshot`, with faulthandler's
  "Garbage-collecting" marker).
- The poisoned lock then wedged every subsequent script rerun at its first
  line, `gc_pause_probe.install()` — the dumps show wedged script threads
  accumulating (2 → 3) across the incident — while media threads stayed
  healthy. The deadlocked callback also wedges the collector mid-collection,
  suppressing garbage collection process-wide, the likely cause of the
  multi-minute audio anomalies observed during incidents.
- The trigger was probabilistic (one gen-0 dice roll per 1 Hz panel
  refresh), which is why hangs appeared sporadic and correlated loosely
  with clicking things.

Fix: the callback uses a non-blocking acquire and drops the sample on
contention (`samples_dropped` in the snapshot). Reproduced the trigger
against the fix: a collection fired while the lock is held completes
without deadlock.

Lessons for refactoring or re-implementation:

- A `gc.callbacks` hook runs on whatever thread triggered the collection,
  including mid-allocation inside a critical section of the same code that
  the hook needs — it must never block on a lock that user code holds
  around allocations. Non-blocking acquire + lossy sampling is the correct
  shape; so is keeping the callback allocation-free where possible.
- Widgets that must not lose interactions (e.g. the snapshot button) cannot
  live inside `run_every` fragments: auto-reruns race with clicks and
  silently swallow them.
- The watchdog + one-click snapshot pattern is what solved this: hangs on
  unreachable hosts self-document to a shared filesystem. Keep both in any
  future incarnation of this app.

## 2026-07-29: Fast-forward audio at long avatar turns — render backpressure

Diagnostics snapshots from production sessions (captured with the new
"Save diagnostics snapshot" button) identified two distinct residual
problems:

1. **Co-tenant GPU contention**: `avatar_forward_batch` intermittently 3x
   slower (100 ms vs 33 ms baseline) on the shared host, draining the output
   buffer to zero and causing real underruns. Mitigations: larger
   `--output-prebuffer-seconds` on shared hosts, and (planned) GPU-utilization
   sampling in metrics to correlate with neighbors' load.
2. **Audio "fast-forwarding" at the start of long avatar turns** — one
   session skipped exactly 4.0 s (`audio_samples_skipped_for_dropped_video`
   64000). Upstream delivers response audio in bursts and rendering runs
   ~2.5x realtime, so unplayed backlog blows past the 200-frame video-queue
   cap and the paired-eviction safety path (built for the barge-in freeze
   fix) fires as routine behavior, skipping media in ~0.6 s bites.

Fix for 2: the render loop now applies backpressure — when the video queue
reaches a high-water mark (150 frames ≈ 6 s ahead), the worker pauses until
playback drains it, so frames are rendered just in time and eviction remains
a last-resort safety only. Verified headless: a 16 s burst that previously
evicted ~4 s now plays with zero drops, zero skipped audio, zero underruns
(`render_backpressure_waits` / `render_backpressure_wait` metrics record the
pauses). Backpressure also bounds the audio buffer (~6 s), so memory stays
capped on both queues.

Trade-offs of backpressure: turn-start latency is unaffected (it only
engages once playback is already 6 s behind the renderer), memory and GPU
use improve (no rendering of frames that would be evicted), but **barge-in
staleness during long responses gets somewhat worse on average**. Before,
a long response was typically fully rendered ahead, so an interrupt flushed
everything and the avatar went quiet almost immediately; now the worker is
usually mid-chunk when the interrupt arrives, and the remainder of that one
model chunk still renders and plays after the flush — up to ~4 s of stale
speech worst case (same bound as before, but hit more often; the input
queue holding the rest of the response is still drained by the flush). If
this proves annoying in practice, the follow-up is a flush epoch: the
worker checks a generation counter before publishing and discards segments
belonging to a superseded response. A second, minor effect: post-ARTalk /
pre-render-wait diagnostics now include intentional waiting, so the new
backpressure metrics should be consulted before reading those as
regressions.

## 2026-07-18: The 131,000-GiB rasterizer "OOM" solved — GPU-arch mismatch

The absurd `diff_gaussian_rasterization` allocation failures (seen headless
here, and in the 2026-07-02 meeting's A100 attempt) were a GPU-architecture
mismatch made invisible by upstream error handling. The installed
`diff_gaussian_rasterization_32d` binary contained SASS and PTX for
**sm_75 only** — the home directory is NFS-shared across GPU hosts, and the
binary had been built on the Turing host. On other architectures the kernels
cannot launch at all, but the rasterizer's `CHECK_CUDA(A, debug)` macro only
checks errors when `debug=True` (GAGAvatar passes `debug=False`), so the
failed launches left buffer-size computations reading garbage and the
failure surfaced as `Tried to allocate 131,0xx GiB` — or occasionally as
silent black frames. Sessions "worked" or "died" depending on which host the
app happened to run on.

Fix applied: rebuilt the extension in the shared env as a fat binary with
`TORCH_CUDA_ARCH_LIST="6.0;7.5;8.0+PTX"` (P100 + Turing + A100), using a
throwaway conda `cuda-nvcc 12.1` toolchain to match torch's cu121. Renders
are now correct and deterministic on the P100 host; the A100 host should be
re-tested (its previous failure was almost certainly this same binary).

Upstream fix submitted:
<https://github.com/xg-chu/diff-gaussian-rasterization/pull/1>. Deeper
analysis there: besides the debug-gated launch checks, the cub size queries
in `fromChunk` discard their error status, so on failure the buffer size is
an uninitialized stack `size_t` (~2^47) — which is exactly where the
131,0xx-GiB number and its per-process "nondeterminism" came from. The same
defect exists in the root graphdeco-inria/diff-gaussian-rasterization, where
this failure mode is a recurring user report. The shared env's installed
extension is built from the patched source (fat binary, sm_60/75/80+PTX).

## 2026-07-15: Windowed smoother and gc.freeze

Two long-session protections:

- `CausalSavgolSmoother` re-filtered its entire history on every feed
  (O(n²) cumulative) and round-tripped the whole growing buffer through the
  GPU↔CPU boundary. It now filters only a small window around the frames
  being emitted — savgol interior frames need just 4 frames of context per
  side, and edge-fitted frames are computed from slices whose edges coincide
  with the true sequence edges — and drops frames that can no longer
  influence output. Verified: `scripts/check_streaming_parity.py` passes
  with max abs diff 0.0, and a 20,000-frame stress run (13+ min of session,
  irregular feed sizes) is bit-exact with one-shot filtering while retaining
  at most 9 frames and flat ~1–2 ms per feed.
- The app now calls `gc.collect()` + `gc.freeze()` once after the runtimes
  and first pipeline are loaded, moving the loaded model graphs (~317k
  objects) into the permanent generation. Measured in a loaded process:
  a gen-2 pass costs 129–162 ms before freeze — exactly the pause size the
  probe kept recording — and ~0 ms after. Organic gen-2 passes can no longer
  stall the worker for a visible duration; the GC panel's "frozen objects"
  readout confirms the freeze took effect. Trade-off: frozen objects are
  never cycle-collected, which is fine for process-lifetime model graphs.

## 2026-07-14: Underrun counter vs idle silence

Since the silence pump flush budget, the output buffer is intentionally empty
during idle, and the audio callback was counting every idle frame as a
playback underrun (~50/s of noise). Empty-buffer frames now count as
`audio_playback_underrun_frames` only while content is in flight (queued
input, busy worker, queued video, or real audio accepted within the last
model-chunk window); otherwise they count as `audio_idle_silence_frames`.
Verified headless: chunk-fill starvation still registers as underruns; six
seconds of true idle registers zero.
