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
