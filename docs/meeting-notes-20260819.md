# Status for the next meeting (2026-08-19)

Numbers come from `benchmarks/*.json` (best run per GPU and config, measured in isolation with no live callbacks). Realtime at 25 fps is 40 ms per frame.

## 1. Running the app on a laptop

The request splits into two things worth separating: running the whole stack locally on a laptop, and having the experience usable from a laptop. They have different answers.

### What blocks it

| component | portability |
| --- | --- |
| ARTalk motion model | portable: a small transformer, fine on CUDA, MPS or CPU |
| GAGAvatar photoreal renderer | **CUDA only**: `diff_gaussian_rasterization_32d` is a custom CUDA extension with no Metal or CPU path |
| FLAME mesh renderer | pytorch3d: CUDA or CPU, no usable MPS path for its kernels |
| Avatar registration (tracker) | pytorch3d again, already runs on CPU by default (measured faster than GPU) |

There is exactly one hard blocker, and it is the Gaussian rasterizer.

### CUDA laptop: feasible today at reduced settings

| GPU | GAGAvatar 512 | mesh 512 | mesh 256 | peak VRAM (batch 8 / batch 4) |
| --- | --- | --- | --- | --- |
| A100 | 12.2 ms/f (0.33x) | 7.6 ms/f | 3.0 ms/f | 8.3 GB / 5.6 GB |
| RTX 8000 | 22.6 ms/f (0.60x) | 6.9 ms/f | 3.8 ms/f | 7.6 GB / 5.1 GB |
| P100 | 31.8 ms/f (0.82x) | 22.4 ms/f | 7.2 ms/f | 7.6 GB / 5.1 GB |

Two constraints. **VRAM**: 8 GB for batch 8, or about 5 GB at batch 4, so a 6 GB laptop GPU works with less headroom. **Launch overhead**: the renderer issues roughly 218 kernel launches per frame and is bound by that rather than by FLOPs (an A100 is only 2.4x a P100 here), so the laptop's single-thread CPU and driver overhead matter as much as its GPU tier.

A modern 8 GB laptop GPU should land near the RTX 8000's 0.60x, which is workable at 512 and comfortable at 384. The work is environment setup: building the rasterizer for the laptop's architecture and installing pytorch3d. No application changes.

### Apple Silicon: blocked, with three ways around it

1. **Render in the browser.** Send motion instead of video: 106 floats per frame is about 10 KB/s against a video stream, and the client GPU does the rendering. This avoids the CUDA extension entirely and works on any laptop with a current browser. Partial work already exists in the `web-based-renderer` worktree: a WebGPU upsampler runtime, quantized preview payloads, and frame prefetching. It also decouples us from server GPU supply, since the server would no longer render per client.
2. **Port the rasterizer to Metal**, or adopt an existing Metal Gaussian splatting implementation. Real work, and it buys Apple Silicon only.
3. **Mesh mode on CPU or MPS.** Probably runs, but it is the FLAME mesh face rather than the photoreal avatar, which undersells the demo.

Recommendation: option 1 is the strategic answer for both laptop classes, and it is the one with existing groundwork. A CUDA laptop running the full stack is a reasonable short-term path if someone needs an offline demo.

## 2. The other items

**Mic and speaker feedback loop: done and confirmed fixed.** Browser echo cancellation, noise suppression and automatic gain control are now explicit toggles, with both states forced so the comparison is clean rather than depending on browser defaults. Verified live rather than assumed: the loop reproduces with the processing turned off, and is gone with it on. Server-side half-duplex gating stays on the shelf in case a setup with external speakers defeats the browser's canceller.

**Integrating the new streaming ARTalk (Fallingwater): done, with a caveat.** It is wired in and selectable with `--motion-model fallingwater`. It runs in our existing environment, which was the main risk: the author uses torch 2.10 against our 2.4, and it needed only two small compatibility shims. Standalone it reaches 0.64x realtime on a P100. However it performs 176 sequential transformer forwards per chunk against ARTalk's 4, so combined with rendering at 512 it does not fit the realtime budget on our Turing card. Generation is also still chunk-wise at 4 s, so it does not lower the latency floor yet, though the architecture leaves room for sub-chunk streaming later. Quality has not been compared head to head.

**Retraining ARTalk: not started, and the biggest remaining item.** The data manifests (190.6 h) are ready and a 1 s chunk-size experiment spec is committed. This is where the real latency win is: the 4 s chunk floor dominates response latency, and no amount of rendering optimization touches it.

**Different conversational models: done.** A PersonaPlex bridge landed alongside the OpenAI one, plus endpoint presets for xAI Grok and a local speech-to-speech server. The OpenAI Realtime protocol is the shared interface, so further backends are mostly configuration. `audio_demo.py` runs the same bridges against a passthrough audio pipeline, so a backend can be judged by ear without the avatar's prebuffer in the way; it needs no GPU and no assets.

## 3. Unplanned findings and fixes

**The "avatar stops moving" bug is root-caused and fixed.** Streamlit's source watcher re-stats every loaded module's path on the same event loop that drives the WebRTC tracks, so the server stopped sending all audio and video for 0.3 to 3.7 s at a time. Confirmed from the browser (both tracks at zero packets, no packet loss, STUN round trips spiking to 4.2 s) and from the server with thread dumps captured mid-stall. Fixed in the app; a bug report with a dependency-free reproduction is drafted for upstream Streamlit in `docs/streamlit-source-watcher-issue.md`.

**The avatar now appears as soon as the pipeline is ready.** A fresh session used to render nothing until the first response arrived, so the demo opened on a blank frame and only came alive once someone spoke. The pipeline now renders the avatar at rest straight away, and the silence padding that already covered the gaps between turns was extended to cover the gap before the first one, so the model is producing frames from the start rather than waiting for input. Measured with no input at all: rendered frames are served from 0.4 s after the pipeline is ready. It matters more for demos than the diff suggests, since the first thing anyone sees is now a face rather than a blank panel.

**We were sharing a GPU without realizing it.** The app ran on a card already occupied by another tenant's job and pinned at its 260 W power cap, giving 1.66x realtime instead of the 0.60x we benchmarked, while the second GPU on the machine sat idle. Demos need a pinned, dedicated GPU: this alone accounted for a 2.8x slowdown and most of the stuttering we were chasing.
