# Running the Realtime Pipeline on Cheaper GPUs — Design Plan

Goal: serve the ARTalk + GAGAvatar realtime app with sustained headroom on
GPUs far below A100 class — consumer cards (RTX 3060 12 GB / 4060 8 GB) and
entry cloud parts (T4 16 GB, L4) — and publish a measured "minimum viable
GPU" matrix alongside the library release.

## Where we start (measured)

The stack already runs well below A100: on a 2016 Tesla P100 16 GB, a
4-second GAGAvatar 512 chunk renders in ~1.8 s (realtime ratio 0.35–0.44,
zero underruns). The problem is therefore not "make it possible" but
"widen the envelope": more sustained headroom (contention absorption — see
the 2026-07-29 jitter findings), less VRAM (currently ~8.5 GB reserved,
which excludes 8 GB cards), and less PCIe traffic (cheap hosts have narrow
links).

Cost structure per 4 s chunk (from profiler traces, GAGAvatar 512 batch 8):

| Component | Cost | Notes |
| --- | --- | --- |
| ARTalk streamer feed | ~140 ms | audio encoder dominates |
| Render 96–100 frames | ~1.5–1.75 s | conv stacks + upsampler + rasterize |
| RGB float32 → CPU | ~130 ms + 25 MB/batch PCIe | uint8 would be 4x smaller |
| Kernel launches | ~21,800 per chunk (~218/frame) | overhead-bound on weak CPUs |

## Levers, ordered by measured value / risk

### Tier 1 — precision and transfer (low risk, largest VRAM + speed wins)

1. **fp16 autocast** for the GAGAvatar conv stacks, upsampler, and ARTalk
   model. Expect ~1.5–2x on tensor-core GPUs (T4/RTX/L4) and halved
   activation VRAM. The Gaussian rasterizer stays fp32 (its CUDA extension
   is fp32-only) with casts at its boundary; FLAME LBS stays fp32 under
   autocast's default op policy. Gate: parity harness (below) must pass.
2. **GPU-side uint8 conversion before the device-to-host copy** (shelved
   optimization): scale/clamp/permute on GPU, copy uint8 — 4x less PCIe
   per batch and removes the CPU-side conversion.
3. **channels_last memory format** for the conv-heavy paths — near-free
   throughput on tensor cores.

### Tier 2 — configuration envelope (zero code risk, product decisions)

4. **Quality presets**: render-res ladder (512 / 384 / 256) exposed as
   named presets; measure quality (the upsampler already super-resolves,
   so lower Gaussian-render resolutions may degrade gracefully).
5. **Auto batch sizing**: pick the largest render batch that fits the
   detected VRAM; document `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`
   for 8 GB cards.

### Tier 3 — structural compute (medium effort)

6. **Kernel-launch overhead**: ~218 launches per frame is
   launch-overhead-bound on cheap CPUs. CUDA Graphs via
   `torch.compile(mode="reduce-overhead")` on the per-batch renderer
   forward — shapes are static per batch size, which is the favorable
   case. Gate by compute capability >= 7.0 (Pascal lacks triton support —
   the same constraint the tracker hit); the existing warm-up pre-roll
   absorbs compile time.
7. **ARTalk audio encoder** fp16 (+compile where supported) — trims the
   chunk-latency floor contribution.
8. **Lighter upsampler** (distillation/pruning or architecture swap) —
   the largest single consumer; requires model-team coordination, ties
   into the retraining track.

### Tier 4 — serving strategy (no code)

9. **Right-sizing guidance**: the contention investigation showed a
   dedicated smaller GPU beats a shared big one for this workload — we
   need sustained 2–3x headroom, not peak FLOPs. The deliverable matrix
   makes this concrete for deployment choices.

## The backbone: benchmark + parity harness first

Extend the existing headless test battery into a reusable harness that,
for each (device, resolution, batch, precision, compile) configuration,
reports: realtime ratio, peak VRAM, turn-first-frame latency, underruns
under a synthetic contention load, and quality parity (PSNR/SSIM of
rendered frames vs the fp32/512 reference, plus motion bit-parity where
expected). Every optimization lands only through this gate. The NFS-shared
checkout means the same harness runs on every available host (P100,
Turing, A100) unchanged.

## Acceptance targets (proposal)

- T4 16 GB: ratio <= 0.5 at 512 (2x sustained headroom).
- 8 GB consumer (4060-class): fits in VRAM and ratio <= 0.5 at 384.
- P100 16 GB: remains supported on the non-compile path (current status).
- Quality: no visible degradation at 512 fp16 (PSNR gate vs fp32).

## Phasing

- **P0** — benchmark harness + baseline matrix on the hosts we have.
- **P1** — uint8 DtoH + fp16 autocast behind flags, parity-gated;
  re-measure the matrix.
- **P2** — channels_last + torch.compile/CUDA Graphs on sm_75+.
- **P3** — quality presets + auto batch sizing + VRAM floors; publish the
  min-viable-GPU matrix (feeds the pip/Hugging Face release docs).
- **P4** — upsampler diet with the model team (parallel, longer horizon).
