# Sub-chunk streaming for Fallingwater: measurements and a proposal

Follow-up to the K/V-cache work on the `perf/cache-constant-kv` fork branch (2.55 s -> 1.95 s per 4 s chunk, bit-identical). Two measurements answer the questions that blocked going further, and together they make a concrete case that the 4 s chunk floor can drop to ~160 ms without changing the codec.

## Measurement 1: the accumulator dirties at most 3 decode steps back

The blocker for a real self-attention KV cache was `vqidx_to_accum_next_feat`: it rebuilds every position's input feature each step through area interpolation, so already-decoded positions were not provably frozen. Instrumenting one full chunk decode (176 steps) and diffing the token-embedding sequence between consecutive steps, in decode order:

- 97 changes to already-decoded positions across 176 steps, in a strictly periodic pattern (a 1/2/1-position ripple with period 7, always in scales 2-3).
- **Maximum lag of any change: 3 decode steps** behind the write head. Nothing further back ever changed.

So an exact windowed KV cache exists on the current checkpoint: process only the newest position plus the trailing 3 each step — 4 query tokens instead of 176 — refreshing the cached K/V of those trailing positions. The periodicity comes from the fixed interpolation weights, so the window is expected to be structural rather than data-dependent, but this trace covers one audio clip and one seed; validation across clips and seeds is part of implementing it. Estimated win on top of the constant-K/V cache: another ~2x, putting a chunk near 1 s of compute on a P100. The same bit-identity gate used for the landed cache applies.

## Measurement 2: the renderer sustains streaming cadence

A streaming decoder emits ~2 frames per 80 ms rather than 100 per 4 s, which shrinks the render batch — the wrong direction for a launch-overhead-bound renderer, so it was measured rather than assumed (GAGAvatar 512, P100):

| batch | ms/frame | vs batch 25 |
| --- | --- | --- |
| 1 | 38.2 | +27% |
| 2 | 34.9 | +16% |
| 4 | 32.4 | +8% |
| 8 | 31.1 | +3% |
| 25 | 30.1 | baseline |

Per-frame cost is nearly batch-independent, because launches dominate. Batch 2 stays under realtime on the slowest GPU we have, so **the consumer side does not block sub-chunk streaming**, and 1 s chunks (25-frame batches) are free relative to today.

## The proposal

Two stages, the first requiring no retraining:

1. **Windowed KV cache on the existing checkpoint.** Query only the trailing 4 decode positions per step, cache the rest. Combined with the landed constant-K/V cache this targets roughly 2.5-3x over stock decode. Output must stay bit-identical; the dirty-window measurement defines the window and the validation.

2. **A time-major decode variant, retrained.** The model is already strictly causal at 80 ms (the audio mask is built with zero look-ahead, verified by perturbation upstream). What keeps it chunked is only the coarse-to-fine decode order. A generator trained to decode time-major over the same codec tokens, with a KV cache and an incremental accumulator in place of the per-step rebuild, would emit motion every 80 ms token: the latency floor becomes one token plus render, on the order of 160-200 ms, instead of 4 s. The codec, the audio encoder and the training data pipeline all stay as they are.

Stage 2 belongs upstream; the measurements here, the constant-K/V branch, and the dirty-window trace are the supporting material. Stage 1 is implementable on our side under the same guarded-patch pattern as the landed cache, and its result feeds stage 2's design either way.

## Where this leaves the latency ladder

| configuration | floor |
| --- | --- |
| today, 4 s chunks | ~4 s chunk + decode |
| 1 s-chunk retrain (in flight) | ~1 s chunk + decode |
| stage 2 above | ~2 tokens (160 ms) + render |

The 1 s ARTalk retrain and this proposal are complementary rather than competing: the retrain is the near-term win on the model we control; this is the path below one second on the successor model.
