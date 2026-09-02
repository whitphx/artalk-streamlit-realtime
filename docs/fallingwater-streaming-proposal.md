# Sub-chunk streaming for Fallingwater: measurements and a proposal

Follow-up to the K/V-cache work on the `perf/cache-constant-kv` fork branch (2.55 s -> 1.95 s per 4 s chunk, bit-identical). Two measurements answer the questions that blocked going further, and together they make a concrete case that the 4 s chunk floor can drop to ~160 ms without changing the codec.

## Measurement 1: the accumulator dirties at most 3 decode steps back

The blocker for a real self-attention KV cache was `vqidx_to_accum_next_feat`: it rebuilds every position's input feature each step through area interpolation, so already-decoded positions were not provably frozen. Instrumenting one full chunk decode (176 steps) and diffing the token-embedding sequence between consecutive steps, in decode order:

- 97 changes to already-decoded positions across 176 steps, in a strictly periodic pattern (a 1/2/1-position ripple with period 7, always in scales 2-3).
- **Maximum lag of any change: 3 decode steps** behind the write head. Nothing further back ever changed.
- Swept over 48 chunk decodes — speech, noise and silence inputs, three seeds, cfg 2.0 and 1.0, first chunks and chunks with real previous context — the maximum lag is 3 in every case. The window is structural (it comes from the fixed interpolation weights), not data-dependent.

Bounded dirty inputs alone would not make a cache exact if attention could reach positions that change later. It cannot: the within-chunk self-attention mask is built from `id_to_seq` as `decode_step(query) >= decode_step(key)` (`build_attn_mask`), i.e. **decode-order causal**. A cached position's deep layers therefore depend only on positions decoded before it, and with the dirty window bounded at 3, recomputing the trailing 4 positions per step against cached K/V for everything older is exact by induction — 4 query tokens instead of 176. A drift probe agrees: across one full chunk, the decoded row's logits between consecutive full recomputes are identical to the bit at every step where its input was unchanged (99 of 175), exactly as the causal mask predicts.

Estimated win on top of the constant-K/V cache (superseded, see below): another ~2x on paper. **Measured 2026-09-02: no win on P100.** A clean windowed decode (per-layer K/V buffers with the prefix folded in, decode-order bias and rope precomputed, no per-step allocation) runs at 0.99x versus the constant-K/V baseline; a naive first cut was 0.78x. The mechanism is exact (teacher-forced committed-logit diff 5.7e-6 across all 176 steps). It does not speed up decode because windowing shrinks the tensors in each step but leaves the operation and kernel-launch count unchanged: every step still runs all six layers over the same modules, and this decode is launch-overhead-bound, so wall-clock tracks the number of launches, not their size. The consequence for this ladder: on launch-bound decode only *fewer steps* help, which is what the 1 s-chunk retrain and stage 2 below deliver — not a smaller per-step window. One implementation note: exact output comparison against the stock loop must be done on logits under teacher forcing, because the stock sampler draws from the RNG for all 176 rows each step and a windowed decoder would not, so sampled bits diverge for RNG reasons alone; the distributions are identical.

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

One stage, since the first was measured out:

1. **Windowed KV cache on the existing checkpoint — tried, no win (2026-09-02).** Querying only the trailing 4 decode positions per step is exact (teacher-forced logit diff 5.7e-6) but runs at 0.99x on P100: the decode is launch-bound, and windowing shrinks tensors without reducing the per-step operation count. Kept here as a recorded negative; the mechanism is sound and could pay off only where launch overhead is removed (CUDA graphs / compile over the AR loop), which is a separate hard problem.

2. **A time-major decode variant, retrained.** The model is already strictly causal at 80 ms (the audio mask is built with zero look-ahead, verified by perturbation upstream). What keeps it chunked is only the coarse-to-fine decode order. A generator trained to decode time-major over the same codec tokens, with a KV cache and an incremental accumulator in place of the per-step rebuild, would emit motion every 80 ms token: the latency floor becomes one token plus render, on the order of 160-200 ms, instead of 4 s. The codec, the audio encoder and the training data pipeline all stay as they are.

The retrain belongs upstream; the measurements here, the constant-K/V branch, and the dirty-window trace are the supporting material. It is the real lever because it cuts the number of decode steps, which is what launch-bound decode is bound by — unlike the windowed cache, which cut only per-step tensor size and so did nothing.

## Where this leaves the latency ladder

| configuration | floor |
| --- | --- |
| today, 4 s chunks | ~4 s chunk + decode |
| 1 s-chunk retrain (in flight) | ~1 s chunk + decode |
| time-major retrain (stage 2) | ~2 tokens (160 ms) + render |

The 1 s ARTalk retrain and this proposal are complementary rather than competing: the retrain is the near-term win on the model we control; this is the path below one second on the successor model.
