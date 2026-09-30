# Status for the meeting (2026-09-30)

Everything below is measured; eval numbers are LVE/MHD in mm on the 485-clip held-out split unless noted. The 4 s released model is the agreed baseline throughout.

## 1. Results for the directions agreed on 2026-09-16

**Longer audio input with 1 s output: implemented, evaluated, and it is not a quality lever — recommend not adopting.**

- Implementation (config-gated, default-off): the decoder cross-attends to up to N preceding 1 s chunks. Each chunk is encoded once and reused, so a streaming caller pays no extra encoder cost; training draws a variable number of valid past chunks per sample and masks the rest, which also makes the model robust in a stream's first seconds.
- Three trained arms (5 s variable, 5 s fixed, 2 s), ~20 scored configurations, all to 200k iterations. At convergence, every trained-for-context arm is at or below baseline on lips: trained-for-2-chunks LVE 8.84 vs baseline 8.72, with motion-energy overshoot (1.12 vs 1.02). The 5 s setting as proposed is the worst arm at every training stage.
- The one configuration that beat baseline (5 s weights *run* at 2 chunks: 8.53, 2.2% better) does not survive being trained for directly, so it reads as an off-distribution quirk, not a recipe.
- Why: the previous-motion context already summarises past audio; extra audio keys dilute attention.

**Head orientation (fixed head in the 1 s model): solved. It was a data property, not a chunk-size effect.**

- The delivered training build had head rotation exactly zero in every clip. The rotation still exists in the tracking transforms; it was recovered exactly (round-trip error 1e-7), the dataset rebuilt, and both stages retrained.
- The accepted checkpoint moves its head ~2x more than the 4 s release and ~1/3 of ground truth; side-by-side renders show natural conversational turning. Lips on the 485-clip eval are slightly better than the previous 1 s checkpoint.
- Known cost: the codec pays ~15% on lip reconstruction (0.57 vs 0.49 mm) because three formerly-constant dimensions now carry signal inside an unchanged code budget.

**Silence quality (lips flapping between turns): measured cause, two-part fix shipped.**

- Cause: in the corpus, the jaw during real pauses still moves 61% as much as during speech, and there are no still-mouth windows at all — so training on silenced real pauses teaches a moving mouth on silence.
- Fix 1 (training): the silence augmentation now holds the mouth at the window's pose while keeping eyes/brows/blinks real. Cuts flap 2–3x with quality parity on the full eval.
- Fix 2 (runtime, on by default, flag-gated): a silent-mouth gate freezes the 12 mouth-dominant dimensions when decoded audio is silent, with attack/release smoothing; blinks, brows and head sway pass through. Measured 10–25x flap reduction.
- A third retrain was evaluated and rejected: silence already settles at a near-closed mouth, so residual motion is small wobble behind the gate, not a held-open jaw.

**Prefill at load: already in place** — the pipeline warms up with a silence chunk, the idle pump keeps the model fed between turns, and the avatar renders from ~0.4 s after connection.

**Sliding window + KV cache: built, proven exact, and measured as no win — the decode is launch-bound.**

- An exact windowed KV cache (logit difference ~7e-6 from stock) runs at 0.96–0.99x — break-even — on both P100 and A100. An A100 is only 1.37x a P100 on this decode, hard evidence that wall-clock is kernel-launch count, not compute.
- Consequence: on this hardware only *fewer steps* help (shorter chunks, frame-level), not smaller steps. This redirects the latency effort and is worth stating as a settled negative.

**Frame-level model (40 ms in / 40 ms out): phase 1 complete — lookahead is the win.**

- Audio lookahead improves every metric monotonically: k=2 frames (+80 ms latency) gives −3.9% LVE with the motion-energy gate passed; k=5 (+200 ms) gives −6.3% LVE but damps motion below the gate. Context-window widening is real but ~6x smaller. Effects are 16–25x the measured seed-noise floor (0.025 mm on LVE).
- Even the best arm remains ~19% above the 4 s release on lips; the frame model's value is the 40 ms cadence, not lip parity yet.
- To ship the k=2 arm live, the streamer needs a K-frame delay line and the pipeline an A/V offset — bounded work, not started.
- Not attempted from the asks: mixed chunk sizes (1/2/4 s) in one model; superseded in priority by the context experiment, still open if wanted.

## 2. Deployment and performance since last meeting

- The realtime Space is live on an L4 with both motion models and Cloudflare TURN; the free ZeroGPU offline demo runs alongside. The 1 s checkpoint serves from a private model repo.
- The 1 s model's decode regression is fixed: 110 → 66 ms per chunk by default, 36 ms with opt-in CUDA-graph decode; validated on A100, P100 and RTX 8000 with bit-identical output. A crash class from concurrent graph captures across sessions was root-caused and fixed with a process-wide capture coordinator.
- Live A/B verdict on the 1 s model: lip-sync good; eye motion absent. The 106-dim motion layout carries no eye channels — the frame model's 108-dim layout does, which makes eye motion an argument for that direction rather than a retrain of the current model.

## 3. Corrections to the previous minutes

- The static-lips regression was not caused by the style wiring. The style bug was real and separately fixed, but the collapse was the silence attractor, fixed by the turn-start context reset plus the silence-robust retrain.
- "1 s decreased AR core performance" holds only for frame-matching metrics (LVE 8.7 vs 7.9). On dynamics the 1 s model is better: FDD 23 vs 33, motion energy 0.95 vs 0.61 of ground truth. No 2 s model has ever been measured by us.
- The ~50% speed win was the 1 s decode path, not GAGAvatar; GAGAvatar's fp16/CUDA-graph work was separate.
- The head-orientation regression was a data property (zero rotation in the delivered build), not a consequence of the 1 s chunk — and it is now fixed by recovering the pose from the tracking transforms.
- Our silence augmentation already used real pause windows, close to the "natural clips" ask; what the corpus genuinely lacks is idle/listening footage — no still-mouth windows exist in it at all.

## 4. Proposed next steps

1. Frame-level: train the untested combination (context + 2-frame lookahead), and build the delay-line streamer so the best arm can be judged live.
2. Eye motion: pursue via the 108-dim layout (real blink/gaze data exists in the source), which the frame direction already uses.
3. Decide whether the 1 s model repo goes public (author's data OK pending).
4. If mixed chunk sizes are still wanted after the context negative, scope it as one training run with the variable-mask machinery already built.
