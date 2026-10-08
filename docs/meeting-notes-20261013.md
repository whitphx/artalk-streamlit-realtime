# Status for the meeting (2026-10-13)

Everything below is measured. Eval numbers are LVE/MHD in mm on the 485-clip held-out split, run on an A100 through the production streaming path with a fixed seed per clip and without the live smoother; the 4 s released model is the baseline. Last cycle's evals ran on an H200, and the A100 rerun differs only in the last digit (1 s model LVE 8.55 here vs 8.56 then). Blink rates use one detector throughout: a dip of the eyelid aperture below 0.6 of its local open level, at most 480 ms long.

## 1. Results for the directions agreed on 2026-09-30

**Eye blinking without retraining: the 1 s model emits almost no blinks and none in silence; a runtime blink injector restores a natural rate with mouth and head untouched.** It is on main and enabled in the Space's configuration.

- Measurement first, because our previous minutes said blinks pass through the silence handling while the demo showed none. Over all 485 clips, ground truth blinks 16.8 times per minute. The 1 s model emits 0.99 per minute (6% of ground truth) and the 4 s release 1.51 (9%). In 60 s of silence both emit none. The ratios depend on the threshold: counting deep closures only, as here, the models reach 6 to 9% of ground truth; counting shallow dips as well, 20 to 40%.
- The 4 s release behaves differently with a style input. With the app's default style it blinks at a ground-truth-like rate while speaking (about 21 per minute on 26 clips). The 1 s model stays at about 1 per minute with or without style. In silence neither model blinks, with or without style. So the blinkless face in the demo is the 1 s model's, and the idle face of both models.
- FLAME has no eyelid blendshape; closure lives in the expression coefficients. GAGAvatar does render a closure driven by FLAME expression.
- The injector adds the corpus's average blink direction (closed-eye minus open-eye expression, from ground-truth blinks) to the 88 expression dimensions outside the mouth set. Mouth, jaw and head dimensions stay bit-identical, and with the flag off the output is byte-identical to the current pipeline. Intervals are drawn from a log-normal fitted to ground-truth intervals (16.6 per minute); each blink closes faster than it opens, about 470 ms in total. Blinks the model makes itself count toward the target rate, through a small credit counter (at most 2 credits), and no injected blink starts during a model blink.
- Scale 2 of the corpus direction is the default: in GAGAvatar renders on two avatars, closure deepens from 1.5 to 2 and stops changing from 2 to 2.5. One residual: on one avatar the far eye keeps a slit when the head is turned.
- Rates with the injector at scale 2, on 12 test clips streamed back to back (7.9 min, ground truth 20.6 per minute there, 5 seeds): 1 s model 13 to 18 per minute without style and 16 to 20 with style; 4 s release 12 to 20 without style. With the release and the default style the combined rate is 25 to 27 per minute, because about 7 injected blinks per minute still pass while the model blinks on its own. That is above this stretch's ground truth but within the human range (the 90th percentile of per-clip ground-truth rates is 45 per minute), and we kept it so the face recovers quickly after a turn. In silence the injector gives 17.3 per minute for both models.
- Turn boundaries: in the first 10 s of silence after speech, the injector runs at 12.9 per minute. A sliding 60 s rate window that we tried first gave 2.0 per minute there, i.e. a stare after every speaking turn.
- Live verdict on side-by-side clips: blinks look almost natural, the rate looks right, and closure at scale 2 is not complete but acceptable. The lips visibly dropped on each blink.
- Lip motion: FLAME couples eyelids and lips, so with the corpus blink direction the lip landmarks move about 1.0 mm at the blink peak even though the mouth dimensions are untouched, against about 0.2 mm attributable to natural blinks. The default is now a lip-cancelling direction, a regularized fit that keeps every eyelid aperture identical while moving the lips about 0.25 mm on the mesh (the same across 200 random face shapes). In GAGAvatar renders on four avatars the mouth motion during a blink drops by 30 to 60%; a small residual (about 0.4 to 1.3 px at 512 px) remains because the renderer responds to the expression as a whole.
- The cost is distance from real blinks: at the peak, about 27 expression coefficients fall outside the range seen in tracked data, against about 9 for the corpus direction. No odd expressions appeared on the four avatars rendered, and a launch flag switches back to the corpus direction.

**Sliding 4 s window on the 4 s model, keeping the last 1 s: implemented and evaluated; not adopted.**

- Each 1 s step runs the released 4 s model unchanged on the last 4 s of audio and commits only the newest 25 frames. The decision rule was written in advance: adopt if LVE is within 2% of the release and the chunk-boundary jerk is no worse than the 1 s model's.

| 485 clips | LVE | MHD | FDD | motion energy vs GT | seam jerk ratio |
|---|---|---|---|---|---|
| sliding 4 s window | 7.82 | 1.79 | 29.3 | 0.61 | 4.90 |
| 4 s release (whole chunks) | 7.85 | 1.77 | 33.4 | 0.61 | 1.24 |
| 1 s model | 8.55 | 1.91 | 27.0 | 0.87 | 1.24 |

- Lips pass: LVE matches the release (7.82 vs 7.85) at 1 s latency. Seams fail: at every 1 s commit the motion jumps about 5x more than within the output's own frames (seam jerk ratio 4.90, against 1.24 for the 1 s model and 0.99 for ground truth). The sliding window is worse than the 1 s model on this measure in all 485 clips. Renders show it plainly, for example the mouth going from open to closed in a single frame. The live smoother would soften seams somewhat; that was not measured.
- The comparison is about how much the seam stands out from the output's own motion. In absolute terms the sliding window's boundary jerk is 1.34x the 1 s model's without head pose and about equal with it, because the 1 s model moves much more overall.
- It also inherits the release's under-animation (motion energy 0.61 of ground truth vs 0.87 for the 1 s model) and costs about 4.2x the release's decode per output second (75 ms per 1 s step on an A100, about 1.1x the 1 s model).
- Why: the codec tokenizes a whole 4 s window jointly and coarse to fine, so every window regenerates the 3 s that were already shown, and the kept second continues that regenerated past rather than what the viewer saw. Conditioning on the model's own previous window instead was worse (LVE 8.20, seam ratio 6.63).

**Client-side rendering (WebGPU) and the motion model on a client CPU: feasible with the FLAME mesh only. The CPU is not faster than the GPU.**

- Motion model on CPU, 8 pinned cores of a 2016 Xeon with native PyTorch: the 1 s model takes 401 ms per 1 s chunk (0.40x realtime), 4.7x slower than a P100 and 6.1x slower than an A100 on the same eager path. The 4 s release takes 812 ms per 4 s chunk (single clean run). Kernel-launch overhead is real on GPU, but it costs tens of ms where the CPU's arithmetic costs hundreds. This is not a laptop and not a browser runtime, which would likely be slower still. The GPU reference timings come from an earlier 1 s checkpoint of the same architecture.
- A client CPU can still run the motion model in realtime, at about 0.3 to 0.4 s of extra latency per chunk. But see section 4: on some clips the 1 s model's output on CPU differs badly from its GPU output, so quality needs checking before relying on this.
- Photoreal rendering: GAGAvatar's cost is its neural upsampler, 48.7 M parameters and 137 GFLOP per 512 px frame, so 25 fps needs about 3.4 TFLOP/s sustained. A June browser prototype running it through ONNX Runtime Web on WebGPU reached about 2 fps end to end (including fetching server-rasterized input; the browser hardware was not recorded). On CPU it takes 0.62 s per frame.
- Library gap: the per-frame dynamic data is small (5,023 head Gaussian positions, the same as the mesh), but GAGAvatar composites 32 feature channels per Gaussian before the upsampler, and none of the browser splatting libraries we surveyed (Spark, GaussianSplats3D, Babylon.js, PlayCanvas, Visionary, LAM's web renderer) is documented to composite more than RGB. Visionary, built for per-frame generated Gaussians, is the closest fit. A custom 32-channel WebGPU rasterizer would be modest work but only helps once the upsampler is fast enough.
- FLAME mesh fallback: already runs in the browser, and needs 424 bytes per frame if the client evaluates FLAME from the 106-float motion code, or 60 KB per frame if the server sends vertices. Shipping FLAME to browsers needs a check against its licence, which restricts redistribution.
- Not attempted: building a WebGPU renderer (the ask was feasibility).

**Local conversation model: PersonaPlex and the avatar run together in realtime on one 80 GB A100; it does not fit a 24 GB L4 without quantization.**

- PersonaPlex-7B and the realtime avatar (1 s motion model plus GAGAvatar at 512 px) ran together for a 120 s exchange through the existing bridge. Both kept up with realtime, with no audio underruns.
- PersonaPlex took 71 ms per 80 ms frame on average, against about 47 ms alone. 38% of frames went over budget, but only in short bursts, and it never fell more than one frame behind. The avatar's compute went from 38% to 82% of realtime. The GPU was busy about 89% of the time, so headroom is roughly 10 to 20%.
- Latency from model emission to the listener hearing it stayed at about 2.9 s with no drift; at most about 0.5 s of that is from sharing the GPU.
- Peak GPU memory was 32.5 GiB, about 8 GiB of it the avatar's allocator cache rather than live data. That fits a 40 GB A100 but not the 24 GB L4 at bf16. With int8 PersonaPlex and a trimmed avatar cache we estimate about 17 GiB; whether an L4 is fast enough is unmeasured and doubtful (about a seventh of the A100's memory bandwidth, and the A100 was already 89% busy).
- Limits: one run, one avatar, continuous test speech. Interruptions (barge-in) were not tested. PersonaPlex is English-only.

**A/V sync drift after switching the avatar face: reproduced, root-caused, and fixed behind an opt-in setting.**

- The drift happens only on the first switch to a photoreal face in a process while a session is playing, which is the Space's normal flow because it opens on the mesh. Later switches are clean.
- Cause: that switch builds GAGAvatar on the script thread, and loading its weights holds Python's GIL in chunks of about 1 s. The loop that paces the WebRTC tracks stalls, the browser receives a gap of about 1 s and then a burst, and Chrome's jitter-buffer recovery leaves audio 130 to 220 ms ahead of video for 20 to 40 s, decaying to about 40 ms after a minute. Skipped warm-up and an unreset prebuffer were ruled out; the server-side A/V alignment stays within one frame throughout.
- Fix: an option, off by default, that loads the photoreal renderer at the first page load instead of at the first switch. With it on, the gap and the offset are gone. It is on main and enabled on the Space; it costs some startup time and VRAM.
- Evidence and limits: three cold reproductions without the fix and one run with it, all on a P100 with Chrome over loopback; not yet tested on the L4 Space. Absolute browser offsets carry a constant bias of about ±30 ms, so the before/after comparison is what counts. A motion-model switch mid-session causes a smaller drift of the same kind, which this fix does not cover.

## 2. Deployment

- The realtime Space was redeployed on 2026-10-08 with the current main: the face-switch preload and the blink injector (lip-cancelling by default) are enabled there. Neither has been checked live on the L4 yet.
- No new model was trained this cycle; the 1 s head-pose checkpoint and the 4 s release remain the models of record.

## 3. Corrections to the previous minutes

- Our own minutes of 2026-09-30 said the silent-mouth gate lets blinks pass through and the silence augmentation keeps real blinks. The silent-mouth gate is not blink-neutral. FLAME's expression basis couples eyelid and mouth, and about a quarter (26 to 29%) of the eyelid closure in tracked real blinks is carried by the 12 dimensions the gate freezes. With the gate fully engaged, detected blinks drop from 16.8 to 2.8 per minute. In practice this costs no blinks today: the gate engages only on silent input, and in silence neither motion model produces blinks. The missing blinks in the demo come from the 1 s model (none in silence, about 1 per minute while speaking), not from the gate. The 4 s release blinks at about the real rate while speaking with the app's default style, so "the models never blink" holds for the 1 s model and for silence only.
- "Resolving a bug that blocked head motion transformation data from entering training": it was a data property, not a code bug. The delivered training build had head rotation zeroed in every clip; the rotation was recovered exactly (round-trip error 1e-7) from the tracking transforms and the dataset rebuilt.
- "Synthesizing training data containing silent segments with only head motion and eye blinking": the augmentation holds the mouth at the window's pose on real pause windows while eyes, brows and blinks stay real (2 to 3x less flapping). Most of the silence fix comes from the runtime silent-mouth gate (10 to 25x).
- "The retrained one-second model has been uploaded to Hugging Face": it is in a private model repo, not public. Making it public is still pending the dataset author's OK.
- "For single users, it might be faster than GPU due to kernel launch overhead": measured and refuted on our hardware (section 1). The GPU is 4.7 to 6.1x faster at batch 1 even without CUDA graphs.

## 4. Open questions / proposed next directions

1. Blinks: the comparison clips (off, corpus direction, lip-cancelling direction) are ready to show. If the combined rate with the 4 s release and its default style (25 to 27 per minute) reads as too much, a leaky-bucket credit counter would cut the leak without slowing the recovery after a turn. Removing the renderer's residual mouth motion would need a renderer-side change.
2. Face-switch preload: enabled on the Space; verify it on the L4. The motion-model switch drift is a separate, smaller fix.
3. PersonaPlex barge-in: a scripted interruption mid-reply through the bridge, judged by ear and by the playback offset. Full duplex has no explicit turn boundary and the listener hears audio about 2.9 s behind the model, so this is the largest open risk of that route. Hours on one A100.
4. int8 PersonaPlex, alone and next to the avatar, first on an A100 and then on an L4: the only way to fit the 24 GB card. GPU-day scale.
5. A cascaded local stack (ASR, LLM, TTS) is the alternative if English-only or the latency is unacceptable. The app has an endpoint preset for it, but nothing about it has been measured.
6. CPU vs GPU quality: on 2 of 3 clips checked, the 1 s model's LVE on CPU is 1.7 to 2.1x its GPU value (16.4 vs 7.7, 14.5 vs 8.3). The cause is unknown. It needs a proper check before a client-CPU motion model is relied on.
7. Possible train/test overlap: on some test clips the 1 s model's output follows ground truth almost exactly for 100+ frames, which suggests those clips share source videos with training data. Worth checking the split, since it would flatter every model's absolute eval numbers.
8. If rendering moves to the client, the server keeps only the motion model; how many motion-only sessions one GPU can serve is unmeasured, and the decode being launch-bound means contention may cost more than the sum.
9. Not attempted this cycle: mixed chunk sizes in one model and the frame-level follow-ups (context plus 2-frame lookahead, delay-line streamer). Neither was raised at the meeting; both remain available if wanted.
