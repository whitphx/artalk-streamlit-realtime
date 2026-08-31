# Realtime VLA research, and what applies to this stack

A reading list of realtime vision-language-action work was shared for us to mine for architectural ideas. This is what transfers to ARTalk, Fallingwater and GAGAvatar, what does not, and why. Claims about the papers below were checked against the arXiv abstracts rather than taken from the summaries that came with the list.

## The papers split into two groups, and only one is about our models

Six of them (Hume, StreamVLA, tau0-VLA, Hi Robot, pi0.7, ReSteer) are about the policy deciding *what to do*. In this system that layer is the OpenAI Realtime API or PersonaPlex: a model we neither train nor restructure. They describe how to organize the orchestration above the avatar, not how to change the motion models. That distinction is worth making explicitly, because "adopt a System 1 / System 2 architecture" reads like a model change and is a bridge-layer change for us.

Four of them (RTC, VLASH, Reflex, StreamPI) are about executing a slow generative model in real time, which is exactly our problem. The transferable material is there.

## Applies to ARTalk and Fallingwater

**Reflex generalizes work already landed.** Its core move is partitioning the attention context into static, sliding and dynamic regions so the static part gets O(1) cache updates while preserving full-batch-equivalent attention. Caching Fallingwater's per-chunk-constant audio, previous-motion and style K/V is precisely their static region, and it measured 1.30x with bit-identical output. What Reflex adds is the structure for going further: the remaining win needs incremental caching over already-decoded tokens, which is currently blocked because `vqidx_to_accum_next_feat` rebuilds the whole sequence each step through area interpolation, so decoded positions do not stay frozen. Their framing suggests treating that boundary explicitly, with a static prefix, a sliding recompute window sized by the interpolation's reach, and a dynamic tail, rather than treating the cache as all-or-nothing. Reflex reports 2.58x and up to 54% lower reaction latency from the full treatment.

**StreamPI bears on the retrain.** Two ideas, both cheap: temporal memory with no added parameters (bidirectional attention within a unit, causal across units), and random-interval training for robustness to variable frame intervals. Our frame interval is fixed, so the second does not transfer literally, but the mismatch it targets exists here in another form and is recorded in the retraining spec: every turn ends on a zero-padded partial chunk, and every turn begins with a pure-silence previous context, neither of which appears in random crops of continuous speech.

**ReSteer bears on barge-in, as a data problem.** Following an instruction and switching instruction mid-execution are separate capabilities needing separate data. The avatar equivalent is speech cut off mid-utterance. The catch is recorded in the retraining spec: the runtime discards remaining motion on barge-in rather than playing a transition, so this data only pays off alongside a runtime change.

## Applies to GAGAvatar

Only indirectly. Reflex's operator fusion and asynchronous encode/action pipelines target the same disease as our renderer, which is launch-overhead-bound at roughly 218 kernel launches per frame (an A100 is only 2.4x a P100 here, and CUDA-graph capture bought 4% on the RTX 8000). Nothing in this list addresses Gaussian-splat rendering itself. Our render cost stays a graphics problem, and browser-side rendering remains the larger structural answer.

## What does not transfer

**RTC needs a diffusion or flow policy.** The paper is explicit that it applies to diffusion- or flow-based VLAs with no retraining, by freezing the actions guaranteed to execute and inpainting the rest. ARTalk and Fallingwater are discrete autoregressive models over BSQ tokens, and a partially-frozen chunk of discrete tokens cannot be inpainted the way a partially-frozen continuous trajectory can. The principle already exists here in its cheap form, since Fallingwater conditions each chunk on the previous chunk's motion and the media clock defines exactly which frames are committed. Adopting RTC properly would mean moving the motion head to flow matching, which is a redesign rather than a patch. Anyone planning around "applies without retraining" should know it does not apply to this model class.

**VLASH's future-state prediction is mostly moot for lip-sync.** It corrects state drift during inference, where the robot has moved by the time the action is ready. Our motion is a deterministic function of audio already received, so there is no drift to correct. It does apply one level up, where the conversation can move while the LLM generates, which is the dialogue layer again.

## The one System-2 idea that lands on the avatar

Hume's asynchronous split has an avatar analogue that has nothing to do with the LLM: a fast local policy for non-verbal behavior. Blinks, gaze shifts, micro-expressions and backchannel nods could run continuously and independently of speech-driven motion. The avatar today is entirely audio-driven and does almost nothing when not speaking; a resting frame and idle padding were added only recently to stop it being a blank panel. A lightweight always-on behavior policy would improve perceived liveness at negligible compute cost and needs no change to ARTalk.

## Recommendations

1. Fold the deployed-chunk-composition point into the chunk-size retrain. Cheapest real win, and it addresses a mismatch already measured in production.
2. Use Reflex's static/sliding/dynamic framing to structure the bounded-window KV cache experiment on Fallingwater, continuing from the 1.30x already landed.
3. Add interruption sequences to the retraining data only if the runtime is changed to render a closing transition.
4. Treat the System 1 / System 2 papers as orchestration guidance, and separately consider a small idle-behavior policy.

None of this displaces chunk-size reduction as the dominant lever. Going from 4 s to 1 s cuts the latency floor fourfold, and nothing in this list touches that.

## Sources

- Hume: Introducing System-2 Thinking in Visual-Language-Action Model, arXiv:2505.21432
- StreamVLA: Breaking the Reason-Act Cycle via Completion-State Gating, arXiv:2602.01100
- ReSteer: Quantifying and Refining the Steerability of Multitask Robot Policies, arXiv:2603.17300
- tau0-VLA: A Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation, https://tau0-vla.github.io/
- Hi Robot: Open-Ended Instruction Following with Hierarchical VLA Models, arXiv:2502.19417
- pi0.7: A Steerable Generalist Robotic Foundation Model, https://www.pi.website/download/pi07.pdf
- Real-Time Execution of Action Chunking Flow Policies (RTC), arXiv:2506.07339
- VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference, arXiv:2512.01031
- Reflex: Real-Time VLA Control through Streaming Inference, arXiv:2607.14695
- StreamPI: Streaming Multimodal Temporal Modeling for VLA Models, arXiv:2608.26067
