# Portability and deployment plan

Written 2026-08-31. Goal: reproduce this app and its full dependency stack on a new CUDA machine in one move, launch it with a single command that applies the best known settings, and ultimately run the whole demo offline on a GPU laptop with local speech models replacing the OpenAI Realtime API.

Hardware market data below was researched 2026-08-31 (sources linked inline). Performance numbers come from `benchmarks/`, `docs/cheaper-gpu-plan.md`, `docs/meeting-notes-20260819.md`, and the PersonaPlex probe measurements (`~/src/personaplex-probe`, A100, 2026-08-17).

## 1. Target state

Acceptance criteria, in decreasing order of strictness:

1. **Fresh server bring-up:** on a clean CUDA host, `git clone` + one bootstrap command (deps + assets) + one launch command produces a working demo. No hand-built conda env, no sibling worktrees to arrange, no flag archaeology.
2. **Best settings by default:** the launch command detects the GPU and applies the measured per-tier preset (render flags, batch, prebuffer) without the operator knowing they exist.
3. **Offline demo:** an `offline` profile runs with zero outbound network: local conversation backend, no ngrok, no Hugging Face downloads, no STUN.
4. **One-go transfer:** a bundle artifact (image or tarball) moves the environment plus assets to an air-gapped machine.

## 2. What blocks this today

From the 2026-08-31 code assessment, ranked:

1. **Two unreleased sibling packages.** `artalk` and `gagavatar` are editable installs of `refactor/library-api-for-standalone-renderer` worktrees, installed `--no-deps`. Nothing in this repo pins their revisions. Fallingwater (`sys.path` insertion of its checkout) and artalk1s (`ARTALK1S_TRAIN_CODE_DIR`, same trick) add two more unpinned checkouts.
2. **No pinning of the ML stack.** torch 2.4.1+cu121, pytorch3d, and `diff_gaussian_rasterization_32d` exist only in the hand-grown mamba env. No lockfile, no `environment.yml`, no Dockerfile, no CI.
3. **CUDA extension arch coverage.** The rasterizer and pytorch3d must be built for the target GPU's architecture; mismatches fail silently or with nonsense errors (`docs/realtime-performance-notes.md`, the 131k-GiB "OOM"). This has burned the project twice.
4. **Second Python environment** for GAGAvatar_track (avatar registration), with its own ~1.1 GB of assets, provisioned entirely by hand.
5. **Assets are partly manual.** The downloaders cover ARTalk/GAGAvatar weights with hash verification, but not: style motions as a set, tracker assets, the artalk1s and Fallingwater checkpoints (the latter from a private GitHub release), the MOSS tokenizer (7 GB, silently downloaded from HF on first use), or the license-gated `FLAME_with_eye.pt` (manual by design).
6. **Undeclared dependencies.** `aiohttp`, `sphn<0.2`, `torchaudio`, `transformers` are imported by the PersonaPlex/Fallingwater/artalk1s paths but absent from `pyproject.toml`.
7. **Interactive mode requires the OpenAI Realtime API** (or xAI); the local alternatives (PersonaPlex bridge, `huggingface/speech-to-speech` preset) exist but are not packaged as a service anyone can start.
8. **Launch is multi-knob.** `ARTALK_STREAMLIT_PYTHON` + `run_app.sh` + st-remote/ngrok + a dozen app flags + `.streamlit/secrets.toml` + per-model env vars. The best per-GPU flags live in a doc, not in code.
9. **Version-fragile Streamlit patches.** `postScriptGC = false` and the source-watcher monkey-patch are load-bearing for realtime media and tied to Streamlit internals, so the Streamlit version must be pinned exactly.

## 3. Workstream A: reproducible environment

- **A1. Lock the Python stack.** Adopt a lockfile-based env manager (recommendation: `pixi`, since it locks conda-forge CUDA runtime pieces and PyPI deps together and replaces the mamba env one-to-one; `uv` alone covers PyPI only). Commit the manifest + lock: Python 3.12, torch (see A6 for the version fork), torchaudio, transformers, streamlit pinned exact, streamlit-webrtc, av, openai, aiohttp, `sphn<0.2`, numpy. Fix `pyproject.toml`: declare the missing deps, add extras `[personaplex]`, `[fallingwater]`, `[artalk1s]`, raise `requires-python` to match reality (3.12).
- **A2. Prebuilt CUDA extension wheels.** Build `pytorch3d` and `diff_gaussian_rasterization_32d` once per (torch, CUDA, arch-set) as fat binaries (`TORCH_CUDA_ARCH_LIST` covering all deployment archs) and store the wheels where the bootstrap can fetch them (GitHub release on the forks, or a simple wheel index). The env audit script (cuobjdump arch listing, recipe in local notes) becomes part of `demo doctor` (C4).
- **A3. Pin the sibling packages.** Replace "point env vars at worktrees" with git-pinned dependencies on the `whitphx` forks at tagged commits, installable by the lockfile. Vendor-or-pin the Fallingwater checkout and the ARTalk `train_code` directory the same way (fetched by bootstrap at a fixed rev, then handed to the existing `--fallingwater-dir` / `ARTALK1S_TRAIN_CODE_DIR` machinery). Long term this dissolves into the upstreaming ladder's publishing milestone (PyPI wheels for `artalk`/`gagavatar`), but the plan must not wait for it.
- **A4. Tracker environment as code.** A second locked env spec for GAGAvatar_track, provisioned by the same bootstrap, behind an opt-out (a deploy that skips registration skips the env and its 1.1 GB of assets). Registration stays in the demo by default; the ~30 s registration is a demo highlight.
- **A5. Container image for the server path.** Dockerfile on an `nvidia/cuda` base consuming A1's lock and A2's wheels; compose file with the app plus an optional PersonaPlex service; run under NVIDIA Container Toolkit. The image (via registry or `docker save`) is the "move in one go" artifact for lab/cloud servers. For the laptop, prefer the native pixi env from the same lock (audio devices, browser, and thermals are simpler without a container layer); the container remains an option there since Linux + toolkit supports it.
- **A6. torch 2.7+/cu128 migration branch.** Every purchasable current-gen GPU is Blackwell sm_120, which torch 2.4.1/cu121 cannot run (first stable support is [torch 2.7.0+cu128](https://pytorch.org/blog/pytorch-2-7/)). So the lock needs two flavors: `cu121` (existing lab hosts, sm_60/75/80) and `cu128` (new hardware, adds sm_120 to the arch list). Migration work: rebuild both extensions, rerun the parity + benchmark harnesses, revalidate the cuda-graph path and the Streamlit patches, retest Fallingwater (authored against torch 2.10, so likely fine). Also recheck the torch 2.4 inductor bug that blocked `torch.compile`; if fixed in 2.7, compile may supersede the manual CUDA graphs.

## 4. Workstream B: assets and bundles

- **B1. One asset command.** Extend the manifest/downloader coverage to everything the app can load: ARTalk checkpoints (wav2vec + artalk1s `iter_200000`), GAGAvatar + tracked.pt, style motions, tracker assets, Fallingwater checkpoint, and a MOSS tokenizer prefetch into the repo-local `HF_HOME` (first-use download looks like a hang today). Everything hash-verified. FLAME stays manual per its license: the command prints the instruction and verifies the file once provided.
- **B2. Offline bundle builder.** `demo bundle` producing a single tarball (image or env + asset tree + HF cache + optionally `user_avatars/`), and an install path on the target. This is the air-gapped transfer story.
- **B3. License gate for redistribution.** Before any bundle leaves our machines, check what may legally be inside it: FLAME (no redistribution, keep manual), MOSS tokenizer, PersonaPlex weights (NVIDIA open-model terms), ARTalk/GAGAvatar checkpoints, the private Fallingwater release. Output: a list of "in the bundle" vs "fetched per machine with instructions".

## 5. Workstream C: single-command launch and management

- **C1. `demo up` with profiles.** One entrypoint (a small CLI in this package, wrapping today's `run_app.sh` logic) reading a committed TOML config with profiles: `lab` (tunnel + OpenAI backend), `server` (LAN + SSH forward), `offline` (local backend, no network). Env vars remain as overrides. Secrets only required by profiles that need them.
- **C2. GPU presets as data.** Commit the measured preset table (GPU class → flags) and have the launcher auto-detect and apply it. Current table, from `docs/cheaper-gpu-plan.md`: A100/48 GB class `--render-uint8-gpu --renderer-compile`; P100 `--render-uint8-gpu` only; 8 GB tier uint8 + 384 preset; `--renderer-fp16` never (no throughput win, VRAM only); plus prebuffer/segment defaults. New hardware regenerates the row via `scripts/benchmark_pipeline.py`. The launcher should also default `--motion-model` per the latest gate (artalk1s once its live A/B passes).
- **C3. Process management.** systemd unit (native) or compose restart policy (container), `CUDA_VISIBLE_DEVICES` pinning baked into profiles, and the dedicated-GPU requirement documented at the launcher level (shared-GPU contention cost 2.8x once; the launcher should warn when the selected GPU already has resident compute processes).
- **C4. `demo doctor` smoke test.** Post-install validation: import closure per extra, CUDA arch audit of compiled extensions against the local GPU, asset hash check, FLAME presence, one-chunk headless pipeline render (harness exists), optional `audio_demo.py` passthrough for the audio path. This converts the historical silent-failure classes into a checklist.

## 6. Workstream D: fully local speech (offline demo)

The bridge layer is already backend-agnostic (OpenAI Realtime protocol as the shared interface, plus the native PersonaPlex bridge), so this workstream is model selection, packaging, and fitting, not app architecture.

- **D1. PersonaPlex as the primary candidate.** Measured on A100: 18.07 GiB bf16, 0.54x realtime, VRAM-flat, lag bounded at the prebuffer. Tasks: package the moshi server (with our client-side handshake gate; upstream `server.py` bugs documented in the probe notes) as a service the `offline` profile starts; **quantize to ~8-bit** — required on any 24 GB laptop for VRAM (18 + 8.3 GiB does not fit) and probably for speed, since 7B bf16 decode is memory-bandwidth-bound and laptop bandwidth is ~half an A100's; measure quality/latency after quantization by ear via `audio_demo.py`. English-only is acceptable for demos (Japanese is blocked on data regardless).
- **D2. Co-residency measurement.** Avatar renderer + speech model on one GPU has only a naive additive estimate (0.54x + 0.30x ≈ 0.84x on A100). Measure real contention on one lab GPU; define the go/no-go threshold for the laptop (e.g. combined ≤0.9x sustained with no underruns).
- **D3. Fallback backend.** The `huggingface/speech-to-speech` local preset (`ws://127.0.0.1:8765`) with small ASR/LLM/TTS components, if quantized PersonaPlex misses the latency or VRAM budget. Evaluate through the same `audio_demo.py` A/B.
- **D4. Offline network path.** No ngrok (localhost is already a secure context for mic access); verify streamlit-webrtc with host-only ICE candidates and no STUN server; `HF_HUB_OFFLINE=1`; confirm zero outbound calls in the `offline` profile (the OpenAI client must not be constructed).
- **D5. Audio hardware validation.** Laptop mic + speakers with the existing AEC/NS/AGC toggles; if external speakers defeat the browser canceller at a venue, the shelved server-side half-duplex gating is the fallback.

## 7. Workstream E: hardware selection

### Requirements derived from measurements

| Requirement | Value | Why |
| --- | --- | --- |
| GPU vendor/API | NVIDIA CUDA | `diff_gaussian_rasterization_32d` has no Metal/CPU path |
| VRAM, avatar only | ≥ 8 GB (batch 8 @512) or ~6 GB (batch 4) | measured peaks 8.3 / 5.6 GB |
| VRAM, fully offline | ≥ 24 GB with 8-bit speech model (~10 GiB) + avatar; 32 GB removes the squeeze | 18 GiB bf16 PersonaPlex + 8.3 GiB avatar = 26.5 GiB |
| Compute | RTX 8000 class or better → ~0.6x realtime at 512 | renderer is kernel-launch-bound; A100 is only 2.4x a P100 |
| CPU | strong single-thread | same launch-overhead reason |
| Memory bandwidth | as high as possible | 7B speech decode is bandwidth-bound |
| Sustained TGP | 150–175 W chassis (18") | thin chassis run 95–135 W and throttle sustained inference |
| OS | Linux native preferred; WSL2 workable for CUDA extensions | source-built extensions are painful on native Windows |
| System RAM | 64 GB+ | comfortable margin for HF caches, browser, tracker env |

### Market reality (researched 2026-08-31)

**24 GB is the hard ceiling for laptop VRAM.** The two 24 GB mobile parts are the GeForce RTX 5090 Laptop and the RTX PRO 5000 Blackwell Laptop, the same GB203 silicon (10,496 cores, 256-bit, 896 GB/s, 95–175 W; the PRO adds ECC and pro drivers) ([NVIDIA](https://www.nvidia.com/en-us/products/workstations/professional-laptops/), [VideoCardz](https://videocardz.com/newz/nvidia-announces-rtx-pro-blackwell-laptop-gpus-up-to-10496-cuda-cores-and-24gb-gddr7-memory)). Both are sm_120, hence the A6 torch migration is a prerequisite for any purchase. The largest Ada-generation (torch-2.4-compatible) laptop GPU is 16 GB, which the offline working set rules out.

Candidates:

| Option | VRAM | Portability | Price (approx) | Notes |
| --- | --- | --- | --- | --- |
| ASUS ROG Strix Scar 18 (RTX 5090 Laptop, 175 W) | 24 GB | true laptop | $3.4–4.3k | cheapest max-TGP config ([Best Buy](https://www.bestbuy.com/product/asus-rog-strix-scar-18-18-2-5k-240hz-gaming-laptop-intel-core-ultra-9-hx-32gb-ram-nvidia-geforce-rtx-5090-2tb-ssd-off-black/JJGGLHXJY3)) |
| Lenovo Legion Pro 7i / MSI Titan 18 HX (RTX 5090 Laptop) | 24 GB | true laptop | $4.7–4.9k | Titan configurable to 96–128 GB RAM |
| HP ZBook Fury G1i / ThinkPad P16 Gen 3 (RTX PRO 5000) | 24 GB ECC | true laptop | $5–12k realistic | Linux-certified workstation line, up to 192 GB RAM ([StorageReview](https://www.storagereview.com/review/hp-zbook-fury-g1i-18-inch-mobile-workstation-pc)) |
| SFF desktop, RTX 5090 FE (2-slot, SFF-Ready) + portable monitor | **32 GB**, 1.79 TB/s | luggable (~10–15 kg + wall power) | ~$4.5–5.5k prebuilt | whole stack fits unquantized; near-A100 decode bandwidth ([NVIDIA SFF-Ready](https://www.nvidia.com/en-gb/geforce/news/small-form-factor-sff-ready)) |
| eGPU: GIGABYTE AORUS RTX 5090 AI BOX (TB5) + thin laptop | 32 GB | two boxes + cables | $3.0k + host | ~14–27% interconnect penalty; hot-plug friction on Linux ([TweakTown](https://www.tweaktown.com/news/108094/aorus-rtx-5090-ai-box-thunderbolt-5-egpu-is-27-percent-slower-than-desktop-5090-in-gaming-for-dollars2999/index.html)) |
| ASUS ROG XG Mobile eGPU (RTX 5090 Laptop, TB5) | 24 GB | <1 kg brick + thin laptop | $2.5k + host | same 24 GB constraint as laptops |
| DGX Spark / GB10 clones (ASUS Ascent GX10) | 128 GB unified | mini PC | $3–4k | 273 GB/s bandwidth (worse than RTX 8000) → weak single-stream latency; ARM64 + sm_121 is off the paved torch road ([LMSYS review](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)) |

Compute read for our workload: the RTX 5090 Laptop beats the Quadro RTX 8000 on every axis except capacity (~2x FP32, 896 vs 672 GB/s), so the avatar should land at or better than the 0.60x Turing number at 512. The speech model is the risk: ~0.5x of A100 bandwidth puts bf16 PersonaPlex near or over 1.0x realtime, which is why D1's 8-bit quantization is on the critical path, not a nice-to-have.

### Recommendation and selection tasks

- **E1. Primary pick: an 18-inch RTX 5090 Laptop at 175 W with 64 GB+ RAM** (Strix Scar 18 class, ~$4k) running native Linux. It is the only true-laptop tier that can hold the offline stack, and it satisfies the mobile-demo goal directly. The workstation tier (ZBook/P16) buys ECC, ISV Linux certification, and more RAM for roughly 2x the price; worth it only if procurement prefers the enterprise channel.
- **E2. Hedge: SFF desktop RTX 5090 (32 GB) as the venue option** if quantization degrades PersonaPlex unacceptably or the combined load misses realtime on 24 GB. Decide after E4, not now. DGX Spark is not recommended for this latency-sensitive workload.
- **E3. torch 2.7/cu128 migration (A6) lands first.** No current-gen hardware runs without it.
- **E4. Pre-purchase validation on rented silicon.** Rent a desktop RTX 5090 (or RTX PRO 6000 Blackwell) cloud instance, run the cu128 stack there: `benchmark_pipeline.py` for the avatar row, the PersonaPlex probe bf16 vs 8-bit, and the D2 co-residency test. The laptop part is the same die at lower power, so this bounds the answer before spending $4k. Go/no-go: combined sustained ≤ ~0.9x realtime within ~22 GB.
- **E5. Purchase + bring-up:** install Linux, NVIDIA 570+ driver, run the A-workstream bootstrap (which by then must be the only setup step), regenerate the preset row (C2), and do a full offline dress rehearsal (D4/D5) off-network.

## 8. Phasing

| Phase | Contents | Depends on | Outcome gate |
| --- | --- | --- | --- |
| 0. Decisions | vehicle (container vs native per target), offline backend priority, hardware budget/class, registration in or out of the minimal bundle | — | decisions recorded here |
| 1. Reproducible env | A1–A5, B1, C1–C4 on existing lab hosts | 0 | fresh lab/cloud host to running demo in under an hour, one bootstrap + one launch command |
| 2. Offline profile | D1–D5 on a lab GPU | 1 | demo runs with the network cable pulled |
| 3. New-hardware readiness | A6 (cu128), E4 rental validation | 1 (harnesses), parts of 2 (quantized backend) | measured go/no-go for the 24 GB laptop |
| 4. Laptop | E5 purchase + bring-up, B2/B3 bundle for air-gapped copies | 2, 3 | offline dress rehearsal outside the lab |

Phases 1–2 are pure software and start now on existing hardware. Phase 3 is the purchase gate. The browser-rendering direction (send 106 floats/frame, render client-side; groundwork in the `web-based-renderer` worktree) remains the strategic long-term answer that would remove the CUDA-laptop constraint entirely, but it is a separate project and does not block this plan.

## 9. Status

Phase 1 implementation landed 2026-09-01: `pixi.toml`/`pixi.lock` (conda pytorch/pytorch3d layer + pinned PyPI layer + git-pinned `artalk`/`gagavatar`, rasterizer as an in-repo prebuilt fat wheel), `scripts/bootstrap.sh` (env + pinned vendor checkouts), `scripts/provision_track_env.sh` + `envs/track-requirements.txt` (A4), `artalk-demo` CLI (`up` with `launch.toml` profiles/presets, `assets`, `doctor`), and `Dockerfile`/`compose.yaml` (A5, not yet built on a docker host). Open from phase 1: A2 wheel hosting for pytorch3d source builds, the ARTalk fork branch needs a push before `vendor/ARTalk` resolves for other machines, and a fresh-host validation run.
