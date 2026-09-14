# Publishing the Realtime Demo to Hugging Face Spaces

Goal: a public Space running this app on a Hub GPU, with the ARTalk 1 s model published as a Hub model repo next to the original ARTalk and GAGAvatar weights, and the app offering both motion models. Written 2026-09-14 against `main` at 187d251.

## What the survey found

- **Docker SDK is the only fit.** The runtime is a conda PyTorch 2.4.1 + pytorch3d + prebuilt rasterizer stack under pixi; the Streamlit SDK cannot carry that. Docker Spaces expose one port (`app_port`, default 7860), run the container as uid 1000, have no GPU at build time, and allow outbound traffic only on ports 80, 443 and 8080. Creating a Docker Space needs a PRO account or a Team/Enterprise org.
- **ZeroGPU is out.** It is Gradio-only and time-slices the GPU per call; the pipeline here is a persistent worker thread holding CUDA graphs and WebRTC tracks.
- **WebRTC needs a TURN relay.** Browsers cannot reach the container directly through the Spaces proxy, so every session relays media. The pinned `streamlit-webrtc` 0.74.1 (and `main`) only ships `get_hf_ice_servers`, which talks to the community endpoint at `fastrtc-turn-server-login.hf.space`; FastRTC has deprecated that endpoint in favour of Cloudflare TURN under the Hugging Face partnership (10 GB of relayed traffic per month free with an `HF_TOKEN`, then paid). Nothing in `streamlit_app.py` passes `rtc_configuration` today.
- **Public weights are already on the Hub.** The asset manifests pin `xg-chu/ARTalk` (2.0 GB `ARTalk_wav2vec.pt`, `tracked.pt`, style motions) and `xg-chu/GAGAvatar` (`GAGAvatar.pt`, 0.75 GB) by revision and SHA-256. The 1 s model additionally pulls `facebook/wav2vec2-xls-r-300m` (about 1.2 GB) through `transformers` on first use.
- **FLAME cannot be redistributed.** The FLAME model license forbids copying or sharing and says third-party access needs written permission from Max Planck; a public web demo is not addressed. `FLAME_with_eye.pt` (27 MB) is the ARTalk author's packaging of FLAME 2020 with eyes, not the CC-BY FLAME 2023 Open model. The existing Dockerfile already keeps it out of the image for this reason.
- **The rasterizer is research-only.** `diff_gaussian_rasterization_32d` carries the Inria/MPII Gaussian-Splatting license: non-commercial research and evaluation use, redistribution only under the same license with the license text included. A public research demo is inside that, but the Space README must say so and ship the license text.
- **The 1 s checkpoint is not yet self-contained.** `train_code/outputs/ARTalkGen_ARTalkData/Sep11_1625_jdrkn/checkpoints/iter_200000.pt` (270 MB) holds generator + codec weights and a `meta_cfg` whose `DATASET.DATA_PATH`, `DATASET.META_PATH`, `MODEL.VAE_CONFIG.VAE_PATH` and `MODEL.VAE_CONFIG.STATS_PATH` are absolute paths on this host. `ARTalkCodec.__init__` reads `STATS_PATH` unconditionally, so the 4.4 kB `metadata_stats.json` (motion mean/std) must ship with the checkpoint and the loader must repoint it.
- **The ARTalk pins lag what actually runs.** The pixi env has `artalk` installed editable from the `refactor/library-api-for-standalone-renderer` worktree at 8460342, which carries runtime fixes the 1 s path depends on (turn-start context reset d6d17c1, `stop()` join b17cf94, linalg init 8460342). None of those are in the lockfile's `artalk` rev b88ebe3, in the bootstrap `ARTALK_TRAIN_REV` 5126a30, or on the pushed branch head ff6dc44. The Space builds from the lockfile, so this has to be reconciled first.
- **The Docker path is unvalidated.** `Dockerfile` + `compose.yaml` landed with the pixi lock (a12b0d7) but have never been built on a docker-capable host.
- **Interactive mode costs money per second.** Both OpenAI backends read the key from `st.secrets`; there is no per-visitor key entry. Loopback mode (visitor's own microphone drives the avatar) needs no key and is a complete demo of the motion models.
- **Concurrency is per session.** Model weights are `st.cache_resource` (shared), the pipeline is per browser session. Two visitors means two pipelines on one GPU, each with its own render loop, so realtime headroom halves; nothing limits that today.
- **Motion model is a process flag.** `--motion-model` selects one streamer per process, so "original ARTalk next to ARTalk 1 s" in one Space needs a sidebar selector.

## Target configuration

| Item | Choice | Why |
| --- | --- | --- |
| SDK | Docker, `app_port: 7860` | only way to ship the pixi stack |
| Hardware | 1x L4 (24 GB, $0.80/h) first; A10G small (24 GB, $1.00/h) as the fallback if L4's throughput disappoints; T4 small ($0.40/h) as a measured stretch goal | the installed rasterizer and pytorch3d carry sm_80 cubins, which run natively on sm_86 (A10G) and sm_89 (L4) without PTX JIT; VRAM need is ~8.5 GB plus ~3.2 GB for CUDA graphs, so either 24 GB card fits both motion models resident |
| Sleep time | 15 to 30 min idle, plus pause when not demoing | billing is per minute while Running, regardless of use |
| Visibility | private, revisited after FLAME permission | see decisions below |
| Motion models | original ARTalk at launch; sidebar selector built now, ARTalk 1 s added once the author clears it | one Space, one URL, A/B in place later |
| Appearance | mesh + built-in GAGAvatar avatars; **Register avatar** hidden | tracking needs the second, conflicting env |
| Conversation | Loopback by default; Interactive with a visitor-supplied key | no owner key on a public URL |
| TURN | Cloudflare via the HF partnership, `HF_TOKEN` secret | community endpoint is deprecated; Twilio is the paid fallback |
| FLAME | fetched at container start from a **private** Hub repo with the `HF_TOKEN` secret, never in the image or the Space repo | keeps the file off every public surface while the compliance question is settled |
| Public weights | baked into the image at build via the manifest downloaders | cold start after sleep is then container start + weight load, not a 4.5 GB download |
| Persistent storage | none in phase 1 | nothing to persist without avatar registration |

Estimated image: ~8 GB pixi env + ~3 GB weights at launch (ARTalk 2.0, GAGAvatar 0.75, tracked 0.17, styles 0.02), rising by ~1.5 GB when the 1 s model and its wav2vec encoder are added. Well inside the 50 GB ephemeral disk; L4 has 400 GB.

## Decisions (settled 2026-09-14)

1. **Visibility: the Space stays private.** FLAME's license forbids sharing and a public demo server is not addressed; the private-repo fetch contains the file technically, and going public waits on written permission from Max Planck. Private Spaces still bill for hardware while running.
2. **The 1 s model waits for the author.** It was trained on the delivered ARTalk data build (47 k clips); publishing a derived model needs the data owner's agreement, which the user is asking the ARTalk author for. Until then the Space ships only the already published weights: the original ARTalk checkpoint and GAGAvatar from `xg-chu/ARTalk` and `xg-chu/GAGAvatar`. The motion-model selector is still built so adding the 1 s model later is a config change, not a code change.
3. **Namespace `whitphx`.** Space `whitphx/artalk-realtime` (the app's identity is the realtime ARTalk + GAGAvatar demo; "streamlit" is an implementation detail that would date the name). Private asset repo `whitphx/artalk-gated-assets` for `FLAME_with_eye.pt` (a neutral name that can hold further license-gated files). 1 s model, when cleared: `whitphx/ARTalk-1s`, matching the upstream `xg-chu/ARTalk` casing so the two read as a family.
4. **Interactive mode.** Loopback by default plus a session-scoped key field.
5. **Hardware: 1x L4**, with the tier left switchable. The launch presets already select render flags by detected VRAM and compute capability, so moving to A10G, T4 or A100 is a change in the Space's hardware settings, not in the repo; the bring-up checklist in phase 3 is what to rerun after a switch.

## Phases

### Phase 0: prerequisites (no Space yet)

1. ~~Build the image on a docker host first.~~ No docker host is reachable from the lab, so the private Space's own build is the first build (settled 2026-09-14).
2. ~~Reconcile the ARTalk pins.~~ Done 2026-09-14: the refactor branch was pushed (fast-forward to 8460342), the lockfile and `ARTALK_TRAIN_REV` point at it, and the 1 s model loads and streams from the locked env through `vendor/ARTalk/train_code`.

### Phase 1: assets on the Hub (now)

1. Original ARTalk and GAGAvatar: nothing to upload. The README front matter links `xg-chu/ARTalk` and `xg-chu/GAGAvatar` under `models:` (done).
2. FLAME: `artalk-demo assets --gated-repo` (done; env `ARTALK_GATED_ASSETS_REPO`) downloads `FLAME_with_eye.pt` from a private repo with the Hub token and verifies it against the manifest hash. Still to do by the user: create the private repo `whitphx/artalk-gated-assets` and upload the file (the lab token is read-only).

### Phase 1b: the 1 s model (after the author's confirmation)

1. Repack the 1 s checkpoint: keep `model` and `meta_cfg`, drop the `DATASET.*` paths, set `VAE_PATH` to null (codec weights are inside; the loader already passes `init_submodule=False`), set `STATS_PATH` to the relative `metadata_stats.json`. Record the SHA-256.
2. Create `whitphx/ARTalk-1s` (private) with `ARTalk1s_wav2vec.pt`, `metadata_stats.json`, and a model card: architecture (train_code generator, Fallingwater-family decoder), recipe (`CLIP_LENGTH` 25, `V_PATCH_NUMS` [1, 5, 25], `PREV_LENGTH` 75, 100-frame style window, silence-robustness augmentation), audio encoder `facebook/wav2vec2-xls-r-300m` loaded separately, eval numbers against the release (generation LVE/MHD/FDD/velocity ratio from `docs/chunk-size-retraining.md`), known limits (no head rotation by construction of the data), license and provenance, and the `train_code` revision needed to load it.
3. Loader: teach `load_artalk1s_model` to resolve `STATS_PATH` relative to the checkpoint's directory when it is not absolute, so the repacked file loads from any tree. This and the repack can be prepared and tested locally before the confirmation; only the upload waits.
4. Asset plumbing: add the 1 s files to `artalk-demo assets` (manifest entry with repo, revision, size, hash, like the existing ones) so the image build and local hosts share one path; extend `doctor`'s asset check. Add the repo to the Space README `models:` list and the wav2vec prefetch to the Dockerfile.

### Phase 2: app changes (each its own commit)

Status 2026-09-14: 1, 3, 4, 5 and 7 landed and were checked in a headless browser against the running app; 2 is deferred to phase 1b because a selector with one entry is dead code until the 1 s model is published; 6 is deferred until the Space exists and the cold-start time is measured.

1. **ICE configuration.** A `rtc_configuration` for `webrtc_streamer` chosen by `ARTALK_ICE_PROVIDER` (`none` | `hf` | `twilio`). The Cloudflare credential fetch belongs in `streamlit-webrtc` (`get_cloudflare_ice_servers` next to the existing helpers, using the FastRTC credential endpoint with `HF_TOKEN`); until that release, the app can call FastRTC's `get_cloudflare_turn_credentials` directly. Credentials are short-lived, so fetch per session, not at import.
2. **Motion model selector.** Sidebar radio over the motion models that are configured (only the original ARTalk at launch, so the radio has one entry and can stay hidden until a second appears); both model loaders stay `cache_resource` so switching only rebuilds the pipeline (the config tuple already keys on the motion model).
3. **Visitor-supplied API key.** Password `st.text_input` in the Interactive section, session state only, used when no secret is configured; the "Configure the secret" stop becomes "enter a key". Never logged, never in diagnostics snapshots.
4. **Session gate.** A process-wide slot count (`ARTALK_MAX_SESSIONS`, default 1 on Spaces) acquired when a pipeline starts; further visitors see a "demo in use" notice with a retry instead of silently halving the other session's throughput. Release on stop, on WebRTC disconnect, and after an idle timeout so an abandoned tab cannot hold the GPU.
5. **Spaces profile.** `[profiles.spaces]` in `launch.toml`: bind 0.0.0.0, port 7860, `remote = "none"`, `HF_HOME` under the uid-1000 home, `ARTALK_ICE_PROVIDER=hf`. The existing GPU presets already cover L4/A10G (`turing-or-newer`, graphs on) and T4 (16 GB, graphs still fit).
6. **Startup preload.** Load both motion models and GAGAvatar on process start (a background thread behind the existing construction lock) so the first visitor after a wake does not pay the load; the Space reports Running as soon as the port answers.
7. **Hide Register avatar** when no tracker environment is configured.

### Phase 3: the Space

1. Create `whitphx/artalk-realtime` (private, Docker SDK) and deploy with `scripts/push_space.sh`. The Hub's pre-receive hook rejects any pushed history that ever carried a binary outside LFS/Xet, and the rasterizer wheel entered this repository as a plain blob, so the branch itself cannot be pushed; the script pushes snapshot commits of the current tree instead (the wheel is LFS-tracked since 59ec3e0). Plain `git fetch` from the Space also fails with git 2.55 (`expected 'acknowledgments'`), which the snapshot approach sidesteps. The `Dockerfile` already runs as uid 1000, bakes the public weights when the `BAKE_ASSETS=1` variable is set, and starts through `scripts/spaces_start.sh` (gated FLAME fetch, then the `spaces` profile on port 7860).
2. Settings: variables `BAKE_ASSETS=1` and `ARTALK_GATED_ASSETS_REPO=whitphx/artalk-gated-assets`; secret `HF_TOKEN` (read scope on that repo, also used for TURN). No OpenAI key.
3. README: the front matter is in place (`sdk: docker`, `app_port: 7860`, `models:`). Still to add: a license note covering ARTalk (MIT), GAGAvatar (MIT), the Gaussian-Splatting research license (with its text in the repo), FLAME terms, and that Interactive mode uses the visitor's own OpenAI key.
4. Bring-up on L4: check the PTX JIT warm-up time for the rasterizer and the ARTalk extension, that `doctor`'s architecture audit accepts PTX-only coverage, mic permission through the Spaces iframe and on the direct `*.hf.space` URL, and TURN relay for both directions. Measure the realtime ratio and turn-first-frame latency at 512 with graphs on; if L4 is marginal, switch the hardware setting to A10G small and rerun this step. Set the sleep time.
5. Private Space visitors need a Hub account with access, so sharing with collaborators means adding them to the repo. Going public waits on the FLAME permission; a community GPU grant application makes sense at that point.

### Later

- Avatar registration (needs the tracker env in the image, about a minute of CPU per image, and persistent storage for `user_avatars/`).
- PersonaPlex as a local backend (needs a second GPU-class worth of VRAM at 8-bit; see `docs/portability-plan.md`).
- Replicas if demand ever exceeds one session at a time; each replica bills separately.

## Cost sketch

| Hardware | Per hour | Always on, per month | Notes |
| --- | --- | --- | --- |
| T4 small | $0.40 | ~$290 | 16 GB; throughput unmeasured, likely marginal at 512 |
| 1x L4 | $0.80 | ~$580 | recommended starting point |
| A10G small | $1.00 | ~$730 | same VRAM as L4, Ampere |
| A100 large | $2.50 | ~$1,800 | matches the measured lab setup |

With a 15-minute sleep time the bill is demo hours only. TURN relay traffic at roughly 1 to 2 Mbps per session puts the free 10 GB per month at about 12 to 20 session-hours; beyond that Cloudflare bills, or Twilio via the existing helper.

## Sources

- Docker Spaces: https://huggingface.co/docs/hub/spaces-sdks-docker
- Spaces GPU hardware and billing: https://huggingface.co/docs/hub/spaces-gpus
- Spaces overview (paid-plan requirement, networking, secrets): https://huggingface.co/docs/hub/spaces-overview
- FastRTC deployment and TURN (Cloudflare partnership, deprecated community endpoint): https://github.com/gradio-app/fastrtc/blob/main/docs/deployment.md
- FLAME model license: https://flame.is.tue.mpg.de/modellicense.html
