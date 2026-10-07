"""Measure eyelid closure (blinking) in ground-truth and generated motion.

The eyelid aperture of a frame is the mean FLAME distance between upper
and lower eyelid vertices (``EYELID_PAIRS``), with global head pose
zeroed. A blink is a short run of frames whose aperture drops below a
fraction of the clip's open-eye level.

``generate`` reproduces the generation protocol of ARTalk's
``train_code/scripts/eval_gen_generation.py`` (test clips of at least 200
frames, trimmed to whole seconds, whole-clip ``feed`` seeded with 0 per
clip, no style motion unless ``--style`` names one) and saves ground truth plus each model's
raw output. ``trace`` replays clips through the realtime pipeline's
post-model stages in their production order: 20 ms audio items into the
streamer, ``SilentMouthGate`` on each emitted chunk,
``CausalSavgolSmoother``, then optionally ``BlinkInjector`` at each of
``--blink-scales``. ``idle`` does the same from digital silence, with the
gate engaged throughout.
``analyze`` reads the saved motion on CPU and reports blink rate and
closure depth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FPS = 25
SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS

# Upper/lower eyelid vertex pairs: FLAME's own eye-landmark reselection
# (``reselect_eyes`` in ARTalk's FLAME.py, inherited from DECA) puts the
# 68-point eyelid landmarks 37/41 and 38/40 (one eye) and 43/47 and 44/46
# (the other) on these vertices.
EYELID_PAIRS = {
    "eye_a": ((2452, 2360), (2471, 2276)),
    "eye_b": ((1292, 827), (1217, 999)),
}


def load_test_clips(data_dir: Path, split: str, max_clips: int | None):
    from core.libs.utils_lmdb import LMDBEngine

    meta = json.load(open(data_dir / "metadata.json"))[split]
    engine = LMDBEngine(str(data_dir / "data_lmdb"), write=False)
    n = 0
    for key, length in meta:
        if length < 200:
            continue
        rec = engine[key]
        frames = (length // 100) * 100
        gt = torch.from_numpy(rec["motioncode"]).float()[:frames]
        audio = torch.from_numpy(rec["audio"]).float()
        audio = torch.nn.functional.pad(
            audio, (0, max(0, frames * SAMPLES_PER_FRAME - audio.shape[0]))
        )[: frames * SAMPLES_PER_FRAME]
        yield key, gt, audio
        n += 1
        if max_clips and n >= max_clips:
            break
    engine.close()


def build_streamers(args, kinds):
    from artalk_streamlit_realtime.runtime import load_style_motion

    style = load_style_motion(str(args.release_assets), args.style)
    streamers = {}
    if "hp1s" in kinds:
        from artalk_streamlit_realtime.artalk1s import ARTalk1sStreamer, load_artalk1s_model

        model = load_artalk1s_model(args.train_code_dir, args.checkpoint, args.device)
        streamers["hp1s"] = ARTalk1sStreamer(model, style_motion=style)
    if "release" in kinds:
        from artalk.assets import ARTalkAssets
        from artalk.runtime import ARTalkRuntime, ARTalkRuntimeConfig
        from artalk.streaming import ARTalkStreamer

        rt = ARTalkRuntime(ARTalkRuntimeConfig(
            assets=ARTalkAssets.resolve(root=str(args.release_assets)),
            device=args.device, flame_scale=1.0,
        ))
        streamers["release"] = ARTalkStreamer(rt.model, style_motion=style)
    return streamers


def generate_whole(streamer, audio, device):
    torch.manual_seed(0)
    streamer.reset()
    return streamer.feed(audio.to(device)).float().cpu()


def cmd_generate(args):
    sys.path.insert(0, str(args.train_code_dir))
    kinds = args.models.split(",")
    streamers = build_streamers(args, kinds)
    args.out.mkdir(parents=True, exist_ok=True)
    for i, (key, gt, audio) in enumerate(load_test_clips(args.data, args.split, args.max_clips)):
        tracks = {"gt": gt}
        for name, streamer in streamers.items():
            tracks[name] = generate_whole(streamer, audio, args.device)
            if tracks[name].shape[0] != gt.shape[0]:
                raise RuntimeError(f"{name}: {tracks[name].shape[0]} frames for {gt.shape[0]}")
        np.savez(args.out / f"{i:04d}.npz", key=key, **{k: v.numpy() for k, v in tracks.items()})
        print(f"{i} {key} frames={gt.shape[0]}", flush=True)


def post_model_tracks(raw, silent_flags, make_injector, scales, seeds):
    """The realtime pipeline's post-model stages over the chunks a streamer
    emitted: ``SilentMouthGate`` per chunk, ``CausalSavgolSmoother``, then
    ``BlinkInjector`` at each of ``scales`` and injector ``seeds``."""
    from artalk.realtime_pipeline import SilentMouthGate
    from artalk.streaming import CausalSavgolSmoother

    gate, smoother = SilentMouthGate(), CausalSavgolSmoother()
    gated = torch.cat([gate.process(m, s) for m, s in zip(raw, silent_flags)])
    rendered = torch.cat([smoother.feed(gated), smoother.finish()])
    tracks = {"raw": torch.cat(raw), "rendered": rendered}
    for scale in scales:
        for seed in seeds:
            name = f"blink{scale:g}" if len(seeds) == 1 else f"blink{scale:g}_seed{seed}"
            # The injector's output does not depend on chunking, so one call
            # on the whole track equals the pipeline's chunk-by-chunk calls.
            tracks[name] = make_injector(scale, seed).process(rendered)
    return tracks


def injector_maker(args):
    from artalk.flame_model.FLAME import FLAMEModel
    from artalk.realtime_pipeline import BlinkInjector

    flame = FLAMEModel(n_shape=300, n_exp=100, no_lmks=True, model_path=str(args.flame))
    return lambda scale, seed: BlinkInjector.from_flame(flame, scale=scale, seed=seed)


def selected_clips(args):
    clips = [c for i, c in enumerate(load_test_clips(args.data, args.split, args.max_clips))
             if i in args.clip_indices]
    if not args.concat:
        return [(i, *c) for i, c in zip(args.clip_indices, clips)]
    # One continuous stream: the injector and smoother carry state across
    # clip boundaries the way they do across turns in a live session.
    return [(0, "concat", torch.cat([c[1] for c in clips]), torch.cat([c[2] for c in clips]))]


def cmd_trace(args):
    sys.path.insert(0, str(args.train_code_dir))
    streamers = build_streamers(args, args.models.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    item = SAMPLE_RATE // 50
    make_injector = injector_maker(args)
    for i, key, gt, audio in selected_clips(args):
        out = {"gt": gt}
        for name, streamer in streamers.items():
            torch.manual_seed(0)
            streamer.reset()
            raw, flags = [], []
            for s in range(0, audio.shape[0], item):
                samples = audio[s : s + item]
                motion = streamer.feed(samples.to(args.device)).float().cpu()
                if motion.shape[0] == 0:
                    continue
                raw.append(motion)
                # The pipeline's gate rule: RMS of the item that completed the chunk.
                flags.append(float(samples.pow(2).mean().sqrt()) < 0.005)
            tracks = post_model_tracks(raw, flags, make_injector, args.blink_scales,
                                       args.injector_seeds or [i])
            out.update({f"{name}_{k}": v for k, v in tracks.items()})
            out[f"{name}_silent"] = torch.from_numpy(
                np.concatenate([np.full(m.shape[0], f) for m, f in zip(raw, flags)])
            )
            print(f"{i} {key} {name} frames={tracks['raw'].shape[0]} "
                  f"silent-chunks={sum(flags)}/{len(flags)}", flush=True)
        np.savez(args.out / f"trace_{i:04d}.npz", key=key, **{k: v.numpy() for k, v in out.items()})


def cmd_idle(args):
    sys.path.insert(0, str(args.train_code_dir))
    streamers = build_streamers(args, args.models.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    audio = torch.zeros(int(args.seconds) * SAMPLE_RATE)
    for seed in range(args.trials):
        out = {}
        for name, streamer in streamers.items():
            torch.manual_seed(seed)
            streamer.reset()
            motion = streamer.feed(audio.to(args.device)).float().cpu()
            chunks = list(motion.split(streamer.frames_per_chunk))
            tracks = post_model_tracks(chunks, [True] * len(chunks), injector_maker(args),
                                       args.blink_scales, args.injector_seeds or [seed])
            out.update({f"{name}_{k}": v.numpy() for k, v in tracks.items()})
        np.savez(args.out / f"idle_{seed}.npz", **out)
        print(f"idle seed={seed} " + " ".join(f"{k}={v.shape}" for k, v in out.items()), flush=True)


class Aperture:
    def __init__(self, flame_path: Path):
        from artalk.flame_model.FLAME import FLAMEModel

        self.flame = FLAMEModel(n_shape=300, n_exp=100, no_lmks=True, model_path=str(flame_path)).eval()

    @torch.no_grad()
    def __call__(self, motion: np.ndarray) -> np.ndarray:
        """(T, 106) motion -> (T, 2) aperture in mm, one column per eye."""
        m = torch.from_numpy(np.asarray(motion, dtype=np.float32))
        out = []
        for s in range(0, m.shape[0], 500):
            b = m[s : s + 500]
            pose = torch.cat([torch.zeros(b.shape[0], 3), b[:, 103:106]], dim=-1)
            v = self.flame(shape_params=torch.zeros(b.shape[0], 300), expression_params=b[:, :100],
                           pose_params=pose)
            cols = []
            for pairs in EYELID_PAIRS.values():
                d = [(v[:, u] - v[:, lo]).norm(dim=-1) for u, lo in pairs]
                cols.append(torch.stack(d, -1).mean(-1))
            out.append(torch.stack(cols, -1))
        return torch.cat(out).numpy() * 1000.0


def local_open_level(a: np.ndarray, window: int = 2 * FPS + 1) -> np.ndarray:
    """Rolling 90th percentile: the open-eye level around each frame.

    Normalizing by a local rather than a per-clip level keeps slow aperture
    drift (looking down, squinting through a sentence) from counting as
    blinks; a blink is a dip well below the surrounding open level.
    """
    half = window // 2
    padded = np.pad(a, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.percentile(windows, 90, axis=-1)


def blink_events(ratio: np.ndarray, threshold: float, max_len: int = 12, merge_gap: int = 1):
    """Runs below ``threshold`` (gaps up to ``merge_gap`` merged) no longer than ``max_len``.

    ``max_len`` of 12 frames (480 ms) admits slow blinks and rejects eyes
    held shut or half-closed.
    """
    below = ratio < threshold
    runs, start = [], None
    for t, b in enumerate(below):
        if b and start is None:
            start = t
        elif not b and start is not None:
            runs.append([start, t])
            start = None
    if start is not None:
        runs.append([start, len(below)])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b, float(ratio[a:b].min())) for a, b in merged if b - a <= max_len]


THRESHOLDS = (0.6, 0.75)


def clip_stats(ap: np.ndarray, thresholds=THRESHOLDS):
    a = ap.mean(-1)
    open_ = float(np.percentile(a, 90))
    ratio = a / local_open_level(a)
    row = {"frames": len(a), "open_mm": open_, "min_mm": float(a.min()),
           "min_ratio": float(a.min() / open_)}
    for th in thresholds:
        ev = blink_events(ratio, th)
        row[f"n@{th}"] = len(ev)
        row[f"depth@{th}"] = [e[2] for e in ev]
    return row


def summarize(rows, thresholds=THRESHOLDS):
    minutes = sum(r["frames"] for r in rows) / FPS / 60
    s = {"clips": len(rows), "minutes": minutes,
         "open_mm": float(np.mean([r["open_mm"] for r in rows])),
         "median_clip_min_mm": float(np.median([r["min_mm"] for r in rows])),
         "median_clip_min_ratio": float(np.median([r["min_ratio"] for r in rows]))}
    for th in thresholds:
        n = sum(r[f"n@{th}"] for r in rows)
        depths = [d for r in rows for d in r[f"depth@{th}"]]
        s[f"blinks/min@{th}"] = n / minutes
        s[f"clips_with_blink@{th}"] = sum(1 for r in rows if r[f"n@{th}"])
        s[f"mean_event_depth@{th}"] = float(np.mean(depths)) if depths else float("nan")
    return s


def cmd_analyze(args):
    aperture = Aperture(args.flame)
    neutral = aperture(np.zeros((1, 106), dtype=np.float32))[0].mean()
    print(f"neutral-face aperture: {neutral:.3f} mm (vertex pairs {EYELID_PAIRS})")
    files = sorted(args.motion_dir.glob(args.pattern))
    per_track: dict[str, list] = {}
    for f in files:
        z = np.load(f)
        for name in z.files:
            if name == "key" or name.endswith("silent"):
                continue
            row = clip_stats(aperture(z[name]))
            row["file"] = f.name
            per_track.setdefault(name, []).append(row)
    result = {name: summarize(rows) for name, rows in per_track.items()}
    for name, s in result.items():
        print(name, json.dumps(s))
    if args.json_out:
        json.dump({"summary": result, "per_clip": per_track}, open(args.json_out, "w"), indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_gen_args(p):
        p.add_argument("--train-code-dir", required=True, type=Path)
        p.add_argument("--checkpoint", type=Path)
        p.add_argument("--release-assets", type=Path, default=Path("assets"))
        p.add_argument("--data", type=Path)
        p.add_argument("--split", default="test")
        p.add_argument("--max-clips", type=int, default=None)
        p.add_argument("--device", default="cuda")
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--flame", type=Path, default=Path("assets/FLAME_with_eye.pt"))
        p.add_argument("--blink-scales", type=lambda s: [float(x) for x in s.split(",")], default=[])
        p.add_argument("--style", default="default")
        p.add_argument("--injector-seeds", type=lambda s: [int(x) for x in s.split(",")], default=[])

    g = sub.add_parser("generate")
    add_gen_args(g)
    g.add_argument("--models", default="hp1s,release")
    t = sub.add_parser("trace")
    add_gen_args(t)
    t.add_argument("--models", default="hp1s,release")
    t.add_argument("--concat", action="store_true", help="feed the selected clips as one stream")
    t.add_argument("--clip-indices", type=lambda s: [int(x) for x in s.split(",")], required=True)
    i = sub.add_parser("idle")
    add_gen_args(i)
    i.add_argument("--models", default="hp1s,release")
    i.add_argument("--seconds", type=int, default=60)
    i.add_argument("--trials", type=int, default=3)
    a = sub.add_parser("analyze")
    a.add_argument("--motion-dir", required=True, type=Path)
    a.add_argument("--pattern", default="*.npz")
    a.add_argument("--flame", type=Path, default=Path("assets/FLAME_with_eye.pt"))
    a.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    {"generate": cmd_generate, "trace": cmd_trace, "idle": cmd_idle, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
