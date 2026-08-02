#!/usr/bin/env python
"""Benchmark + parity harness for the realtime pipeline.

Runs a matrix of pipeline configurations headless on the current host and
reports, per configuration: throughput (realtime ratio), peak VRAM, and
frame parity against the first configuration sharing its renderer mode and
resolution. Results are printed as a table and written as JSON under
``benchmarks/`` for cross-host aggregation (the checkout is typically on a
shared filesystem, so runs from different GPUs land side by side).

Callbacks are drained at full speed rather than realtime pace: the media
clock races ahead of rendering, so the render loop's backpressure never
engages and the measured cost is pure compute. Realtime-paced behavior
(turn latency, underruns) is measured in live sessions instead.

Usage (run from the repository root, in the app environment):

    python scripts/benchmark_pipeline.py --device cuda
    python scripts/benchmark_pipeline.py --configs gagavatar:512:8,mesh:256:8
"""

from __future__ import annotations

import argparse
import dataclasses
import fractions
import json
import logging
import math
import os
import platform
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.getLogger("artalk.realtime_pipeline").setLevel(logging.ERROR)

from artalk.assets import ARTalkAssets
from artalk.flame_model import RenderMesh
from artalk.realtime_pipeline import ARTalkPipeline
from artalk.runtime import ARTalkRuntime, ARTalkRuntimeConfig

SAMPLE_RATE = 16000
DEFAULT_CONFIGS = "gagavatar:512:4,gagavatar:512:8,gagavatar:512:16,mesh:512:8,mesh:256:8"
PARITY_FRAMES = 16


@dataclass(frozen=True)
class BenchConfig:
    renderer_mode: str
    render_res: int
    render_batch_size: int
    # fp16 runs the GAGAvatar conv stages under autocast (rasterizer stays
    # fp32); parity is judged against the fp32 configuration of the same
    # mode/resolution, so list the fp32 baseline first. compile applies
    # torch.compile(reduce-overhead) to the upsampler (sm_70+ only).
    fp16: bool = False
    compile: bool = False

    @property
    def label(self) -> str:
        label = f"{self.renderer_mode}:{self.render_res}:b{self.render_batch_size}"
        if self.fp16:
            label += ":fp16"
        if self.compile:
            label += ":compile"
        return label

    @property
    def parity_group(self) -> str:
        return f"{self.renderer_mode}:{self.render_res}"


def parse_configs(spec: str) -> list[BenchConfig]:
    configs = []
    for part in spec.split(","):
        mode, res, batch, *extras = part.strip().split(":")
        unknown = set(extras) - {"fp16", "compile"}
        if unknown:
            raise ValueError(f"Unknown config options: {sorted(unknown)}")
        configs.append(
            BenchConfig(
                mode,
                int(res),
                int(batch),
                fp16="fp16" in extras,
                compile="compile" in extras,
            )
        )
    return configs


def load_audio(path: str | None, seconds: float) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    if path is None:
        rng = np.random.default_rng(0)
        return (rng.standard_normal(n, dtype=np.float32) * 3000).astype(np.int16)
    import torchaudio

    wave, sr = torchaudio.load(path)
    wave = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wave).mean(dim=0)
    samples = (wave.numpy() * 32767).astype(np.int16)
    if samples.size < n:
        samples = np.tile(samples, math.ceil(n / samples.size))
    return samples[:n]


def build_gagavatar(device: str, artalk_assets: ARTalkAssets):
    from gagavatar.runtime import GAGAvatarRuntime, GAGAvatarRuntimeConfig

    from artalk_streamlit_realtime.assets import gagavatar_assets_in_artalk_tree

    assets = gagavatar_assets_in_artalk_tree(artalk_assets)
    runtime = GAGAvatarRuntime(
        GAGAvatarRuntimeConfig(assets=assets, device=device)
    )

    class Adapter:
        def __init__(self, rt):
            self.runtime = rt

        def set_avatar_id(self, avatar_id):
            self.runtime.set_avatar_id(avatar_id)

        def build_forward_batch(self, motion_code, _flame_model=None):
            return self.runtime.build_forward_batch(motion_code)

        def forward_expression(self, batch):
            return self.runtime.render_rgb_batch(batch)

    shape_id = runtime.available_avatar_ids()[0]
    return Adapter(runtime), runtime.flame_model, shape_id


def duration_total_ms(durations: dict, key: str) -> float:
    return float(durations.get(key, {}).get("total_ms", 0.0))


def run_config(
    config: BenchConfig,
    artalk_runtime,
    mesh_renderer,
    gagavatar,
    audio: np.ndarray,
    device: str,
    render_uint8_gpu: bool = False,
) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    kwargs = dict(
        model=artalk_runtime.model,
        flame_model=artalk_runtime.flame_model,
        mesh_renderer=mesh_renderer,
        device=device,
        render_res=config.render_res,
        render_batch_size=config.render_batch_size,
        renderer_mode=config.renderer_mode,
        renderer_output_uint8=render_uint8_gpu,
    )
    if config.renderer_mode == "gagavatar":
        adapter, gaga_flame, shape_id = gagavatar
        # The heavy runtime is shared across configs; precision is a config
        # attribute on it, so swap the (frozen) config object per run. The
        # compiled upsampler is cached on the adapter so eager/compiled
        # configs can alternate without recompiling.
        adapter.runtime.config = dataclasses.replace(
            adapter.runtime.config,
            autocast_dtype="float16" if config.fp16 else None,
        )
        model = adapter.runtime.model
        if not hasattr(adapter, "eager_upsampler"):
            adapter.eager_upsampler = model.upsampler
            adapter.compiled_upsampler = None
        if config.compile:
            if adapter.compiled_upsampler is None:
                adapter.compiled_upsampler = torch.compile(
                    adapter.eager_upsampler, mode="reduce-overhead"
                )
            model.upsampler = adapter.compiled_upsampler
        else:
            model.upsampler = adapter.eager_upsampler
        kwargs.update(gagavatar=adapter, gagavatar_flame=gaga_flame, shape_id=shape_id)
    pipeline = ARTalkPipeline(**kwargs)
    # Measurement runs with no live callbacks at all, so their GIL and
    # metrics-lock traffic cannot inflate the stage timings (even throttled
    # drains measurably slowed the worker). Without callbacks the media
    # clock never advances and backpressure would block the worker, so lift
    # the high-water mark for the duration of the benchmark.
    pipeline._video_queue_high_water = 10**9
    pipeline._video_queue_max = 10**9

    n_chunks_expected = audio.size // (4 * SAMPLE_RATE)
    wall_t0 = time.perf_counter()
    for start in range(0, audio.size, SAMPLE_RATE):
        pipeline.push_audio_samples(audio[start : start + SAMPLE_RATE])
    deadline = time.perf_counter() + 300
    last_progress = (0, time.perf_counter())
    while time.perf_counter() < deadline:
        counters = pipeline.metrics_snapshot()["counters"]
        if (
            counters.get("motion_chunks_produced", 0) >= n_chunks_expected
            and counters.get("audio_in_queue_depth", 1) == 0
            and not counters.get("worker_busy")
        ):
            break
        fed = counters.get("audio_samples_fed_to_streamer", 0)
        if fed > last_progress[0]:
            last_progress = (fed, time.perf_counter())
        elif time.perf_counter() - last_progress[1] > 30:
            print(f"  {config.label}: no progress for 30 s, aborting run", flush=True)
            break
        time.sleep(0.2)
    wall_s = time.perf_counter() - wall_t0

    # Collect parity frames after the fact by advancing the media clock
    # (two 20 ms audio callbacks per 40 ms video frame).
    frames: list[np.ndarray] = []
    audio_pts = 0
    for video_pts in range(PARITY_FRAMES * 4):
        if len(frames) >= PARITY_FRAMES:
            break
        for _ in range(2):
            pipeline.audio_source_callback(audio_pts, fractions.Fraction(1, SAMPLE_RATE))
            audio_pts += 320
        frame = pipeline.video_source_callback(video_pts, fractions.Fraction(1, 25))
        arr = frame.to_ndarray()
        if arr.any():
            frames.append(arr)

    snapshot = pipeline.metrics_snapshot()
    counters = snapshot["counters"]
    durations = snapshot["durations"]
    media_s = counters.get("rendered_frames", 0) / 25.0
    compute_ms = sum(
        duration_total_ms(durations, key)
        for key in (
            "artalk_streamer_feed",
            "smoother_feed",
            "avatar_render_batch",
            "rgb_batch_to_numpy",
        )
    )
    result = {
        "config": config.label,
        "parity_group": config.parity_group,
        "chunks": int(counters.get("motion_chunks_produced", 0)),
        "rendered_frames": int(counters.get("rendered_frames", 0)),
        "compute_ratio": (compute_ms / 1000.0) / media_s if media_s else None,
        "wall_ratio": wall_s / media_s if media_s else None,
        "streamer_feed_ms_per_chunk": duration_total_ms(durations, "artalk_streamer_feed")
        / max(counters.get("motion_chunks_produced", 1), 1),
        "render_ms_per_frame": duration_total_ms(durations, "avatar_render_batch")
        / max(counters.get("rendered_frames", 1), 1),
        "rgb_to_cpu_ms_per_frame": duration_total_ms(durations, "rgb_batch_to_numpy")
        / max(counters.get("rendered_frames", 1), 1),
        "peak_vram_alloc_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_vram_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20,
        "max_gpu_utilization_percent": counters.get("max_gpu_utilization_percent"),
        "frames": frames,
    }
    pipeline.stop()
    time.sleep(0.3)
    return result


def psnr(reference: np.ndarray, other: np.ndarray) -> float:
    ref = reference.astype(np.float64) / 255.0
    oth = other.astype(np.float64) / 255.0
    mse = float(np.mean((ref - oth) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def add_parity(results: list[dict]) -> None:
    references: dict[str, list[np.ndarray]] = {}
    for result in results:
        group = result["parity_group"]
        frames = result.pop("frames")
        if group not in references:
            references[group] = frames
            result["parity_psnr_db"] = None  # reference for its group
            continue
        ref = references[group]
        n = min(len(ref), len(frames))
        if n == 0:
            result["parity_psnr_db"] = None
            continue
        values = [psnr(ref[i], frames[i]) for i in range(n)]
        result["parity_psnr_db"] = round(min(values), 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--asset-dir", default=os.environ.get("ARTALK_ASSET_DIR"), type=str)
    parser.add_argument("--artalk-checkpoint", default=os.environ.get("ARTALK_CHECKPOINT"), type=str)
    parser.add_argument(
        "--artalk-audio-encoder",
        default=os.environ.get("ARTALK_AUDIO_ENCODER", "wav2vec"),
        type=str,
    )
    parser.add_argument("--audio", default=None, type=str, help="Optional wav; defaults to seeded noise.")
    parser.add_argument("--seconds", default=12.0, type=float)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS, type=str)
    parser.add_argument(
        "--render-uint8-gpu",
        action="store_true",
        help="Enable GPU-side uint8 conversion in all benchmarked configs.",
    )
    parser.add_argument("--output-dir", default="benchmarks", type=str)
    args = parser.parse_args()

    configs = parse_configs(args.configs)
    audio = load_audio(args.audio, args.seconds)
    artalk_assets = ARTalkAssets.resolve(root=args.asset_dir)
    artalk_runtime = ARTalkRuntime(
        ARTalkRuntimeConfig(
            assets=artalk_assets,
            audio_encoder=args.artalk_audio_encoder,
            checkpoint_path=args.artalk_checkpoint,
            device=args.device,
            flame_scale=1.0,
        )
    )
    mesh_renderers = {
        res: RenderMesh(
            image_size=res,
            faces=artalk_runtime.flame_model.get_faces(),
            scale=1.0,
        )
        for res in sorted({c.render_res for c in configs})
    }
    gagavatar = None
    if any(c.renderer_mode == "gagavatar" for c in configs):
        gagavatar = build_gagavatar(args.device, artalk_assets)

    capability = torch.cuda.get_device_capability(args.device)
    results = []
    for config in configs:
        if config.compile and capability < (7, 0):
            print(
                f"skipping {config.label}: torch.compile requires sm_70+ "
                f"(this GPU is sm_{capability[0]}{capability[1]})",
                flush=True,
            )
            continue
        print(f"running {config.label} ...", flush=True)
        results.append(
            run_config(
                config,
                artalk_runtime,
                mesh_renderers[config.render_res],
                gagavatar,
                audio,
                args.device,
                render_uint8_gpu=args.render_uint8_gpu,
            )
        )
    add_parity(results)

    properties = torch.cuda.get_device_properties(args.device)
    meta = {
        "host": platform.node(),
        "gpu": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "vram_total_mb": properties.total_memory / 2**20,
        "torch": torch.__version__,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "audio": args.audio or f"seeded-noise-{args.seconds:.0f}s",
        "artalk_checkpoint": str(artalk_runtime.checkpoint_path),
        "render_uint8_gpu": args.render_uint8_gpu,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / datetime.now(timezone.utc).strftime(
        f"bench-{platform.node()}-%Y%m%d-%H%M%S.json"
    )
    out_path.write_text(json.dumps({"meta": meta, "results": results}, indent=1))

    print(f"\n{meta['host']} — {meta['gpu']} (sm_{properties.major}{properties.minor})")
    header = (
        f"{'config':<20} {'compute':>8} {'render':>9} {'streamer':>9} "
        f"{'rgb->cpu':>9} {'VRAM MB':>8} {'PSNR dB':>8}"
    )
    print(header)
    for r in results:
        if r["compute_ratio"] is None:
            print(f"{r['config']:<20} {'(no frames rendered — run failed)':>20}")
            continue
        print(
            f"{r['config']:<20} {r['compute_ratio']:>7.3f}x "
            f"{r['render_ms_per_frame']:>7.1f}ms {r['streamer_feed_ms_per_chunk']:>7.0f}ms "
            f"{r['rgb_to_cpu_ms_per_frame']:>7.2f}ms {r['peak_vram_reserved_mb']:>8.0f} "
            f"{r['parity_psnr_db'] if r['parity_psnr_db'] is not None else '   ref':>8}"
        )
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
