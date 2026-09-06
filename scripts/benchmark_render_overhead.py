#!/usr/bin/env python
"""Decompose GAGAvatar render cost into per-frame and per-batch parts.

The renderer processes motion in batches whose size is set by the chunk
schedule: a 100-frame chunk renders as 12x8+4, a 25-frame chunk as
3x8+1 per chunk. If per-batch overhead (kernel dispatch, python setup)
is significant, short chunks pay it four times as often. This script
measures ms/batch across a batch-size sweep, fits the fixed-vs-marginal
split, counts CUDA kernel launches, and prices the concrete schedules.

Run from the repository root in the app environment:

    python scripts/benchmark_render_overhead.py --device cuda
    python scripts/benchmark_render_overhead.py --variant fp16-graph-uint8
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artalk.assets import ARTalkAssets
from artalk.rendering import StreamingRenderer

VARIANTS = {
    "eager": dict(autocast_dtype=None, compile_mode=None, uint8=False),
    "fp16": dict(autocast_dtype="float16", compile_mode=None, uint8=False),
    "fp16-graph": dict(autocast_dtype="float16", compile_mode="cuda-graph", uint8=False),
    "fp16-graph-uint8": dict(autocast_dtype="float16", compile_mode="cuda-graph", uint8=True),
    "graph": dict(autocast_dtype=None, compile_mode="cuda-graph", uint8=False),
}

# Steady-state render schedules per 100 frames of speech, as
# (batch_size, count) runs: how the pipeline slices chunks with
# render_batch_size 8, and candidate alternatives.
SCHEDULES = {
    "4s chunk, batch 8 (12x8+4)": [(8, 12), (4, 1)],
    "1s chunk, batch 8 (4x(3x8+1))": [(8, 12), (1, 4)],
    "1s chunk, batch 25 (4x25)": [(25, 4)],
    "1s chunk, batch 5 (4x5x5)": [(5, 20)],
}


def build_renderer(device: str, asset_dir: str | None, variant: dict, avatar: str):
    from gagavatar.runtime import GAGAvatarRuntime, GAGAvatarRuntimeConfig

    from artalk_streamlit_realtime.assets import gagavatar_assets_in_artalk_tree
    from artalk_streamlit_realtime.runtime import StreamingGAGAvatarAdapter

    artalk_assets = ARTalkAssets.resolve(root=asset_dir)
    assets = gagavatar_assets_in_artalk_tree(artalk_assets)
    runtime = GAGAvatarRuntime(
        GAGAvatarRuntimeConfig(
            assets=assets,
            device=device,
            autocast_dtype=variant["autocast_dtype"],
            compile_mode=variant["compile_mode"],
        )
    )
    return StreamingRenderer(
        mode="gagavatar",
        gagavatar=StreamingGAGAvatarAdapter(runtime),
        gagavatar_flame=runtime.flame_model,
        shape_id=avatar,
        device=torch.device(device),
        stage_sync=False,
        output_uint8=variant["uint8"],
    ), runtime


def time_batches(renderer, motion, batch_size: int, n_batches: int) -> float:
    """Median wall ms per batch, end-to-end (build batch -> RGB on CPU)."""
    times = []
    for i in range(n_batches):
        batch = motion[:batch_size]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        renderer.render_batch_profile(batch)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000.0


def time_stages(runtime, renderer, motion, batch_size: int, iters: int) -> dict:
    """Median ms per stage of the gagavatar path, synced between stages."""
    import numpy as np

    rows = []
    for _ in range(iters):
        batch_motion = motion[:batch_size]
        torch.cuda.synchronize()
        t = [time.perf_counter()]
        batch = runtime.build_forward_batch(batch_motion)
        torch.cuda.synchronize()
        t.append(time.perf_counter())
        gs_params = runtime.model.forward_gaussians(batch)
        torch.cuda.synchronize()
        t.append(time.perf_counter())
        from gagavatar.libs.utils_renderer import render_gaussian

        gen_images = render_gaussian(
            gs_params=gs_params,
            cam_matrix=batch["t_transform"],
            cam_params=runtime.model.cam_params,
        )["images"]
        torch.cuda.synchronize()
        t.append(time.perf_counter())
        sr = runtime.model.upsampler(gen_images)
        torch.cuda.synchronize()
        t.append(time.perf_counter())
        renderer._batch_to_output(sr.clamp(0, 1))
        torch.cuda.synchronize()
        t.append(time.perf_counter())
        rows.append(np.diff(t) * 1000.0)
    med = np.median(np.stack(rows), axis=0)
    names = ["build_batch", "gaussians", "rasterize", "upsampler", "to_output"]
    return {name: round(float(ms), 2) for name, ms in zip(names, med)}


def count_launches(renderer, motion, batch_size: int) -> dict:
    from torch.profiler import ProfilerActivity, profile

    batch = motion[:batch_size]
    renderer.render_batch_profile(batch)  # shape warmup
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        renderer.render_batch_profile(batch)
    kernels = 0
    launch_ms = 0.0
    for evt in prof.key_averages():
        if evt.device_type is not None and str(evt.device_type) == "DeviceType.CUDA":
            kernels += evt.count
        if evt.key in ("cudaLaunchKernel", "cuLaunchKernel"):
            launch_ms += evt.self_cpu_time_total / 1000.0
    return {"cuda_kernels": kernels, "cudaLaunchKernel_self_ms": round(launch_ms, 2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--asset-dir", default=os.environ.get("ARTALK_ASSET_DIR"), type=str)
    parser.add_argument("--avatar", default="1.jpg", type=str)
    parser.add_argument("--variant", default="eager", choices=sorted(VARIANTS), type=str)
    parser.add_argument("--batches", default="1,2,4,5,8,12,16,25", type=str)
    parser.add_argument("--iters", default=30, type=int)
    parser.add_argument("--output-dir", default="benchmarks", type=str)
    parser.add_argument(
        "--stages",
        action="store_true",
        help="Per-stage decomposition (build/gaussians/rasterize/upsampler/out).",
    )
    parser.add_argument("--no-schedules", action="store_true")
    parser.add_argument("--no-launches", action="store_true")
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    renderer, runtime = build_renderer(args.device, args.asset_dir, variant, args.avatar)
    batch_sizes = [int(b) for b in args.batches.split(",")]

    torch.manual_seed(0)
    motion = torch.zeros(32, 106, device=args.device)
    motion[:, :100] = torch.randn(32, 100, device=args.device) * 0.3

    # Warm up every shape that will be timed (cuDNN autotune, allocator,
    # cuda-graph capture happen on first use per shape).
    warm_shapes = set(batch_sizes)
    if not args.no_schedules:
        warm_shapes |= {s for runs in SCHEDULES.values() for s, _ in runs}
    for b in sorted(warm_shapes):
        renderer.render_batch_profile(motion[:b])
    torch.cuda.synchronize()

    sweep = {}
    for b in batch_sizes:
        ms = time_batches(renderer, motion, b, args.iters)
        sweep[b] = {"ms_per_batch": round(ms, 2), "ms_per_frame": round(ms / b, 2)}
        print(f"batch {b:>2}: {ms:7.2f} ms/batch  {ms / b:6.2f} ms/frame", flush=True)

    # Least-squares fixed + marginal split over the sweep.
    xs = torch.tensor(batch_sizes, dtype=torch.float64)
    ys = torch.tensor([sweep[b]["ms_per_batch"] for b in batch_sizes], dtype=torch.float64)
    a = torch.stack([torch.ones_like(xs), xs], dim=1)
    (fixed, marginal) = torch.linalg.lstsq(a, ys[:, None]).solution[:, 0].tolist()
    print(f"\nper-batch fixed cost ~{fixed:.1f} ms, marginal ~{marginal:.2f} ms/frame")

    schedules = {}
    if not args.no_schedules:
        print("\nschedule cost per 100 frames (4 s of speech):")
        for name, runs in SCHEDULES.items():
            total = 0.0
            for b, count in runs:
                total += time_batches(renderer, motion, b, max(args.iters // 2, 5)) * count
            schedules[name] = round(total, 1)
            print(f"  {name:<34} {total:7.1f} ms  ({total / 4000:.3f}x realtime)", flush=True)

    launches = {}
    if not args.no_launches:
        launches = {
            b: count_launches(renderer, motion, b)
            for b in (1, 8, 25)
            if b in warm_shapes
        }
        for b, stats in launches.items():
            print(
                f"batch {b:>2}: {stats['cuda_kernels']} cuda kernels "
                f"({stats['cuda_kernels'] / b:.0f}/frame), "
                f"cudaLaunchKernel self {stats['cudaLaunchKernel_self_ms']} ms"
            )

    stages = {}
    if args.stages:
        print("\nper-stage medians (ms/batch, synced between stages):")
        for b in (1, 8, 25):
            if b not in warm_shapes:
                continue
            stages[b] = time_stages(runtime, renderer, motion, b, args.iters)
            row = "  ".join(f"{k} {v:7.2f}" for k, v in stages[b].items())
            print(f"  batch {b:>2}: {row}", flush=True)

    properties = torch.cuda.get_device_properties(args.device)
    out = {
        "meta": {
            "host": platform.node(),
            "gpu": properties.name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "variant": args.variant,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "sweep": sweep,
        "fit": {"fixed_ms_per_batch": round(fixed, 2), "marginal_ms_per_frame": round(marginal, 3)},
        "schedules_ms_per_100_frames": schedules,
        "launches": launches,
        "stages_ms": stages,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / datetime.now(timezone.utc).strftime(
        f"render-overhead-{platform.node()}-{args.variant}-%Y%m%d-%H%M%S.json"
    )
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\nwritten: {out_path}")
    sys.stdout.flush()
    # Skip interpreter teardown: CUDA Graph artifacts abort in atexit cleanup.
    os._exit(0)


if __name__ == "__main__":
    main()
