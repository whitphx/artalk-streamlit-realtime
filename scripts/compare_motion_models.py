#!/usr/bin/env python
"""Render the same audio through each motion model for side-by-side judging.

Both models drive the identical FLAME mesh renderer and the identical
smoother, so the only difference in the output videos is the motion each
model produced. Realtime constraints do not apply here: this exists to
answer whether a model's lip-sync is better, not whether it keeps pace.
"""

from __future__ import annotations

import argparse
import fractions
import os
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artalk.assets import ARTalkAssets
from artalk.flame_model import RenderMesh
from artalk.rendering import StreamingRenderer
from artalk.runtime import ARTalkRuntime, ARTalkRuntimeConfig
from artalk.streaming import ARTalkStreamer, CausalSavgolSmoother

from artalk_streamlit_realtime.config import ARTALK_FPS, ARTALK_SAMPLE_RATE


def write_video_with_audio(
    path: Path,
    frames: torch.Tensor,
    fps: int,
    audio: torch.Tensor,
    sample_rate: int,
) -> None:
    """Mux (T, H, W, 3) uint8 frames with mono float audio into an mp4."""
    pcm = (audio.clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
    with av.open(str(path), mode="w") as container:
        video_stream = container.add_stream("libx264", rate=fps)
        video_stream.width = frames.shape[2]
        video_stream.height = frames.shape[1]
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=sample_rate, layout="mono")

        for frame in frames:
            av_frame = av.VideoFrame.from_ndarray(frame.numpy(), format="rgb24")
            container.mux(video_stream.encode(av_frame))
        container.mux(video_stream.encode())

        # The AAC encoder takes a fixed number of samples per call.
        frame_size = audio_stream.codec_context.frame_size or 1024
        for start in range(0, pcm.size, frame_size):
            block = pcm[start : start + frame_size]
            if block.size < frame_size:
                block = np.pad(block, (0, frame_size - block.size))
            av_audio = av.AudioFrame.from_ndarray(
                block[None], format="s16", layout="mono"
            )
            av_audio.sample_rate = sample_rate
            av_audio.pts = start
            av_audio.time_base = fractions.Fraction(1, sample_rate)
            container.mux(audio_stream.encode(av_audio))
        container.mux(audio_stream.encode())


def build_streamer(name: str, runtime, args, device):
    if name == "artalk":
        return ARTalkStreamer(runtime.model)
    if name == "artalk1s":
        from artalk_streamlit_realtime.artalk1s import (
            ARTalk1sStreamer,
            load_artalk1s_model,
        )

        model = load_artalk1s_model(
            args.artalk1s_train_code_dir, args.artalk1s_checkpoint, device
        )
        return ARTalk1sStreamer(model)
    from artalk_streamlit_realtime.fallingwater import (
        FallingwaterStreamer,
        load_fallingwater_model,
    )

    model = load_fallingwater_model(
        args.fallingwater_dir, args.fallingwater_checkpoint, device
    )
    return FallingwaterStreamer(model)


def run_model(name: str, audio: torch.Tensor, runtime, renderer, args, device) -> dict:
    streamer = build_streamer(name, runtime, args, device)
    smoother = CausalSavgolSmoother()

    decode_t0 = time.perf_counter()
    chunks = [streamer.feed(audio.to(device))]
    chunks.append(streamer.finish())
    motion = torch.cat([c for c in chunks if c.shape[0]], dim=0)
    smoothed = torch.cat(
        [smoother.feed(motion.float()), smoother.finish()], dim=0
    )
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - decode_t0

    render_t0 = time.perf_counter()
    frames = []
    for start in range(0, smoothed.shape[0], args.render_batch_size):
        batch = smoothed[start : start + args.render_batch_size]
        rgb, _ = renderer.render_batch_profile(batch)
        frames.append((rgb.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu())
    video = torch.cat(frames, dim=0)
    torch.cuda.synchronize()
    render_s = time.perf_counter() - render_t0
    return {"video": video, "decode_s": decode_s, "render_s": render_s}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, type=str)
    parser.add_argument("--output-dir", default="comparisons", type=str)
    parser.add_argument("--asset-dir", default="assets", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--render-res", default=512, type=int)
    parser.add_argument("--render-batch-size", default=8, type=int)
    parser.add_argument("--fallingwater-dir", type=str)
    parser.add_argument("--fallingwater-checkpoint", type=str)
    parser.add_argument(
        "--artalk1s-train-code-dir",
        default=os.environ.get("ARTALK1S_TRAIN_CODE_DIR"),
        type=str,
    )
    parser.add_argument(
        "--artalk1s-checkpoint",
        default=os.environ.get("ARTALK1S_CHECKPOINT"),
        type=str,
    )
    parser.add_argument(
        "--models",
        default="artalk,fallingwater",
        help="Comma-separated model names to render.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    waveform, sample_rate = torchaudio.load(args.audio)
    audio = torchaudio.functional.resample(
        waveform.mean(0), orig_freq=sample_rate, new_freq=ARTALK_SAMPLE_RATE
    )
    print(f"audio: {audio.shape[0] / ARTALK_SAMPLE_RATE:.2f}s")

    runtime = ARTalkRuntime(
        ARTalkRuntimeConfig(
            assets=ARTalkAssets.resolve(root=args.asset_dir),
            device=args.device,
            flame_scale=1.0,
        )
    )
    renderer = StreamingRenderer(
        mode="mesh",
        basic_vae=runtime.model.basic_vae,
        flame_model=runtime.flame_model,
        mesh_renderer=RenderMesh(
            image_size=args.render_res,
            faces=runtime.flame_model.get_faces(),
            scale=1.0,
        ),
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.audio).stem

    for name in args.models.split(","):
        name = name.strip()
        if not name:
            continue
        print(f"\n=== {name} ===", flush=True)
        result = run_model(name, audio, runtime, renderer, args, device)
        path = output_dir / f"{stem}-{name}.mp4"
        write_video_with_audio(
            path, result["video"], ARTALK_FPS, audio, ARTALK_SAMPLE_RATE
        )
        frames = result["video"].shape[0]
        print(
            f"frames={frames} decode={result['decode_s']:.1f}s "
            f"render={result['render_s']:.1f}s -> {path}"
        )


if __name__ == "__main__":
    main()
