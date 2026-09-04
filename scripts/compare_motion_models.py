#!/usr/bin/env python
"""Render the same audio through each motion model for side-by-side judging.

All models drive the identical smoother and renderer, so the only
difference in the output videos is the motion each model produced.
Realtime constraints do not apply here: this exists to answer whether a
model's motion is better, not whether it keeps pace.

Renders the FLAME mesh by default; pass ``--renderers mesh,gagavatar``
to also render through the GAGAvatar photoreal head the app uses. With
more than one model, a labeled side-by-side video is written in
addition to the per-model ones.
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

MODEL_LABELS = {
    "artalk": "ARTalk 4s (release)",
    "artalk1s": "ARTalk 1s (ours)",
    "fallingwater": "Fallingwater",
}


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


def build_renderer(kind: str, runtime, artalk_assets, args, device):
    if kind == "mesh":
        return StreamingRenderer(
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
    if kind == "gagavatar":
        from gagavatar.runtime import GAGAvatarRuntime, GAGAvatarRuntimeConfig

        from artalk_streamlit_realtime.assets import gagavatar_assets_in_artalk_tree
        from artalk_streamlit_realtime.runtime import StreamingGAGAvatarAdapter

        assets = gagavatar_assets_in_artalk_tree(artalk_assets)
        gaga_runtime = GAGAvatarRuntime(
            GAGAvatarRuntimeConfig(
                model_path=str(assets.model_path),
                tracked_path=str(assets.tracked_path),
                flame_model_path=str(assets.flame_model_path),
                device=args.device,
            )
        )
        return StreamingRenderer(
            mode="gagavatar",
            gagavatar=StreamingGAGAvatarAdapter(gaga_runtime),
            gagavatar_flame=gaga_runtime.flame_model,
            shape_id=args.gagavatar_avatar,
            device=device,
        )
    raise ValueError(f"Unknown renderer: {kind!r}")


def generate_motion(streamer, audio: torch.Tensor, device) -> tuple[torch.Tensor, float]:
    smoother = CausalSavgolSmoother()
    torch.manual_seed(0)
    t0 = time.perf_counter()
    streamer.reset()
    chunks = [streamer.feed(audio.to(device))]
    chunks.append(streamer.finish())
    motion = torch.cat([c for c in chunks if c.shape[0]], dim=0)
    smoothed = torch.cat(
        [smoother.feed(motion.float()), smoother.finish()], dim=0
    )
    torch.cuda.synchronize()
    return smoothed.cpu(), time.perf_counter() - t0


def render_frames(renderer, motion: torch.Tensor, batch_size: int) -> torch.Tensor:
    frames = []
    for start in range(0, motion.shape[0], batch_size):
        batch = motion[start : start + batch_size]
        rgb, _ = renderer.render_batch_profile(batch)
        frames.append((rgb.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu())
    return torch.cat(frames, dim=0)


def label_band(text: str, width: int, height: int = 40) -> torch.Tensor:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=height * 5 // 8)
    draw.text(
        (width // 2, height // 2), text, fill=(255, 255, 255), font=font, anchor="mm"
    )
    return torch.from_numpy(np.asarray(img).copy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, type=str, nargs="+")
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
    parser.add_argument(
        "--renderers",
        default="mesh",
        help="Comma-separated renderers: mesh, gagavatar.",
    )
    parser.add_argument(
        "--gagavatar-avatar",
        default="1.jpg",
        help="Avatar id in tracked.pt for the gagavatar renderer.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    renderer_names = [r.strip() for r in args.renderers.split(",") if r.strip()]

    artalk_assets = ARTalkAssets.resolve(root=args.asset_dir)
    runtime = ARTalkRuntime(
        ARTalkRuntimeConfig(
            assets=artalk_assets,
            device=args.device,
            flame_scale=1.0,
        )
    )
    renderers = {
        name: build_renderer(name, runtime, artalk_assets, args, device)
        for name in renderer_names
    }
    streamers = {name: build_streamer(name, runtime, args, device) for name in models}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_path in args.audio:
        waveform, sample_rate = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(
            waveform.mean(0), orig_freq=sample_rate, new_freq=ARTALK_SAMPLE_RATE
        )
        stem = Path(audio_path).stem
        print(f"\n=== {stem}: {audio.shape[0] / ARTALK_SAMPLE_RATE:.2f}s ===", flush=True)

        motions = {}
        for name in models:
            motions[name], decode_s = generate_motion(streamers[name], audio, device)
            print(f"{name}: frames={motions[name].shape[0]} decode={decode_s:.1f}s", flush=True)

        for renderer_name, renderer in renderers.items():
            panels = []
            for name in models:
                t0 = time.perf_counter()
                video = render_frames(renderer, motions[name], args.render_batch_size)
                torch.cuda.synchronize()
                render_s = time.perf_counter() - t0
                path = output_dir / f"{stem}-{name}-{renderer_name}.mp4"
                write_video_with_audio(path, video, ARTALK_FPS, audio, ARTALK_SAMPLE_RATE)
                print(f"render={render_s:.1f}s -> {path}", flush=True)
                label = MODEL_LABELS.get(name, name)
                band = label_band(label, video.shape[2])
                panels.append(
                    torch.cat([band[None].expand(video.shape[0], -1, -1, -1), video], dim=1)
                )
            if len(panels) > 1:
                combined = torch.cat(panels, dim=2)
                path = output_dir / f"{stem}-{'-vs-'.join(models)}-{renderer_name}.mp4"
                write_video_with_audio(
                    path, combined, ARTALK_FPS, audio, ARTALK_SAMPLE_RATE
                )
                print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
