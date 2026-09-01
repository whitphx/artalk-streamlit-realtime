"""Streaming adapter for the frame-by-frame causal motion model.

Wraps ``artalk_frame.CausalFrameModel`` (one motion frame per 40 ms of
audio, bounded GRU state) in the streamer surface ``ARTalkPipeline``
consumes, following the adapters for the other external models. The
model package is imported from a caller-supplied checkout or snapshot
directory, so training-side code stays out of this repo.

With ``frames_per_chunk`` of 1 the pipeline's chunk machinery degenerates
to per-frame operation: the silence pump's flush budget becomes one
frame, and the renderer receives 1-frame batches, which costs the
batch-8 amortization. If per-frame rendering proves too slow at 512, the
place to buffer a few frames is the pipeline's segmenting, not this
adapter — buffering here would silently re-add latency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PIPELINE_SAMPLE_RATE = 16_000
FPS = 25


def load_frame_model(package_dir: str | Path, checkpoint_path: str | Path, device):
    """Load a trained CausalFrameModel checkpoint.

    ``package_dir`` is a directory containing the ``artalk_frame`` package
    (the training checkout or a snapshot of it). Checkpoints carry the
    trainer's EMA weights plus ``meta_cfg``.
    """
    package_dir = str(Path(package_dir).resolve())
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)

    from artalk_frame import CausalFrameModel

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "meta_cfg" not in ckpt:
        raise ValueError(f"{checkpoint_path} has no meta_cfg; not a self-contained checkpoint")
    m = ckpt["meta_cfg"]["MODEL"]
    model = CausalFrameModel(
        motion_dim=m["MOTION_DIM"],
        expression_dim=m["EXPRESSION_DIM"],
        sample_rate=m["SAMPLE_RATE"],
        motion_fps=m["MOTION_FPS"],
        audio_dim=m["AUDIO_DIM"],
        style_dim=m["STYLE_DIM"],
        motion_embed_dim=m["MOTION_EMBED_DIM"],
        hidden_dim=m["HIDDEN_DIM"],
        num_layers=m["NUM_LAYERS"],
        dropout=m["DROPOUT"],
        style_dropout=m["STYLE_DROPOUT"],
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    stray = [k for k in missing if not k.startswith("face_decoder.")]
    if stray or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={stray}, unexpected={unexpected}")
    model.eval().to(device)
    return model


def motion_108_to_106(motion: torch.Tensor) -> torch.Tensor:
    """(N, 108) exp100+gpose3+jaw1+eye4 -> (N, 106) exp100+gpose3+jaw3.

    For render paths that speak the released layout. Eye motion has no
    slot there; renderers that support eyes should consume the native
    108 frame instead (``eye4`` maps to FLAME's 6-dim eye pose as
    ``[e0, e1, 0, e2, e3, 0]``)."""
    exp_gpose = motion[:, :103]
    jaw = motion[:, 103:104]
    return torch.cat([exp_gpose, jaw, torch.zeros_like(motion[:, :2])], dim=-1)


class FrameModelStreamer:
    """Per-frame streaming with the pipeline's streamer surface.

    Emits released-layout 106-dim frames by default so every existing
    render path works; ``native_layout=True`` emits the model's own
    layout for eye-aware renderers.
    """

    def __init__(self, model, style_motion=None, native_layout: bool = False):
        self.model = model
        self.frames_per_chunk = 1
        self.patch_audio_length = int(model.samples_per_frame)
        self._native_layout = bool(native_layout)
        self.motion_dim = model.motion_dim if native_layout else 106
        self._style_motion = style_motion
        self.reset()

    @property
    def device(self):
        return self.model.device

    @torch.inference_mode()
    def reset(self):
        self._audio_buffer = torch.zeros(0, dtype=torch.float32, device=self.device)
        self._state = self.model.init_stream_state(
            batch_size=1, style_motion=self._style_motion
        )

    @torch.inference_mode()
    def feed(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() != 1:
            raise ValueError(f"audio must be 1-D, got shape {tuple(audio.shape)}")
        audio = audio.to(device=self.device, dtype=torch.float32)
        self._audio_buffer = torch.cat([self._audio_buffer, audio])

        outputs = []
        spf = self.patch_audio_length
        while self._audio_buffer.shape[0] >= spf:
            frame_audio = self._audio_buffer[:spf]
            self._audio_buffer = self._audio_buffer[spf:]
            motion, self._state = self.model.step(frame_audio[None], self._state)
            outputs.append(motion)
        if not outputs:
            return torch.zeros(0, self.motion_dim, dtype=torch.float32, device=self.device)
        out = torch.cat(outputs, dim=0)
        return out if self._native_layout else motion_108_to_106(out)

    @torch.inference_mode()
    def finish(self) -> torch.Tensor:
        if self._audio_buffer.shape[0] == 0:
            return torch.zeros(0, self.motion_dim, dtype=torch.float32, device=self.device)
        pad = self.patch_audio_length - self._audio_buffer.shape[0]
        return self.feed(self._audio_buffer.new_zeros(pad))
