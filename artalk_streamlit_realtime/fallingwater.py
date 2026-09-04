"""Streaming adapter for the Fallingwater talking-head model.

Fallingwater (https://github.com/xg-chu/Fallingwater) is the streaming
successor to ARTalk by the same author. This module loads its
``FallingwaterGen`` checkpoint and exposes a streamer with the same
surface as ``artalk.streaming.ARTalkStreamer`` (``feed``/``finish``/
``reset``, ``patch_audio_length``, ``frames_per_chunk``,
``_audio_buffer``), so ``ARTalkPipeline`` can drive either model.

The chunk-decode loop is a port of ``FallingwaterGen.inference`` (see
``core/models/fallingwater_gen/models.py`` in the Fallingwater repo),
restructured from file-at-once to feed-as-you-go: the per-chunk carry
state there (previous chunk's audio and motion) becomes instance state
here.

Unit conventions: the app-side pipeline clocks everything in 16 kHz
samples and 25 fps frames; Fallingwater consumes 24 kHz audio and emits
108-dim motion (exp100 + gpose3 + jaw1 + eye4). This adapter takes 16 kHz
audio in, resamples per chunk, and returns 106-dim ARTalk-style motion
(exp100 + gpose3 + jaw3) out.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torchaudio

from .config import _env_flag

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16_000
FPS = 25


def _shim_transformers_v4() -> None:
    """Alias transformers v5 symbols the MOSS tokenizer's remote code
    imports; harmless no-ops on environments that already have them."""
    import transformers.modeling_utils as mu
    import transformers.utils as tu

    if not hasattr(mu, "PreTrainedAudioTokenizerBase"):
        mu.PreTrainedAudioTokenizerBase = mu.PreTrainedModel
    if not hasattr(tu, "auto_docstring"):

        def auto_docstring(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda obj: obj

        tu.auto_docstring = auto_docstring


def _install_constant_kv_cache() -> None:
    """Stop recomputing the per-chunk-constant K/V on every AR step.

    A chunk is decoded in ``sum(patch_nums)`` autoregressive steps (176 for
    this checkpoint), and each step re-runs the whole decoder. Only the
    partially decoded motion changes between steps: the audio, the previous
    chunk's motion and the style templates are fixed for the chunk, yet
    their projections are recomputed all 176 times. Cache them per chunk.

    Measured on a P100: 2.63 s -> 2.01 s per 4 s chunk (1.31x), with output
    bit-identical to the unpatched model. Patching upstream's classes is
    only safe while they look as expected, so the signatures are checked
    first and the patch is skipped rather than risking wrong output.
    Set FALLINGWATER_KV_CACHE=0 to skip it regardless.
    """
    import inspect

    from core.models.fallingwater_gen import transformer

    if not _env_flag("FALLINGWATER_KV_CACHE", True):
        return
    if getattr(transformer, "_const_kv_patched", False):
        return

    expected = {
        transformer.FusedAttn: ["self", "feat", "prev_feat", "style_feat", "self_attn_info"],
        transformer.CrossAttn: ["self", "x", "context", "cross_attn_info"],
    }
    for cls, params in expected.items():
        if list(inspect.signature(cls.forward).parameters) != params:
            logger.warning(
                "[fallingwater] %s.forward is not the signature this cache was "
                "written against; leaving it unpatched", cls.__name__,
            )
            return

    def fused_attn_forward(self, feat, prev_feat, style_feat, self_attn_info):
        q = self.rearrange_q(self.to_q(self.self_norm(feat)))
        cached = getattr(self, "_const_kv", None)
        if cached is None:
            style_k, style_v = self.rearrange_kv(
                self.style_to_kv(self.style_norm(style_feat))
            ).unbind(0)
            prev_k, prev_v = self.rearrange_kv(
                self.prev_to_kv(self.prev_norm(prev_feat))
            ).unbind(0)
            cached = (style_k, style_v, prev_k, prev_v)
            self._const_kv = cached
        style_k, style_v, prev_k, prev_v = cached
        self_k, self_v = self.rearrange_kv(self.self_to_kv(self.self_norm(feat))).unbind(0)
        k = torch.cat([style_k, prev_k, self_k], dim=1)
        v = torch.cat([style_v, prev_v, self_v], dim=1)
        q = self.rearrange_rope(self.rope(q, input_pos=self_attn_info["q_rope_pos"]))
        k = self.rearrange_rope(self.rope(k, input_pos=self_attn_info["k_rope_pos"]))
        v = self.rearrange_rope(v)
        out = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=self_attn_info["attn_bias"]
        )
        return self.to_out(self.rearrange_out(out))

    def cross_attn_forward(self, x, context, cross_attn_info):
        q = self.rearrange_qkv(self.q_proj(self.self_norm(x)))
        cached = getattr(self, "_const_kv", None)
        if cached is None:
            ctx = self.context_norm(context)
            k = self.rearrange_rope(
                self.rope(
                    self.rearrange_qkv(self.k_proj(ctx)),
                    input_pos=cross_attn_info["k_rope_pos"],
                )
            )
            v = self.rearrange_rope(self.rearrange_qkv(self.v_proj(ctx)))
            cached = (k, v)
            self._const_kv = cached
        k, v = cached
        q = self.rearrange_rope(self.rope(q, input_pos=cross_attn_info["q_rope_pos"]))
        out = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=cross_attn_info["attn_bias"]
        )
        return self.out_proj(self.rearrange_out(out))

    transformer.FusedAttn.forward = fused_attn_forward
    transformer.CrossAttn.forward = cross_attn_forward
    transformer._const_kv_patched = True


def _clear_constant_kv(model) -> None:
    for module in model.modules():
        if hasattr(module, "_const_kv"):
            del module._const_kv


def _absolutize_asset_paths(config: dict, repo_dir: str) -> None:
    """Resolve the config's repo-relative asset paths against the checkout.

    Fallingwater is written to run from its own directory, so the config
    baked into the checkpoint carries paths like ``./assets/...``. The app
    runs from elsewhere, and chdir would be unsafe with a live pipeline in
    other threads, so rewrite the paths instead. Only the motion statistics
    are needed at inference; the codec and audio-encoder weights come from
    the checkpoint and Hugging Face.
    """
    for key, value in config.items():
        if isinstance(value, dict):
            _absolutize_asset_paths(value, repo_dir)
        elif key == "STATS_PATH" and isinstance(value, str) and not os.path.isabs(value):
            config[key] = os.path.normpath(os.path.join(repo_dir, value))


def load_fallingwater_model(repo_dir: str | Path, checkpoint_path: str | Path, device):
    """Build ``FallingwaterGen`` from a self-contained checkpoint, the way
    the repo's ``infer.py`` does (``init_submodule=False``: codec weights
    ship inside the checkpoint and the audio encoder loads from HF)."""
    # expanduser: see load_artalk1s_model.
    resolved = Path(repo_dir).expanduser().resolve()
    repo_dir = str(resolved)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
        importlib.invalidate_caches()
    _shim_transformers_v4()

    try:
        from core.libs.utils import ConfigDict
        from core.models import build_model
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] != "core":
            raise
        raise ModuleNotFoundError(
            f"no `core` package under {resolved} "
            f"(exists={resolved.is_dir()}, from {repo_dir!r}); "
            "pass --fallingwater-dir pointing at a Fallingwater checkout"
        ) from exc

    _install_constant_kv_cache()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "meta_cfg" not in ckpt:
        raise ValueError(f"{checkpoint_path} has no meta_cfg; not a self-contained checkpoint")
    _absolutize_asset_paths(ckpt["meta_cfg"], repo_dir)
    meta_cfg = ConfigDict(ckpt["meta_cfg"], gpus=1, cli_args=[])
    model = build_model(meta_cfg.MODEL, init_submodule=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    missing = [k for k in missing if not k.startswith("audio_encoder")]
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval().to(device)
    model._fallingwater_meta_cfg = meta_cfg
    return model


def motion_108_to_106(motion: torch.Tensor) -> torch.Tensor:
    """(N, 108) exp100+gpose3+jaw1+eye4 -> (N, 106) exp100+gpose3+jaw3.

    Fallingwater's FLAME pads its scalar jaw code to a rotation vector as
    ``[jaw, 0, 0]``; eye pose has no slot in the 106-dim layout (the
    GAGAvatar render path zeroes eyes regardless)."""
    exp_gpose = motion[:, :103]
    jaw = motion[:, 103:104]
    return torch.cat([exp_gpose, jaw, torch.zeros_like(motion[:, :2])], dim=-1)


class FallingwaterStreamer:
    """Drives ``FallingwaterGen`` chunk-by-chunk over a live audio feed."""

    def __init__(self, model, tau: float = 1.0, cfg: float = 2.0):
        self.model = model
        self.tau = float(tau)
        self.cfg = float(cfg)
        self.frames_per_chunk = max(model.patch_nums)
        # Pipeline-facing chunk size is in 16 kHz samples; the model-facing
        # one is in its own (24 kHz) rate.
        self.patch_audio_length = int(self.frames_per_chunk / FPS * PIPELINE_SAMPLE_RATE)
        self._model_sample_rate = int(model._sample_rate)
        self._model_chunk_samples = int(self.frames_per_chunk / FPS * self._model_sample_rate)
        self.motion_dim = 106
        self.reset()

    @property
    def device(self):
        return self.model.device

    @torch.inference_mode()
    def reset(self):
        model = self.model
        self._audio_buffer = torch.zeros(0, dtype=torch.float32, device=self.device)
        # Previous chunk's audio in the model's sample rate; zeros before
        # the first chunk, matching the leading zero-chunk pad in
        # FallingwaterGen.inference.
        self._prev_model_audio = torch.zeros(
            1, self._model_chunk_samples, dtype=torch.float32, device=self.device
        )
        prev_uncond = torch.zeros(
            1, self.frames_per_chunk, model.motion_dim, dtype=torch.float32, device=self.device
        )
        self._prev_motion_code = torch.cat([prev_uncond, prev_uncond], dim=0)
        # Style templates are zeros, as in infer.py (no style conditioning), so
        # the conditional and unconditional halves of the CFG pair coincide.
        meta = model._fallingwater_meta_cfg
        num_templates = max(1, int(meta.DATASET.TEMPLATE_NUM))
        template_len = int(meta.DATASET.TEMPLATE_LEN)
        self._templates = torch.zeros(
            2, num_templates * template_len, model.motion_dim,
            dtype=torch.float32, device=self.device,
        )
        self._audio_uncond_feat = None

    @torch.inference_mode()
    def feed(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() != 1:
            raise ValueError(f"audio must be 1-D, got shape {tuple(audio.shape)}")
        audio = audio.to(device=self.device, dtype=torch.float32)
        self._audio_buffer = torch.cat([self._audio_buffer, audio])

        outputs = []
        while self._audio_buffer.shape[0] >= self.patch_audio_length:
            chunk = self._audio_buffer[: self.patch_audio_length]
            self._audio_buffer = self._audio_buffer[self.patch_audio_length:]
            outputs.append(self._step_chunk(chunk))
        if outputs:
            return torch.cat(outputs, dim=0)
        return torch.zeros(0, self.motion_dim, dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def finish(self) -> torch.Tensor:
        valid_samples = self._audio_buffer.shape[0]
        if valid_samples == 0:
            return torch.zeros(0, self.motion_dim, dtype=torch.float32, device=self.device)
        valid_frames = math.ceil(valid_samples / PIPELINE_SAMPLE_RATE * FPS)
        pad = self.patch_audio_length - valid_samples
        chunk = torch.cat([self._audio_buffer, self._audio_buffer.new_zeros(pad)])
        self._audio_buffer = self._audio_buffer.new_zeros(0)
        motion = self._step_chunk(chunk)
        return motion[:valid_frames]

    def _step_chunk(self, chunk_16k: torch.Tensor) -> torch.Tensor:
        """One 4 s chunk: 16 kHz audio in, (frames_per_chunk, 106) out.

        Port of one iteration of the chunk loop in
        ``FallingwaterGen.inference``, with CFG's conditional/unconditional
        pair stacked in the batch dimension exactly as upstream does."""
        model = self.model
        # The cached projections belong to the previous chunk's audio and
        # motion; this chunk's differ.
        _clear_constant_kv(model)
        model_audio = torchaudio.functional.resample(
            chunk_16k[None], orig_freq=PIPELINE_SAMPLE_RATE, new_freq=self._model_sample_rate
        )
        if model_audio.shape[1] < self._model_chunk_samples:
            model_audio = torch.cat(
                [model_audio, model_audio.new_zeros(1, self._model_chunk_samples - model_audio.shape[1])],
                dim=1,
            )
        else:
            model_audio = model_audio[:, : self._model_chunk_samples]

        audio_pair = torch.cat([self._prev_model_audio, model_audio], dim=-1)
        audio_feat = model.audio_encoder(audio_pair)
        if self._audio_uncond_feat is None:
            self._audio_uncond_feat = model.audio_encoder(torch.zeros_like(audio_pair))
        audio_feat = torch.cat([audio_feat, self._audio_uncond_feat], dim=0)

        sos_token = model.sos_embed.expand(2, 1, -1)
        prev_motion_feat = model.get_motion_feat(self._prev_motion_code)

        next_ar_vqfeat = torch.cat(
            [sos_token, torch.zeros_like(sos_token).repeat(1, sum(model.patch_nums) - 1, 1)], dim=1
        )
        pred_motion_bits = torch.zeros(
            1, sum(model.patch_nums), model.base_codec.code_dim, device=self.device
        )
        for inseq_id in model.attn_blocks.seq_to_id:
            attn_feat = model.attn_blocks(next_ar_vqfeat, audio_feat, prev_motion_feat, self._templates)
            motion_logits = model.logits_head(attn_feat)
            motion_logits = motion_logits.mul(1 / self.tau)
            motion_logits = motion_logits.view(motion_logits.shape[0], motion_logits.shape[1], -1, 2)
            if self.cfg > 1.0:
                motion_logits = self.cfg * motion_logits[:1] + (1 - self.cfg) * motion_logits[1:]
            else:
                motion_logits = motion_logits[:1]
            motion_bits = _sample_idx_with_top_p(motion_logits)
            pred_motion_bits[:, inseq_id] = motion_bits[:, inseq_id]
            next_ar_vqfeat = model.base_codec.vqidx_to_next_feat(
                pred_motion_bits, len(model.patch_nums) - 2, "accum_next"
            )
            next_ar_vqfeat = model.code_token_embed(next_ar_vqfeat)
            next_ar_vqfeat = torch.cat([next_ar_vqfeat, next_ar_vqfeat], dim=0)
            next_ar_vqfeat = torch.cat([sos_token, next_ar_vqfeat], dim=1)

        pred_motion_code = model.base_codec.vqidx_to_motion(pred_motion_bits)
        self._prev_motion_code = torch.cat([pred_motion_code, pred_motion_code], dim=0)
        self._prev_model_audio = model_audio
        return motion_108_to_106(pred_motion_code[0])


def _sample_idx_with_top_p(logits_BlcV: torch.Tensor, top_p: float = 0.97) -> torch.Tensor:
    # Same nucleus sampling as upstream's sample_idx_with_top_p_.
    B, l, c, V = logits_BlcV.shape
    logits_BlV = logits_BlcV.view(B, -1, V)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(
            sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove),
            -torch.inf,
        )
    sampled_idx = torch.multinomial(
        logits_BlV.softmax(dim=-1).view(-1, V), num_samples=1, replacement=True
    )
    return sampled_idx.reshape(B, l, c)
