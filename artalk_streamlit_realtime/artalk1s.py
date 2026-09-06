"""Streaming adapter for retrained short-chunk ARTalk generators.

Checkpoints from the ARTalk repo's ``train_code`` are a different
architecture from the released runtime model (no state-dict overlap), so
they cannot load into ``artalk.models.BitwiseARModel``. This module loads
them through the training code itself and exposes the streamer surface
``ARTalkPipeline`` consumes, following ``fallingwater.py`` — the training
generator shares that model family's design.

The generator's own ``inference()`` carries one chunk of previous motion
and one chunk of style, but the 1 s recipe trains with three chunks of
previous context and a 100-frame style window
(``docs/chunk-size-retraining.md``). Driving ``inference()`` one chunk
per call lets this adapter maintain those trained context lengths
externally instead of inheriting the narrower defaults.
"""

from __future__ import annotations

import importlib
import logging
import math
import sys
from pathlib import Path

import torch

from .config import _env_flag

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16_000
FPS = 25


def load_artalk1s_model(train_code_dir: str | Path, checkpoint_path: str | Path, device):
    """Build the training-code generator from a self-contained checkpoint.

    ``init_submodule=False`` skips the config's ``VAE_PATH`` reload: the
    codec weights ship inside the checkpoint. The audio encoder loads from
    the Hugging Face cache and is excluded from checkpoints by design.
    """
    # expanduser: a "~/..." path arriving unexpanded would otherwise resolve
    # against the cwd and be inserted silently.
    argument = train_code_dir
    resolved = Path(train_code_dir).expanduser().resolve()
    train_code_dir = str(resolved)
    if train_code_dir not in sys.path:
        sys.path.insert(0, train_code_dir)
        # These checkouts live on a shared filesystem that other work mutates,
        # so a directory listing this process cached earlier can be stale.
        importlib.invalidate_caches()

    try:
        from core.libs.utils import ConfigDict
        from core.models import build_model
    except ModuleNotFoundError as exc:
        # Only explain a missing `core` itself; a missing dependency *of* the
        # training code is a different problem and keeps its own message.
        if (exc.name or "").split(".")[0] != "core":
            raise
        raise ModuleNotFoundError(
            f"no `core` package under {resolved} "
            f"(exists={resolved.is_dir()}, argument was {argument!r}); "
            "pass --artalk1s-train-code-dir pointing at an ARTalk train_code "
            "directory"
        ) from exc

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "meta_cfg" not in ckpt:
        raise ValueError(f"{checkpoint_path} has no meta_cfg; not a self-contained checkpoint")
    meta_cfg = ConfigDict(ckpt["meta_cfg"], gpus=1, cli_args=[])
    model = build_model(meta_cfg.MODEL, init_submodule=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    stray = [
        k for k in missing
        if not k.startswith(("audio_encoder.", "base_codec.face_decoder."))
    ]
    if stray or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={stray}, unexpected={unexpected}")
    model.eval().to(device)
    model._artalk1s_meta_cfg = meta_cfg
    return model


def _install_decoder_constant_cache() -> bool:
    """Keep the decoder's constant index tensors on the GPU.

    ``MixedARTalkDecoder.forward`` rebuilds its attention mask and rope
    positions on the CPU on every scale step and copies them to the
    device, and ``CrossAttention.forward`` does the same for its context
    positions. Both are pure functions of shapes that never change within
    a session, so cache them per shape on the module's device. Besides
    removing per-step CPU work, this eliminates the pageable
    host-to-device copies that CUDA graph capture forbids. Patching
    upstream classes is only safe while they look as expected, so the
    signatures are checked first.
    """
    import inspect

    from core.models.artalk_gen import transformer

    if getattr(transformer, "_artalk1s_const_cache", False):
        return True

    expected = {
        transformer.MixedARTalkDecoder.expand_attn_mask: ["self", "num_style_prev"],
        transformer.CrossAttention.forward: ["self", "x", "context", "rope_pos", "context_rope_pos"],
    }
    for fn, params in expected.items():
        if list(inspect.signature(fn).parameters) != params:
            logger.warning(
                "[artalk1s] %s is not the signature this cache was written "
                "against; leaving the decoder unpatched", fn.__qualname__,
            )
            return False

    orig_expand = transformer.MixedARTalkDecoder.expand_attn_mask

    def expand_attn_mask_cached(self, num_style_prev=2):
        cache = getattr(self, "_expand_cache", None)
        if cache is None:
            cache = self._expand_cache = {}
        entry = cache.get(num_style_prev)
        if entry is None:
            mask, rope_pos = orig_expand(self, num_style_prev)
            device = self.lvl_embed.weight.device
            entry = (mask.to(device), rope_pos.to(device))
            cache[num_style_prev] = entry
        return entry

    def cross_attention_forward(self, x, context, rope_pos, context_rope_pos):
        patch_len, patch_offsets = context_rope_pos
        cache = getattr(self, "_ctx_rope_cache", None)
        if cache is None:
            cache = self._ctx_rope_cache = {}
        key = (patch_len, patch_offsets, context.shape[1])
        ctx_pos = cache.get(key)
        if ctx_pos is None:
            ctx_pos = (
                torch.linspace(0, patch_len, steps=(context.shape[1] + 1))[:-1].long()
                + patch_offsets
            ).to(x.device)
            cache[key] = ctx_pos
        x, context = self.self_norm(x), self.context_norm(context)
        q = self.rearrange_qkv(self.q_proj(x))
        k = self.rearrange_qkv(self.k_proj(context))
        v = self.rearrange_qkv(self.v_proj(context))
        q = self.rearrange_rope(self.rope(q, input_pos=rope_pos))
        k = self.rearrange_rope(self.rope(k, input_pos=ctx_pos))
        v = self.rearrange_rope(v)
        out = torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v)
        return self.out_proj(self.rearrange_out(out))

    transformer.MixedARTalkDecoder.expand_attn_mask = expand_attn_mask_cached
    transformer.CrossAttention.forward = cross_attention_forward
    transformer._artalk1s_const_cache = True
    return True


class _GraphedDecoder:
    """Replay the AR decoder's kernel sequence from a CUDA graph.

    The scale steps are launch-bound (a step costs the same at sequence
    length 1 and 31), so collapsing each step's ~200 launches into one
    graph replay cuts it several-fold. torch.compile's reduce-overhead
    mode achieves the same standalone but its cudagraph trees re-record
    whenever other GPU work runs between calls, which the pipeline's
    interleaved rendering does every chunk; a manual capture with private
    static buffers is immune to interleaving. One graph per step shape
    (three per chunk size), each holding only decoder-sized activations.
    """

    def __init__(self, module):
        self.module = module
        self._graphs: dict[tuple, tuple] = {}
        self._capture_failed = False

    def __call__(self, *inputs):
        if self._capture_failed:
            return self.module(*inputs)
        key = tuple(tuple(t.shape) for t in inputs)
        entry = self._graphs.get(key)
        if entry is None:
            try:
                entry = self._capture(inputs)
            except Exception:
                self._capture_failed = True
                self._graphs.clear()
                logger.warning(
                    "[artalk1s] decoder graph capture failed; continuing eager",
                    exc_info=True,
                )
                return self.module(*inputs)
            self._graphs[key] = entry
        graph, static_in, static_out = entry
        for dst, src in zip(static_in, inputs):
            dst.copy_(src)
        graph.replay()
        # The static output is overwritten by the next replay; hand the
        # caller its own copy.
        return static_out.clone()

    def _capture(self, inputs) -> tuple:
        # Quiesce the device: capture aborts if other in-flight work
        # interleaves, and the pipeline runs with stage syncs disabled.
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.module(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        static_in = tuple(t.clone() for t in inputs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            static_out = self.module(*static_in)
        return (graph, static_in, static_out)


def _fast_decode_supported(model) -> bool:
    """The fast chunk step inlines ``inference()``, so it is only safe while
    the model still looks like the code it was written against."""
    import inspect

    # model.inference is a bound method, so "self" is not in the signature.
    expected = [
        "audio", "style_motion_code", "prev_motion_code", "tau", "cfg", "kwargs",
    ]
    if list(inspect.signature(model.inference).parameters) != expected:
        return False
    return all(
        hasattr(model, attr)
        for attr in (
            "audio_encoder", "get_motion_feat", "code_token_embed", "sos_embed",
            "attn_blocks", "logits_head", "patch_nums",
        )
    ) and all(
        hasattr(model.base_codec, attr)
        for attr in ("vqidx_to_next_feat", "vqidx_to_motion")
    )


class ARTalk1sStreamer:
    """Drives a train-code ARTalk generator chunk-by-chunk over live audio.

    The per-chunk decode re-encodes context that cannot have changed: the
    style window is fixed until ``set_style``, the CFG-unconditional halves
    are all zeros, and all previous-context windows but the newest were
    already encoded on earlier chunks. At a 1 s chunk those fixed costs
    recur four times as often as the 4 s recipe paid them, so the default
    path inlines ``inference()``'s single-chunk step and caches every
    per-chunk-constant feature. Cached entries are produced by the same
    ops at the same tensor shapes as the plain path, keeping the output
    identical. Set ``ARTALK1S_FAST_DECODE=0`` to fall back to plain
    ``inference()``.
    """

    def __init__(self, model, style_motion=None, tau: float = 1.0, cfg: float = 2.0):
        self.model = model
        self.tau = float(tau)
        self.cfg = float(cfg)
        self.frames_per_chunk = max(model.patch_nums)
        self.patch_audio_length = int(self.frames_per_chunk / FPS * PIPELINE_SAMPLE_RATE)
        self.motion_dim = model.motion_dim
        meta = model._artalk1s_meta_cfg
        self._prev_frames = int(meta.DATASET.PREV_LENGTH)
        self._style_frames = int(meta.DATASET.STYLE_LENGTH)
        self._fast = (
            _env_flag("ARTALK1S_FAST_DECODE", True)
            and self._prev_frames % self.frames_per_chunk == 0
            and _fast_decode_supported(model)
        )
        if self._fast:
            from core.models.artalk_gen.models import sample_idx_with_top_p_

            self._sample_bits = sample_idx_with_top_p_
            self._decoder = model.attn_blocks
            if (
                _env_flag("ARTALK1S_GRAPH_DECODER", True)
                and self.device.type == "cuda"
                # Capture needs the decoder's constant tensors on-device;
                # without the cache every step does pageable host-to-device
                # copies, which capture forbids.
                and _install_decoder_constant_cache()
            ):
                self._decoder = _GraphedDecoder(model.attn_blocks)
        else:
            logger.warning("[artalk1s] fast decode disabled; using plain inference()")
        self.set_style(style_motion)
        self.reset()

    @property
    def device(self):
        return self.model.device

    @torch.inference_mode()
    def set_style(self, style_motion=None):
        """Condition generation on a style motion window.

        Accepts ``(frames, motion_dim)`` of any length; shorter windows
        (e.g. the released 50-frame presets) are tiled to the trained
        style length. ``None`` selects the zero unconditional-style input
        (STYLE_FREE training drops style the same way).
        """
        if style_motion is None:
            self._style_motion = torch.zeros(
                1, self._style_frames, self.motion_dim, dtype=torch.float32, device=self.device
            )
            self._refresh_style_cache()
            return
        if style_motion.dim() == 2:
            style_motion = style_motion[None]
        if style_motion.dim() != 3 or style_motion.shape[-1] != self.motion_dim:
            raise ValueError(
                f"style_motion must be (frames, {self.motion_dim}), "
                f"got shape {tuple(style_motion.shape)}"
            )
        reps = -(-self._style_frames // style_motion.shape[1])
        style_motion = style_motion.repeat(1, reps, 1)[:, : self._style_frames]
        self._style_motion = style_motion.to(device=self.device, dtype=torch.float32)
        self._refresh_style_cache()

    @torch.inference_mode()
    def _refresh_style_cache(self):
        if not self._fast:
            return
        style2 = torch.cat(
            [self._style_motion, torch.zeros_like(self._style_motion)], dim=0
        )
        self._style_feat = self.model.code_token_embed(
            self.model.get_motion_feat(style2)
        )

    @torch.inference_mode()
    def reset(self):
        self._audio_buffer = torch.zeros(0, dtype=torch.float32, device=self.device)
        self._prev_motion = torch.zeros(
            1, self._prev_frames, self.motion_dim, dtype=torch.float32, device=self.device
        )
        if self._fast:
            self._init_feature_cache()

    @torch.inference_mode()
    def _init_feature_cache(self):
        """Precompute every per-chunk-constant decoder input.

        Each cache entry is produced by the same op at the same tensor
        shape ``inference()`` would use, so downstream results match the
        plain path exactly: context windows are encoded as CFG pairs
        (conditional row + zero unconditional row), matching the
        batch-of-2 ``get_motion_feat`` calls inside ``inference()``.
        """
        m = self.model
        zero_pair = torch.zeros(
            2, self.frames_per_chunk, self.motion_dim, dtype=torch.float32, device=self.device
        )
        zero_feat = m.code_token_embed(m.get_motion_feat(zero_pair))
        n_windows = self._prev_frames // self.frames_per_chunk
        # Conditional windows, oldest first; the unconditional half of the
        # previous context is all zeros for every chunk.
        self._prev_window_feats = [zero_feat[0:1]] * n_windows
        self._zero_window_feat = zero_feat[1:2]
        self._prev_uncond_feat = torch.cat([self._zero_window_feat] * n_windows, dim=1)

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
        return self._step_chunk(chunk)[:valid_frames]

    def _step_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if self._fast:
            return self._step_chunk_fast(chunk)
        out = self.model.inference(
            chunk[None],
            style_motion_code=self._style_motion,
            prev_motion_code=self._prev_motion,
            tau=self.tau,
            cfg=self.cfg,
        )
        motion = out["pred_motion_code"]  # (1, frames_per_chunk, motion_dim)
        self._prev_motion = torch.cat([self._prev_motion, motion], dim=1)[
            :, -self._prev_frames :
        ]
        return motion[0]

    @torch.inference_mode()
    def _step_chunk_fast(self, chunk: torch.Tensor) -> torch.Tensor:
        """One chunk of ``inference()`` with cached constant context.

        Mirrors the model's own loop body op for op; only *when* the
        context features are computed changes.
        """
        m = self.model
        audio_feat = m.audio_encoder(chunk[None])
        audio_feat2 = torch.cat([audio_feat, torch.zeros_like(audio_feat)], dim=0)
        prev_feat = torch.cat(
            [torch.cat(self._prev_window_feats, dim=1), self._prev_uncond_feat], dim=0
        )
        sos = m.sos_embed.expand(2, 1, -1)
        seq = sos
        patch_bits: list[torch.Tensor] = []
        for pidx in range(len(m.patch_nums)):
            attn_feat = self._decoder(seq, audio_feat2, prev_feat, self._style_feat)
            logits = m.logits_head(attn_feat)
            logits = logits[:, sum(m.patch_nums[:pidx]) :]
            logits = logits.mul(1 / self.tau)
            logits = logits.view(logits.shape[0], logits.shape[1], -1, 2)
            if self.cfg > 1.0:
                logits = self.cfg * logits[:1] + (1 - self.cfg) * logits[1:]
            else:
                logits = logits[:1]
            patch_bits.append(self._sample_bits(logits))
            if pidx < len(m.patch_nums) - 1:
                nxt = m.base_codec.vqidx_to_next_feat(
                    torch.cat(patch_bits, dim=1), pidx, "accum_next"
                )
                nxt = m.code_token_embed(nxt)
                nxt = torch.cat([nxt, nxt], dim=0)
                seq = torch.cat([sos, nxt], dim=1)
        motion = m.base_codec.vqidx_to_motion(torch.cat(patch_bits, dim=1))
        # Roll the caches: encode only the newly generated window, as a
        # CFG pair so the kernel shapes match the plain path.
        new_pair = torch.cat([motion, torch.zeros_like(motion)], dim=0)
        new_feat = m.code_token_embed(m.get_motion_feat(new_pair))
        self._prev_window_feats = self._prev_window_feats[1:] + [new_feat[0:1]]
        self._prev_motion = torch.cat([self._prev_motion, motion], dim=1)[
            :, -self._prev_frames :
        ]
        return motion[0]
