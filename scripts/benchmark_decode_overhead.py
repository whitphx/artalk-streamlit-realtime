#!/usr/bin/env python
"""Decompose the ARTalk-1s per-chunk decode cost.

The 1 s model pays its `inference()` fixed costs four times as often as
the 4 s model, and the pipeline benchmark shows decode is where the 1 s
configuration loses (110 ms per 1 s chunk vs 74 ms per 4 s chunk on an
A100). This times the stages of one steady-state chunk: audio encoder,
prev/style context encoding, each autoregressive scale step, and the
codec decode, so the fix targets the right stage.

Run from the repository root in the app environment:

    python scripts/benchmark_decode_overhead.py \
        --train-code-dir <train_code> --checkpoint <stage2.pt>
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artalk_streamlit_realtime.artalk1s import ARTalk1sStreamer, load_artalk1s_model


def sync_ms(fn, iters: int) -> float:
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-code-dir", default=os.environ.get("ARTALK1S_TRAIN_CODE_DIR"), type=str
    )
    ap.add_argument(
        "--checkpoint", default=os.environ.get("ARTALK1S_CHECKPOINT"), type=str
    )
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--iters", default=30, type=int)
    ap.add_argument(
        "--compile-decoder",
        default=None,
        type=str,
        help="torch.compile mode for the AR decoder (e.g. reduce-overhead).",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_artalk1s_model(args.train_code_dir, args.checkpoint, device)
    if args.compile_decoder:
        print(f"compiling decoder with mode={args.compile_decoder}")
        model.attn_blocks = torch.compile(
            model.attn_blocks, mode=args.compile_decoder, dynamic=False
        )
    streamer = ARTalk1sStreamer(model)
    # The breakdown and the "chunk total" measure the plain inference()
    # path; the fast path is timed against it at the end.
    streamer._fast = False

    torch.manual_seed(0)
    audio = torch.randn(16000, device=device) * 0.05
    # Steady state: a non-empty previous-context window.
    streamer.feed(audio)
    prev = streamer._prev_motion
    style = streamer._style_motion

    chunk = audio[None]
    patch_len = max(model.patch_nums)

    # End-to-end chunk, through the streamer step.
    def full_chunk():
        torch.manual_seed(0)
        streamer._prev_motion = prev
        streamer._step_chunk(audio)

    total_ms = sync_ms(full_chunk, args.iters)

    # Stage 1: audio encoder.
    audio_ms = sync_ms(lambda: model.audio_encoder(chunk), args.iters)

    # Stage 2: conditioning features, at inference()'s CFG-doubled batch.
    prev2 = torch.cat([prev, torch.zeros_like(prev)], dim=0)
    style2 = torch.cat([style, torch.zeros_like(style)], dim=0)
    style_ms = sync_ms(lambda: model.get_motion_feat(style2), args.iters)
    prev_ms = sync_ms(lambda: model.get_motion_feat(prev2), args.iters)

    # Stage 3: the AR scale steps. Reproduce inference()'s loop with the
    # CFG-doubled batch and time each decoder call separately.
    from core.models.artalk_gen.models import sample_idx_with_top_p_

    audio_feat1 = model.audio_encoder(chunk)
    audio_feats = torch.cat([audio_feat1, torch.zeros_like(audio_feat1)], dim=0)
    style_feat = model.code_token_embed(model.get_motion_feat(style2))
    prev_feat = model.code_token_embed(model.get_motion_feat(prev2))
    sos = model.sos_embed.expand(2, 1, -1)

    step_ms = {}
    seq = sos
    patch_bits = []
    for pidx, pn in enumerate(model.patch_nums):
        cur = seq
        step_ms[f"decoder_step{pidx} (seq {cur.shape[1]})"] = sync_ms(
            lambda cur=cur: model.attn_blocks(cur, audio_feats, prev_feat, style_feat),
            args.iters,
        )
        attn_feat = model.attn_blocks(cur, audio_feats, prev_feat, style_feat)
        logits = model.logits_head(attn_feat)[:, sum(model.patch_nums[:pidx]) :]
        logits = logits.view(logits.shape[0], logits.shape[1], -1, 2)
        logits = 2.0 * logits[:1] + (1 - 2.0) * logits[1:]
        bits = sample_idx_with_top_p_(logits)
        patch_bits.append(bits)
        if pidx < len(model.patch_nums) - 1:
            nxt = model.base_codec.vqidx_to_next_feat(
                torch.cat(patch_bits, dim=1), pidx, "accum_next"
            )
            nxt = model.code_token_embed(nxt)
            nxt = torch.cat([nxt, nxt], dim=0)
            seq = torch.cat([sos, nxt], dim=1)

    # Stage 4: codec decode of the final bits.
    bits_full = torch.cat(patch_bits, dim=1)
    codec_ms = sync_ms(lambda: model.base_codec.vqidx_to_motion(bits_full), args.iters)

    print(f"\nchunk total (streamer._step_chunk): {total_ms:7.2f} ms")
    rows = {
        "audio_encoder": audio_ms,
        "get_motion_feat(style 100)": style_ms,
        "get_motion_feat(prev 75)": prev_ms,
        **step_ms,
        "codec vqidx_to_motion": codec_ms,
    }
    accounted = 0.0
    for name, ms in rows.items():
        print(f"  {name:<28} {ms:7.2f} ms")
        accounted += ms
    print(f"  {'(sum of stages)':<28} {accounted:7.2f} ms")

    # Fast-path parity and speed: identical RNG stream, so any divergence
    # is a real numerical difference, not sampling noise.
    fast = ARTalk1sStreamer(model)
    slow = ARTalk1sStreamer(model)
    slow._fast = False
    if not fast._fast:
        print("\nfast decode unsupported on this model; parity check skipped")
        os._exit(0)
    torch.manual_seed(7)
    session = torch.randn(16000 * 8 + 4000, device=device) * 0.05
    torch.manual_seed(0)
    out_fast = torch.cat([fast.feed(session), fast.finish()], dim=0)
    torch.manual_seed(0)
    out_slow = torch.cat([slow.feed(session), slow.finish()], dim=0)
    identical = torch.equal(out_fast, out_slow)
    max_diff = (out_fast - out_slow).abs().max().item()
    print(f"\nfast-path parity over 8.25 s: identical={identical} max_diff={max_diff:.3e}")

    fast.reset()
    fast.feed(audio)
    def fast_chunk():
        torch.manual_seed(0)
        fast._step_chunk(audio)
    fast_ms = sync_ms(fast_chunk, args.iters)
    print(f"chunk total fast path: {fast_ms:7.2f} ms (plain {total_ms:7.2f} ms, "
          f"{total_ms / fast_ms:.2f}x)")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
