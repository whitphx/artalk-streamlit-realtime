"""The "AI-generated" badge drawn into every video frame the server sends.

It is part of the pixels rather than an element on the page, so it stays on
the face in the WebRTC stream, in screen recordings and in saved frames. The
browser renderer draws the same badge itself (``client_render_js/badge.js``).
"""

from __future__ import annotations

from functools import cache

import numpy as np
from artalk.realtime_pipeline import ARTalkPipeline
from PIL import Image, ImageDraw, ImageFont

AI_BADGE_TEXT = "AI-generated"
# Black at this opacity under white text keeps the text readable on a pale
# face as well as on the black background.
BACKING_ALPHA = 0.6


@cache
def badge_overlay(height: int, width: int) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Where the badge goes in a ``height`` x ``width`` frame, and what to
    blend there: ``frame * (1 - alpha) + color``, with the color already
    multiplied by alpha."""
    font_px = max(10, round(height * 0.036))
    font = ImageFont.load_default(size=font_px)
    left, top, right, bottom = font.getbbox(AI_BADGE_TEXT)
    pad_x, pad_y = round(font_px * 0.5), round(font_px * 0.3)
    w, h = right - left + 2 * pad_x, bottom - top + 2 * pad_y
    backing = Image.new("L", (w, h), 0)
    ImageDraw.Draw(backing).rounded_rectangle(
        (0, 0, w - 1, h - 1), radius=round(h * 0.3), fill=round(255 * BACKING_ALPHA)
    )
    text = Image.new("L", (w, h), 0)
    ImageDraw.Draw(text).text((pad_x - left, pad_y - top), AI_BADGE_TEXT, font=font, fill=255)
    back_a = np.asarray(backing, dtype=np.float32)[..., None] / 255.0
    text_a = np.asarray(text, dtype=np.float32)[..., None] / 255.0
    # White text over the black backing.
    alpha = text_a + back_a * (1.0 - text_a)
    color = np.repeat(255.0 * text_a, 3, axis=2)
    margin = round(height * 0.02)
    return height - margin - h, width - margin - w, 1.0 - alpha, color


def stamp_ai_badge(frames: np.ndarray) -> None:
    """Draws the badge in place into a (T, H, W, 3) uint8 batch."""
    y, x, keep, color = badge_overlay(frames.shape[1], frames.shape[2])
    h, w = keep.shape[:2]
    region = frames[:, y : y + h, x : x + w]
    region[...] = region * keep + color + 0.5


class BadgedARTalkPipeline(ARTalkPipeline):
    """ARTalkPipeline whose frames carry the badge.

    The badge goes on where the render worker turns a rendered batch into
    arrays, so it costs the worker rather than the outbound track's
    callback, and the resting frame shown before any speech has it too.
    """

    def _rgb_batch_to_arrays(self, rgb_batch):
        frames = super()._rgb_batch_to_arrays(rgb_batch)
        stamp_ai_badge(frames)
        return frames
