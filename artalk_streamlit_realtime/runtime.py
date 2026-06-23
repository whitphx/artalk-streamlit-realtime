"""Cached runtime loaders for packaged ARTalk and GAGAvatar."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from artalk.assets import ARTalkAssets
from artalk.flame_model import RenderMesh
from artalk.runtime import ARTalkRuntime, ARTalkRuntimeConfig, available_styles

from .config import DEFAULT_STYLE


@st.cache_resource
def load_artalk_runtime(device: str, render_res: int, asset_root: str):
    assets = ARTalkAssets.from_root(asset_root)
    runtime = ARTalkRuntime(
        ARTalkRuntimeConfig(
            assets=assets,
            device=device,
            flame_scale=1.0,
        )
    )
    mesh = RenderMesh(
        image_size=render_res,
        faces=runtime.flame_model.get_faces(),
        scale=1.0,
    )
    return runtime, mesh


@st.cache_data
def list_gagavatar_ids(tracked_path: str | None):
    if not tracked_path:
        return []
    try:
        tracked = torch.load(tracked_path, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        return []
    return ["avatar"] if "avatar" in tracked and len(tracked) == 1 else sorted(tracked)


@st.cache_data
def list_style_ids(asset_dir: str):
    return available_styles(asset_dir)


@st.cache_data
def load_style_motion(asset_dir: str, style_id: str):
    if style_id == DEFAULT_STYLE:
        return None
    style_motion = torch.load(
        Path(asset_dir) / "style_motion" / f"{style_id}.pt",
        map_location="cpu",
        weights_only=True,
    )
    if tuple(style_motion.shape) != (50, 106):
        raise ValueError(f"Invalid style motion shape: {tuple(style_motion.shape)}")
    return style_motion


class StreamingGAGAvatarAdapter:
    """Adapter from packaged ``gagavatar.runtime`` to ARTalk's renderer API."""

    def __init__(self, runtime):
        self.runtime = runtime

    def set_avatar_id(self, avatar_id: str):
        self.runtime.set_avatar_id(avatar_id)

    def build_forward_batch(self, motion_code: torch.Tensor, _flame_model=None):
        return self.runtime.build_forward_batch(motion_code)

    def forward_expression(self, batch: dict):
        return self.runtime.render_rgb_batch(batch)


@st.cache_resource
def load_gagavatar(
    device: str,
    model_path: str,
    tracked_path: str | None,
    flame_model_path: str | None,
):
    from gagavatar.runtime import GAGAvatarRuntime, GAGAvatarRuntimeConfig

    runtime = GAGAvatarRuntime(
        GAGAvatarRuntimeConfig(
            model_path=model_path,
            tracked_path=tracked_path,
            flame_model_path=flame_model_path,
            device=device,
        )
    )
    return StreamingGAGAvatarAdapter(runtime), runtime.flame_model

