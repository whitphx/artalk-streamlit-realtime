"""Asset resolution glue between packaged ARTalk and GAGAvatar."""

from __future__ import annotations

from pathlib import Path

from artalk.assets import ARTalkAssets
from gagavatar.assets import AssetConfigError as GAGAvatarAssetConfigError
from gagavatar.assets import GAGAvatarAssets


def gagavatar_assets_in_artalk_tree(artalk_assets: ARTalkAssets) -> GAGAvatarAssets:
    """GAGAvatar assets nested inside an ARTalk asset tree, with FLAME shared
    at the tree root — the layout this app documents and its downloaders
    produce."""
    artalk_root = Path(artalk_assets.root)
    root = artalk_root / "GAGAvatar"
    return GAGAvatarAssets(
        root=root,
        model_path=root / "GAGAvatar.pt",
        tracked_path=root / "tracked.pt",
        flame_model_path=artalk_root / "FLAME_with_eye.pt",
    )


def resolve_gagavatar_assets(args, artalk_assets: ARTalkAssets) -> GAGAvatarAssets:
    has_gagavatar_override = any(
        [
            args.gagavatar_asset_dir,
            args.gagavatar_model_path,
            args.gagavatar_tracked_path,
            args.gagavatar_flame_model_path,
        ]
    )
    if has_gagavatar_override:
        return GAGAvatarAssets.resolve(
            root=args.gagavatar_asset_dir,
            model_path=args.gagavatar_model_path,
            tracked_path=args.gagavatar_tracked_path,
            flame_model_path=args.gagavatar_flame_model_path,
        )
    if args.asset_dir:
        return gagavatar_assets_in_artalk_tree(artalk_assets)
    try:
        return GAGAvatarAssets.from_pyproject()
    except GAGAvatarAssetConfigError:
        return gagavatar_assets_in_artalk_tree(artalk_assets)
