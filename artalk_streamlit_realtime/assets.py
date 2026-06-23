"""Asset resolution glue between packaged ARTalk and GAGAvatar."""

from __future__ import annotations

from artalk.assets import ARTalkAssets
from gagavatar.assets import AssetConfigError as GAGAvatarAssetConfigError
from gagavatar.assets import GAGAvatarAssets


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
        return GAGAvatarAssets.from_artalk_assets(artalk_assets)
    try:
        return GAGAvatarAssets.from_pyproject()
    except GAGAvatarAssetConfigError:
        return GAGAvatarAssets.from_artalk_assets(artalk_assets)

