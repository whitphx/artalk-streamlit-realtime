"""`artalk-demo assets`: populate and verify the full asset tree.

Wraps the package downloaders (which verify sizes and SHA-256 against their
manifests) and adds the pieces they do not cover: the license-gated FLAME
file, the optional Fallingwater checkpoint (private GitHub release), and the
MOSS audio tokenizer prefetch (7 GB from Hugging Face that otherwise
downloads silently on first use, looking like a hang).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FLAME_INSTRUCTION = """\
FLAME_with_eye.pt is license-gated and never downloaded automatically.
Obtain it under the FLAME license (https://flame.is.tue.mpg.de/) and place it
at: {path}"""

MOSS_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

# The retrained 1 s model (docs/hf-spaces-plan.md, phase 1b). Its repo is
# private until publication is cleared, so the files are read with the Hub
# token and pinned here by content rather than by manifest.
ARTALK1S_FILES = {
    "ARTalk1s_wav2vec.pt": "d69da34d813a49ba87d05c193fe378639e5f59641a40a2bf7c4196d9a7a3b2af",
    "metadata_stats.json": "b1e7c7b8dee5031cb5935d10f572b808c0ba60416a33f5f439c8d45e7b92a11e",
}


def _run_downloader(module: str, *args: str) -> bool:
    completed = subprocess.run(
        [sys.executable, "-m", module, *args], cwd=REPO_ROOT
    )
    return completed.returncode == 0


def _fetch_fallingwater_checkpoint(dest_dir: Path) -> bool:
    dest = dest_dir / "iter_75000.pt"
    if dest.exists():
        print(f"fallingwater checkpoint present: {dest}")
        return True
    gh = shutil.which("gh")
    if gh is None:
        print(
            "gh CLI not found; download the Fallingwater checkpoint manually:\n"
            "  gh release download checkpoints --repo xg-chu/Fallingwater "
            f"--pattern iter_75000.pt --dir {dest_dir}"
        )
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            gh,
            "release",
            "download",
            "checkpoints",
            "--repo",
            "xg-chu/Fallingwater",
            "--pattern",
            "iter_75000.pt",
            "--dir",
            str(dest_dir),
        ]
    )
    if completed.returncode != 0:
        print("Fallingwater checkpoint download failed (release is on a private repo).")
        return False
    return True


def _fetch_gated_flame(root: Path, repo_id: str) -> bool:
    """FLAME cannot be redistributed, so a deployment keeps its copy in a
    private Hub repo of its own and completes the tree from there."""
    dest = root / "FLAME_with_eye.pt"
    if dest.exists():
        return True
    from artalk.assets import iter_manifest_assets, verify_asset
    from huggingface_hub import hf_hub_download

    print(f"download: {repo_id}/{dest.name} -> {dest}")
    try:
        hf_hub_download(repo_id, dest.name, local_dir=root)
    except Exception as exc:
        print(f"gated asset download failed ({type(exc).__name__}): {exc}")
        return False
    asset = next(
        a for a in iter_manifest_assets(include_optional=True) if a["path"] == dest.name
    )
    try:
        verify_asset(dest, asset)
    except Exception:
        dest.unlink()
        raise
    return True


def _fetch_artalk1s(root: Path, repo_id: str) -> bool:
    from artalk.assets import sha256_file
    from huggingface_hub import hf_hub_download

    dest_dir = root / "ARTalk1s"
    ok = True
    for name, expected in ARTALK1S_FILES.items():
        dest = dest_dir / name
        if dest.exists():
            continue
        print(f"download: {repo_id}/{name} -> {dest}")
        try:
            hf_hub_download(repo_id, name, local_dir=dest_dir)
        except Exception as exc:
            print(f"ARTalk 1s download failed ({type(exc).__name__}): {exc}")
            ok = False
            continue
        if sha256_file(dest) != expected:
            dest.unlink()
            print(f"ARTalk 1s download failed: {name} does not match its pinned hash")
            ok = False
    return ok


def _prefetch_moss_tokenizer() -> bool:
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    from huggingface_hub import snapshot_download

    print(f"prefetching {MOSS_REPO} into {os.environ['HF_HOME']} (~7 GB)")
    snapshot_download(MOSS_REPO)
    return True


def run(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    ok = True

    if not args.check_only:
        ok &= _run_downloader(
            "artalk.assets", "download", "--root", str(root), "--include-optional"
        )
        ok &= _run_downloader(
            "gagavatar.assets", "download", "--root", str(root / "GAGAvatar")
        )
        if args.fallingwater:
            ok &= _fetch_fallingwater_checkpoint(root / "Fallingwater")
            ok &= _prefetch_moss_tokenizer()
        if args.gated_repo:
            ok &= _fetch_gated_flame(root, args.gated_repo)
        if args.artalk1s_repo:
            ok &= _fetch_artalk1s(root, args.artalk1s_repo)

    flame = root / "FLAME_with_eye.pt"
    if not flame.exists():
        print(FLAME_INSTRUCTION.format(path=flame))
        ok = False
    if not any((root / "style_motion").glob("*.pt")):
        print(f"no style motions under {root / 'style_motion'}")
        ok = False

    from artalk.assets import ARTalkAssets

    from .assets import gagavatar_assets_in_artalk_tree

    try:
        artalk_assets = ARTalkAssets.from_root(root)
        artalk_assets.validate(
            audio_encoder=os.environ.get("ARTALK_AUDIO_ENCODER", "wav2vec")
        )
        gagavatar_assets_in_artalk_tree(artalk_assets).validate(require_tracked=True)
    except Exception as exc:
        print(f"asset validation failed: {exc}")
        return 1
    print(f"asset tree valid: {root.resolve()}")
    return 0 if ok else 1
