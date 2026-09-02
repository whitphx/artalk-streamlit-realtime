"""`artalk-demo doctor`: validate an installation before a demo.

Every historical silent-failure class gets an explicit check: missing or
wrong-revision packages, CUDA extensions built without the local GPU's
architecture (which fail with garbage numbers, not clean errors), incomplete
asset trees, and the Streamlit pin the realtime monkey-patches depend on.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_MODULES = [
    "numpy",
    "torch",
    "torchaudio",
    "transformers",
    "av",
    "openai",
    "streamlit",
    "streamlit_webrtc",
    "pytorch3d",
    "diff_gaussian_rasterization_32d",
    "artalk",
    "gagavatar",
]
OPTIONAL_MODULES = {
    "aiohttp": "personaplex",
    "sphn": "personaplex",
    "wandb": "fallingwater",
}


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def result(self, ok: bool | None, label: str, detail: str = "") -> None:
        if ok is None:
            mark = "SKIP"
        elif ok:
            mark = "ok"
        else:
            mark = "FAIL"
            self.failures += 1
        print(f"[{mark:>4}] {label}" + (f": {detail}" if detail else ""))


def _module_version(name: str) -> str:
    try:
        return importlib.metadata.version(name.replace("_", "-"))
    except importlib.metadata.PackageNotFoundError:
        module = sys.modules.get(name)
        return getattr(module, "__version__", "?")


def check_imports(report: Report) -> None:
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            report.result(False, f"import {name}", repr(exc))
        else:
            report.result(True, f"import {name}", _module_version(name))
    for name, extra in OPTIONAL_MODULES.items():
        try:
            importlib.import_module(name)
        except Exception:
            report.result(None, f"import {name}", f"[{extra}] extra not installed")
        else:
            report.result(True, f"import {name}", _module_version(name))


def check_cuda(report: Report):
    import torch

    if not torch.cuda.is_available():
        report.result(False, "CUDA available", "torch.cuda.is_available() is False")
        return None
    index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(index)
    name = torch.cuda.get_device_name(index)
    report.result(True, "CUDA available", f"{name}, sm_{capability[0]}{capability[1]}")
    return capability


def _extension_libraries(module_name: str) -> list[Path]:
    module = importlib.import_module(module_name)
    root = Path(next(iter(module.__path__)))
    return sorted(root.rglob("*.so"))


def check_extension_archs(report: Report, capability) -> None:
    """The rasterizer and pytorch3d must embed cubins for the local GPU (or
    PTX at or below it); a miss produces corrupt output rather than a clean
    error (docs/realtime-performance-notes.md)."""
    cuobjdump = shutil.which("cuobjdump") or shutil.which(
        "cuobjdump", path=str(Path(sys.executable).parent)
    )
    for module_name in ("diff_gaussian_rasterization_32d", "pytorch3d"):
        try:
            libraries = _extension_libraries(module_name)
        except Exception as exc:
            report.result(False, f"{module_name} extension", repr(exc))
            continue
        cuda_libraries = []
        archs: set[str] = set()
        ptx: set[str] = set()
        for library in libraries:
            if cuobjdump is None:
                continue
            listing = subprocess.run(
                [cuobjdump, "--list-elf", "--list-ptx", str(library)],
                capture_output=True,
                text=True,
            ).stdout
            elf = set(re.findall(r"\bsm_(\d+)\b", listing))
            ptx_here = set(re.findall(r"\bcompute_(\d+)\b", listing))
            if elf or ptx_here:
                cuda_libraries.append(library.name)
                archs |= elf
                ptx |= ptx_here
        if cuobjdump is None:
            report.result(None, f"{module_name} arch audit", "cuobjdump not found")
            continue
        if not cuda_libraries:
            report.result(None, f"{module_name} arch audit", "no CUDA cubins found")
            continue
        if capability is None:
            report.result(None, f"{module_name} arch audit", "no CUDA device")
            continue
        local = capability[0] * 10 + capability[1]
        covered = any(int(a) == local for a in archs) or any(
            int(p) <= local for p in ptx
        )
        detail = f"sm={sorted(archs)} ptx={sorted(ptx)} local=sm_{local}"
        report.result(covered, f"{module_name} arch audit", detail)


def check_assets(report: Report) -> None:
    from artalk.assets import ARTalkAssets

    from .assets import gagavatar_assets_in_artalk_tree

    asset_dir = os.environ.get("ARTALK_ASSET_DIR")
    try:
        if asset_dir:
            artalk_assets = ARTalkAssets.from_root(asset_dir)
        else:
            artalk_assets = ARTalkAssets.from_pyproject()
        artalk_assets.validate(
            audio_encoder=os.environ.get("ARTALK_AUDIO_ENCODER", "wav2vec")
        )
    except Exception as exc:
        report.result(False, "ARTalk assets", str(exc))
        return
    report.result(True, "ARTalk assets", str(artalk_assets.root))
    try:
        gagavatar_assets = gagavatar_assets_in_artalk_tree(artalk_assets)
        gagavatar_assets.validate(require_tracked=True)
    except Exception as exc:
        report.result(False, "GAGAvatar assets", str(exc))
        return
    report.result(True, "GAGAvatar assets", str(gagavatar_assets.flame_model_path))


def check_streamlit_pins(report: Report) -> None:
    """The realtime path depends on two version-fragile Streamlit hacks."""
    config = REPO_ROOT / ".streamlit" / "config.toml"
    text = config.read_text() if config.exists() else ""
    ok = re.search(r"postScriptGC\s*=\s*false", text) is not None
    report.result(
        ok,
        "Streamlit postScriptGC disabled",
        str(config) if ok else "runner.postScriptGC=false missing (~150 ms GC pauses)",
    )
    from .streamlit_patches import disable_streamlit_source_watcher  # noqa: F401

    report.result(True, "source-watcher patch importable")


def check_render(report: Report) -> None:
    with tempfile.TemporaryDirectory() as output_dir:
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "benchmark_pipeline.py"),
                "--device",
                "cuda",
                "--seconds",
                "4",
                "--configs",
                "mesh:256:4",
                "--output-dir",
                output_dir,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    detail = completed.stdout.strip().splitlines()[-1:] or completed.stderr.strip().splitlines()[-1:]
    report.result(
        completed.returncode == 0,
        "headless render smoke",
        detail[0] if detail else "",
    )


def run(args: argparse.Namespace) -> int:
    report = Report()
    print(f"interpreter: {sys.executable}")
    check_imports(report)
    capability = None
    if "torch" in sys.modules:
        capability = check_cuda(report)
    check_extension_archs(report, capability)
    check_assets(report)
    check_streamlit_pins(report)
    if args.render:
        check_render(report)
    print(f"{report.failures} failure(s)")
    return 1 if report.failures else 0
