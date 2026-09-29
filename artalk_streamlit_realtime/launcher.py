"""`artalk-demo`: launch the app with a profile and measured per-GPU presets.

Precedence for every setting the launcher computes: explicit app arguments
(after `--`) win over the caller's environment, which wins over the selected
profile, which wins over the GPU preset. Presets and profiles are applied as
environment-variable defaults only, so they never override a caller decision
(`config.parse_args` reads env vars as argparse defaults).
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class LaunchConfigError(Exception):
    pass


def _load_launch_config() -> dict:
    config: dict = {}
    for name in ("launch.toml", "launch.local.toml"):
        path = REPO_ROOT / name
        if not path.exists():
            continue
        with path.open("rb") as f:
            layer = tomllib.load(f)
        config = _merge(config, layer)
    if not config:
        raise LaunchConfigError(f"no launch.toml found in {REPO_ROOT}")
    return config


def _merge(base: dict, layer: dict) -> dict:
    merged = dict(base)
    for key, value in layer.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class Gpu:
    def __init__(
        self, index: int, name: str, vram_gb: float, capability: float | None
    ) -> None:
        self.index = index
        self.name = name
        self.vram_gb = vram_gb
        self.capability = capability
        self.compute_pids: list[int] = []

    def __str__(self) -> str:
        cap = f"sm_{int(self.capability * 10)}" if self.capability else "sm_?"
        busy = f", {len(self.compute_pids)} compute proc(s)" if self.compute_pids else ""
        return f"GPU {self.index}: {self.name} ({self.vram_gb:.0f} GB, {cap}{busy})"


def _nvidia_smi(*query_args: str) -> list[list[str]]:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    try:
        out = subprocess.run(
            [smi, *query_args, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    return [
        [field.strip() for field in line.split(",")]
        for line in out.splitlines()
        if line.strip()
    ]


def _discover_gpus() -> list[Gpu]:
    gpus = []
    uuid_to_gpu: dict[str, Gpu] = {}
    for row in _nvidia_smi("--query-gpu=index,uuid,name,memory.total,compute_cap"):
        # compute_cap needs a reasonably recent driver; tolerate its absence.
        if len(row) == 4:
            index, uuid, name, mem = row
            cap = None
        elif len(row) == 5:
            index, uuid, name, mem, cap_s = row
            try:
                cap = float(cap_s)
            except ValueError:
                cap = None
        else:
            continue
        gpu = Gpu(int(index), name, float(mem) / 1024.0, cap)
        gpus.append(gpu)
        uuid_to_gpu[uuid] = gpu
    for row in _nvidia_smi("--query-compute-apps=gpu_uuid,pid"):
        if len(row) == 2 and row[0] in uuid_to_gpu:
            uuid_to_gpu[row[0]].compute_pids.append(int(row[1]))
    return gpus


def _select_gpu(gpus: list[Gpu], requested: int | None) -> Gpu | None:
    if not gpus:
        return None
    if requested is not None:
        matches = [g for g in gpus if g.index == requested]
        if not matches:
            raise LaunchConfigError(f"--gpu {requested} not found")
        return matches[0]
    # A shared GPU cost a 2.8x slowdown once (docs/meeting-notes-20260819.md);
    # prefer a fully idle card.
    idle = [g for g in gpus if not g.compute_pids]
    return idle[0] if idle else min(gpus, key=lambda g: len(g.compute_pids))


def _match_preset(presets: list[dict], gpu: Gpu) -> dict | None:
    for preset in presets:
        if "max_vram_gb" in preset and gpu.vram_gb > preset["max_vram_gb"]:
            continue
        if "min_vram_gb" in preset and gpu.vram_gb < preset["min_vram_gb"]:
            continue
        if "min_capability" in preset or "max_capability" in preset:
            if gpu.capability is None:
                continue
            if gpu.capability < preset.get("min_capability", 0.0):
                continue
            if gpu.capability > preset.get("max_capability", 99.0):
                continue
        return preset
    return None


def _bin(name: str) -> str:
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else name


def _build_command(profile: dict, app_args: list[str]) -> list[str]:
    bind = profile.get("bind", "0.0.0.0")
    port = profile.get("port", 8501)
    remote = profile.get("remote", "none")
    app = str(REPO_ROOT / "streamlit_app.py")
    if remote == "none":
        command = [
            _bin("streamlit"),
            "run",
            app,
            "--server.address",
            str(bind),
            "--server.port",
            str(port),
        ]
        if app_args:
            command += ["--", *app_args]
    else:
        command = [
            _bin("st-remote"),
            "--host",
            str(bind),
            "--port",
            str(port),
            "--provider",
            str(remote),
            app,
        ]
        if app_args:
            # st-remote's own separator, then streamlit's.
            command += ["--", "--", *app_args]
    return command


def _compute_env(profile: dict, preset: dict | None, gpu: Gpu | None) -> dict[str, str]:
    env: dict[str, str] = {
        "PYTHONNOUSERSITE": "1",
        "HF_HOME": str(REPO_ROOT / ".cache" / "huggingface"),
    }
    if preset:
        env.update({k: str(v) for k, v in preset.get("env", {}).items()})
    env.update({k: str(v) for k, v in profile.get("env", {}).items()})
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
    vendor = {
        "FALLINGWATER_DIR": REPO_ROOT / "vendor" / "Fallingwater",
        "ARTALK1S_TRAIN_CODE_DIR": REPO_ROOT / "vendor" / "ARTalk" / "train_code",
        "GAGAVATAR_TRACK_DIR": REPO_ROOT
        / "vendor"
        / "GAGAvatar"
        / "gagavatar"
        / "libs"
        / "GAGAvatar_track",
    }
    for key, path in vendor.items():
        if path.is_dir():
            env.setdefault(key, str(path))
    track_python = REPO_ROOT / ".envs" / "track" / "bin" / "python"
    if track_python.exists():
        env.setdefault("GAGAVATAR_TRACK_PYTHON", str(track_python))
    asset_dir = os.environ.get("ARTALK_ASSET_DIR")
    artalk1s = Path(asset_dir or REPO_ROOT / "assets") / "ARTalk1s" / "ARTalk1s_wav2vec.pt"
    if artalk1s.exists():
        env.setdefault("ARTALK1S_CHECKPOINT", str(artalk1s))
    if asset_dir:
        env.setdefault("GAGAVATAR_MODEL_PATH", f"{asset_dir}/GAGAvatar/GAGAvatar.pt")
        env.setdefault("GAGAVATAR_TRACKED_PATH", f"{asset_dir}/GAGAvatar/tracked.pt")
        env.setdefault("GAGAVATAR_FLAME_MODEL_PATH", f"{asset_dir}/FLAME_with_eye.pt")
    # The caller's environment wins over everything computed above.
    return {k: v for k, v in env.items() if k not in os.environ}


def _start_services(profile: dict) -> list[subprocess.Popen]:
    procs = []
    for service in profile.get("services", []):
        print(f"starting service {service['name']}: {' '.join(service['command'])}")
        procs.append(
            subprocess.Popen(
                service["command"],
                cwd=service.get("cwd"),
                env={**os.environ, **service.get("env", {})},
            )
        )
    return procs


def _stop_services(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def up(args: argparse.Namespace, app_args: list[str]) -> int:
    config = _load_launch_config()
    profiles = config.get("profiles", {})
    if args.profile not in profiles:
        raise LaunchConfigError(
            f"unknown profile {args.profile!r}; available: {', '.join(profiles)}"
        )
    profile = _merge(config.get("defaults", {}), profiles[args.profile])

    requested_gpu = args.gpu
    if requested_gpu is None and "CUDA_VISIBLE_DEVICES" in os.environ:
        # The caller already pinned the device; keep the pin (_compute_env
        # never overrides the caller's environment) but still match a preset.
        pinned = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0].strip()
        requested_gpu = int(pinned) if pinned.isdigit() else None
        if requested_gpu is None:
            gpu = None
        else:
            gpu = _select_gpu(_discover_gpus(), requested_gpu)
    else:
        gpu = _select_gpu(_discover_gpus(), requested_gpu)
    preset = None
    if gpu is not None:
        print(str(gpu))
        if gpu.compute_pids:
            print(
                "WARNING: the selected GPU has resident compute processes; "
                "sharing a GPU degrades realtime performance badly "
                "(measured 2.8x)."
            )
        preset = _match_preset(config.get("presets", []), gpu)
        print(f"preset: {preset['name'] if preset else 'none'}")

    env = _compute_env(profile, preset, gpu)
    command = _build_command(profile, app_args)

    print(f"profile: {args.profile}")
    for key, value in sorted(env.items()):
        print(f"  {key}={value}")
    print(f"  {' '.join(command)}")
    if args.dry_run:
        return 0

    os.environ.update(env)
    services = _start_services(profile)
    if services:
        # exec would orphan the services; supervise the app instead.
        app_proc = subprocess.Popen(command)
        try:
            return app_proc.wait()
        except KeyboardInterrupt:
            app_proc.send_signal(signal.SIGINT)
            return app_proc.wait()
        finally:
            _stop_services(services)
    os.execvp(command[0], command)


def main() -> int:
    parser = argparse.ArgumentParser(prog="artalk-demo")
    sub = parser.add_subparsers(dest="subcommand", required=True)
    up_parser = sub.add_parser(
        "up", help="launch the app; app args go after `--`"
    )
    up_parser.add_argument(
        "--profile",
        default=os.environ.get("ARTALK_DEMO_PROFILE", "lab"),
        help="profile from launch.toml (default: lab)",
    )
    up_parser.add_argument(
        "--gpu", type=int, default=None, help="GPU index to pin (default: idlest)"
    )
    up_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the selected GPU, preset, env, and command without launching",
    )
    assets_parser = sub.add_parser(
        "assets", help="download and verify the asset tree"
    )
    assets_parser.add_argument(
        "--root", default=os.environ.get("ARTALK_ASSET_DIR") or "assets"
    )
    assets_parser.add_argument(
        "--fallingwater",
        action="store_true",
        help="also fetch the Fallingwater checkpoint and MOSS tokenizer (~7 GB)",
    )
    assets_parser.add_argument(
        "--gated-repo",
        default=os.environ.get("ARTALK_GATED_ASSETS_REPO"),
        help=(
            "private Hub repo holding FLAME_with_eye.pt; read with the "
            "Hugging Face token (HF_TOKEN or `hf auth login`)"
        ),
    )
    assets_parser.add_argument(
        "--artalk1s-repo",
        default=os.environ.get("ARTALK1S_REPO"),
        help="Hub repo holding the retrained 1 s model (read with the Hub token)",
    )
    assets_parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the tree without downloading",
    )
    doctor_parser = sub.add_parser("doctor", help="validate the installation")
    doctor_parser.add_argument(
        "--render",
        action="store_true",
        help="also run a short headless GPU render (loads the full models)",
    )

    argv = sys.argv[1:]
    app_args: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, app_args = argv[:split], argv[split + 1 :]
    args = parser.parse_args(argv)
    try:
        if args.subcommand == "up":
            return up(args, app_args)
        if args.subcommand == "doctor":
            from . import doctor

            return doctor.run(args)
        if args.subcommand == "assets":
            from . import fetch_assets

            return fetch_assets.run(args)
    except LaunchConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
