#!/usr/bin/env python
"""Track one face image into a GAGAvatar tracked-avatar entry.

Runs under the GAGAvatar_track environment (``GAGAVATAR_TRACK_PYTHON`` /
``GAGAVATAR_TRACK_DIR``); the app invokes it as a subprocess because the
tracker's dependency stack (torch build, onnx tooling) conflicts with the
realtime runtime environment.
"""

import argparse
import os
import sys
from pathlib import Path

# The tracker's dependencies use torch.compile, which fails on GPUs the
# triton backend does not support and would never amortize in a one-shot
# subprocess anyway.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "--track-dir",
        required=True,
        type=str,
        help="GAGAvatar_track checkout directory (provides the engines package).",
    )
    parser.add_argument("--focal-length", type=float, default=12.0)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--no-matting", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, args.track_dir)
    import torch
    import torchvision
    from engines import CoreEngine

    image = (
        torchvision.io.read_image(
            args.image, mode=torchvision.io.image.ImageReadMode.RGB
        )
        .to(args.device)
        .float()
    )
    engine = CoreEngine(focal_length=args.focal_length, device=args.device)
    key = "avatar"
    try:
        results = engine.track_image([image], [key], if_matting=not args.no_matting)
    except Exception as exc:
        print(f"tracking failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if not results or key not in results:
        print("no face detected in the uploaded image", file=sys.stderr)
        return 2

    # Save tensors only, so consumers can load with weights_only=True.
    entry = {
        name: torch.as_tensor(value) if not isinstance(value, torch.Tensor) else value
        for name, value in results[key].items()
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(output.name + ".tmp")
    torch.save(entry, tmp_path)
    os.replace(tmp_path, output)
    print(f"tracked entry written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
