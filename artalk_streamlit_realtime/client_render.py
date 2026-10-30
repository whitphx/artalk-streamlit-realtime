"""Client-side rendering: the browser draws the avatar, the server sends motion.

With ``--client-render`` the pipeline stops before GAGAvatar's network. For
every frame it computes the FLAME head vertices and the camera exactly as the
GAGAvatar renderer would (forehead smoothing included) and sends them over a
WebRTC data channel on the connection that carries the audio. The browser
splats the avatar's Gaussians at those vertices and runs a small network
distilled from GAGAvatar on WebGPU (``client_render_js/``).

Each message on the ``motion`` channel is one frame, little-endian:

    offset  type      field
    0       4 bytes   magic b"ARV1"
    4       u32       frame index (the pipeline's 25 fps frame counter)
    8       u32       avatar generation (bumped on every avatar switch)
    12      f32       jaw opening (motion code channel 103)
    16      f64       media time of the frame's first audio sample, in seconds
                      on the outbound audio track's clock
    24      f64       server wall clock when the message was sent, ms since
                      the Unix epoch
    32      f32       how long before that audio leaves the server the message
                      was sent, ms
    36      f32       RMS of the audio frame emitted with the message
    40      f64       media time where that audio frame starts, in seconds
    48      u32       RTP timestamp under which the browser plays the frame's
                      first audio sample
    52      u32       1 when the RTP timestamp is known, else 0
    56      12 x f32  camera, the 3x4 transform GAGAvatar builds its view from
    104     int16     head-vertex offsets (x, y, z per vertex) from the
                      avatar bundle's neutral vertices, in QUANT_STEP units

The audio RMS and the jaw opening let the browser estimate the audio-to-face
offset it actually shows against the one the server intended.

The avatar bundles, networks and ONNX Runtime files are read from
``--client-render-dir`` (layout in ``ClientRenderAssets``) and served to the
browser through Streamlit's component file route. They are not part of this
repository.
"""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components
import torch
from artalk.metrics import active_pipeline_metrics, current_pipeline_metrics
from artalk.realtime_pipeline import (
    AUDIO_OUT_PTIME,
    AUDIO_SAMPLES_PER_VIDEO_FRAME,
    ARTalkPipeline,
    with_pipeline_metrics,
)

CHANNEL_LABEL = "motion"
# A frame spans about 25 packets, so with no retransmissions one lost packet
# loses the whole frame. A retransmission is worth it while the frame can
# still arrive before the browser shows it, and frames arrive a few hundred
# ms early; a limit on retransmission count would not bound that time.
# aiortc's lifetime starts when the frame enters SCTP, and its retransmission
# timer waits at least 1 s, so only fast retransmissions ever happen and an
# expired frame is abandoned rather than resent.
CHANNEL_OPTIONS = {"ordered": False, "maxPacketLifeTime": 120}
# 2^-17 units per step covers +-0.25 units, several times the largest offset
# seen in recorded conversations, at about 1/400 px of a 512 px frame.
QUANT_STEP = 2.0**-17
HEAD_VERTICES = 5023
HEADER = struct.Struct("<4sIIfddffdII")
CAMERA_OFFSET = HEADER.size
OFFSETS_OFFSET = CAMERA_OFFSET + 12 * 4
RECORD_BYTES = OFFSETS_OFFSET + HEAD_VERTICES * 3 * 2
# The frame index and the timing fields are only known when the frame is
# sent; the worker fills in the rest.
STAMP_INDEX = struct.Struct("<I")
STAMP_TIMING = struct.Struct("<ddffdII")
# The send time and the lead, rewritten when a held frame goes out later.
STAMP_SENT = struct.Struct("<df")
JS_DIR = Path(__file__).with_name("client_render_js")
ORT_BUNDLE = "ort.webgpu.bundle.min.mjs"
ORT_CDN_URL = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/{ORT_BUNDLE}"
ORT_SHA256 = "0730816731ce2cdfb06f273da065d4a1c7091eb9d1756a83b2916c8a89d015e2"
ORT_PATCHED_SHA256 = "5557036315cfbc7facf8a4bcbcff03703e0f6a6a54f7a172e1e29d14cce3223d"
# ONNX Runtime Web 1.30 binds a graph-captured session's pre-allocated
# outputs on the first run only:
# https://github.com/microsoft/onnxruntime/blob/v1.30.0/js/web/lib/wasm/wasm-core-impl.ts#L781-L834
# Later runs recognise them by comparing the returned value with the handle
# made for that run, which never matches:
# https://github.com/microsoft/onnxruntime/blob/v1.30.0/js/web/lib/wasm/wasm-core-impl.ts#L876
# The output then takes the path for an output ORT allocated itself, which
# looks its GPU buffer up through a handle freed after the first run. That
# throws, or works when the allocator hands the same address back. The patch
# treats every gpu-buffer output of a captured session as pre-allocated.
# The names are the minifier's, so it only applies to the file ORT_SHA256
# identifies.
ORT_PATCH_SITE = b"if(ae===O[R]||oe.includes(O[R])){"
ORT_PATCH = b'if(ae===O[R]||oe.includes(O[R])||A&&i[R]?.[3]==="gpu-buffer"){'
NETWORKS = {
    "trained_lowres16": "Network, 256 px core (lowres16)",
    "trained_plain16": "Network, 512 px (plain16)",
    "none": "None: data path only (jaw bar)",
}
# Which estimate of when a frame's audio plays schedules the face.
SYNC_SCHEMES = {
    "arrival": "Arrival + jitter buffer",
    "rtp": "RTP playout clock",
}
# streamlit-webrtc imports the script from a data: URL, against which
# relative imports do not resolve; the modules are served next to the
# component's own iframe instead.
FRONTEND_SCRIPT = """
export default async function (context) {
  const base = new URL(`../${context.data.dirs.js}/`, location.href);
  const { start } = await import(new URL("session.js", base).href);
  return start(context);
}
"""


def _read_gsb_array(path: Path, name: str) -> np.ndarray:
    """One array of a GSB1 avatar bundle: 'GSB1', u32 header length, 8 zero
    bytes, a JSON header listing the arrays, then the array data."""
    with path.open("rb") as f:
        magic, header_len = struct.unpack("<4sI8x", f.read(16))
        if magic != b"GSB1":
            raise ValueError(f"{path} is not a GSB1 bundle")
        header = json.loads(f.read(header_len))
        entry = next((a for a in header["arrays"] if a["name"] == name), None)
        if entry is None:
            raise KeyError(f"{path} has no array {name!r}")
        f.seek(16 + header_len + entry["offset"])
        data = f.read(entry["nbytes"])
    return np.frombuffer(data, dtype=entry["dtype"]).reshape(entry["shape"])


class ClientRenderAssets:
    """The directory the browser loads the renderer's data from:

    avatars/<avatar id>/avatar.gsb   Gaussians and neutral head vertices
    models.json, models/             the networks as ONNX, fp16 and fp32
    ort/                             optional local ONNX Runtime Web files,
                                     the bundle as install_patched_ort leaves it
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not (self.root / "models.json").is_file():
            raise FileNotFoundError(f"{self.root / 'models.json'} not found")
        self.models = json.loads((self.root / "models.json").read_text())["models"]
        ort = self.root / "ort" / ORT_BUNDLE
        self.has_local_ort = ort.is_file()
        # Graph capture is only safe on the patched copy, see ORT_PATCH.
        self.ort_patched = (
            self.has_local_ort and hashlib.sha256(ort.read_bytes()).hexdigest() == ORT_PATCHED_SHA256
        )

    def avatar_ids(self) -> list[str]:
        avatars = self.root / "avatars"
        if not avatars.is_dir():
            return []
        return sorted(p.parent.name for p in avatars.glob("*/avatar.gsb"))

    def head_ref(self, avatar_id: str) -> np.ndarray:
        ref = _read_gsb_array(self.root / "avatars" / avatar_id / "avatar.gsb", "head_ref")
        if ref.shape != (HEAD_VERTICES, 3):
            raise ValueError(f"{avatar_id}: head_ref has shape {ref.shape}")
        return ref


def install_patched_ort(root: str | Path) -> Path:
    """Writes the patched ONNX Runtime Web bundle to ``<root>/ort/``, from
    the copy already there or else from jsDelivr."""
    path = Path(root) / "ort" / ORT_BUNDLE
    bundle = path.read_bytes() if path.is_file() else urllib.request.urlopen(ORT_CDN_URL).read()
    digest = hashlib.sha256(bundle).hexdigest()
    if digest == ORT_PATCHED_SHA256:
        return path
    if digest != ORT_SHA256:
        raise ValueError(f"{path} is not ONNX Runtime Web 1.30.0's {ORT_BUNDLE}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bundle.replace(ORT_PATCH_SITE, ORT_PATCH))
    return path


def serve_dirs(assets: ClientRenderAssets) -> dict[str, str]:
    """Registers the asset and script directories with Streamlit's component
    file route; returns their names, which the browser turns into URLs."""
    # Declared components that are never rendered: the route is all they add.
    data = components.declare_component("data", path=str(assets.root))
    js = components.declare_component("js", path=str(JS_DIR))
    return {"data": data.name, "js": js.name}


class ClientRenderAdapter:
    """Stands in for StreamingGAGAvatarAdapter. ``build_forward_batch`` is the
    GAGAvatar runtime's, so vertices and camera match the server renderer;
    ``forward_expression`` packs them into motion messages instead of running
    the network.
    """

    def __init__(self, runtime, assets: ClientRenderAssets):
        # Per-view copies as in StreamingGAGAvatarAdapter: the runtime keeps
        # the selected avatar and the forehead smoothing state on itself.
        self.runtime = copy.copy(runtime)
        self.runtime.model = copy.copy(runtime.model)
        self.assets = assets
        self.avatar_id: str | None = None
        self.generation = 0
        self._head_ref: torch.Tensor | None = None
        # set_avatar_id runs on the script thread while the worker builds.
        self._lock = threading.Lock()

    def set_avatar_id(self, avatar_id: str):
        head_ref = torch.from_numpy(self.assets.head_ref(avatar_id).copy())
        with self._lock:
            self.runtime.set_avatar_id(avatar_id)
            self._head_ref = head_ref.to(self.runtime.device)
            self.avatar_id = avatar_id
            self.generation += 1

    def build_forward_batch(self, motion_code: torch.Tensor, _flame_model=None):
        with self._lock:
            batch = self.runtime.build_forward_batch(motion_code)
            return {
                "points": batch["t_points"],
                "camera": batch["t_transform"][:, :3, :4],
                "jaw": motion_code[:, 103],
                "head_ref": self._head_ref,
                "generation": self.generation,
            }

    def forward_expression(self, batch: dict) -> torch.Tensor:
        n = batch["points"].shape[0]
        offsets = (batch["points"] - batch["head_ref"]) / QUANT_STEP
        clamped = int((offsets.abs() > 32767).sum())
        metrics = active_pipeline_metrics()
        if clamped and metrics is not None:
            metrics.inc("client_render_clamped_offsets", clamped)
        offsets = offsets.round().clamp(-32768, 32767).to(torch.int16).cpu().numpy()
        camera = batch["camera"].float().cpu().numpy()
        jaw = batch["jaw"].float().cpu().numpy()
        records = np.zeros((n, RECORD_BYTES), dtype=np.uint8)
        for i in range(n):
            HEADER.pack_into(records[i], 0, b"ARV1", 0, batch["generation"], jaw[i], 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
        records[:, CAMERA_OFFSET:OFFSETS_OFFSET] = camera.astype("<f4").reshape(n, -1).view(np.uint8)
        records[:, OFFSETS_OFFSET:] = offsets.astype("<i2").reshape(n, -1).view(np.uint8)
        # The pipeline treats a uint8 batch as finished frames; each "frame"
        # here is one message.
        return torch.from_numpy(records)


class ClientRenderPipeline(ARTalkPipeline):
    """ARTalkPipeline whose frames are motion messages for the browser.

    The pipeline's media clock decides which frame belongs to the audio being
    emitted, as it does for the outbound video track. Here the audio source
    callback sends that frame on the data channel instead, so a message
    leaves with the audio it describes. There is no video track.

    A frame only goes to the channel when nothing is waiting there for SCTP.
    While SCTP's congestion window is full, a frame waits on the pipeline
    instead, and the next frame replaces it: a frame queued behind older ones
    would arrive after its audio, and would hold back the fresh frames
    behind it too.
    """

    def __init__(self, **kwargs):
        # Frames must leave the renderer as the adapter's raw bytes.
        kwargs["renderer_output_uint8"] = False
        self.channel_source = lambda: None
        # (pts, time_base) -> the outbound audio's RTP timestamp at pts, or None.
        self.rtp_timestamp = lambda pts, time_base: None
        # The frame waiting to be sent: the message, the lead stamped into
        # it and time.perf_counter() when it was stamped.
        self._held_frame: tuple[bytearray, float, float] | None = None
        self._warmed_channel = None
        super().__init__(**kwargs)

    @property
    def client_renderer(self) -> ClientRenderAdapter:
        return self._renderer._gaga

    def set_avatar(self, avatar_id: str) -> None:
        """Switches the avatar without rebuilding the pipeline, so the
        conversation and the WebRTC session carry on. Frames already queued
        keep the previous generation and the browser skips them once it has
        loaded the new avatar."""
        if avatar_id != self.client_renderer.avatar_id:
            self.client_renderer.set_avatar_id(avatar_id)

    @with_pipeline_metrics
    def audio_source_callback(self, pts, time_base):
        frame = super().audio_source_callback(pts, time_base)
        metrics = current_pipeline_metrics()
        channel = self.channel_source()
        if channel is not None and channel is not self._warmed_channel and channel.ready_state == "open":
            # Until SCTP has measured a round trip, its retransmission timer
            # waits 3 s rather than at least 1 s. When the last packets of a
            # frame are lost, no later packets report the loss, and until
            # that timer fires they fill the congestion window and block the
            # frames behind them. The playback prebuffer leaves time for a
            # short message to take a sample before the first frame.
            channel.send("warm-up")
            self._warmed_channel = channel
        target_index = self._media_clock_frame_index()
        rendered, frame_index = self._pop_video_frame_for_media_clock(target_index)
        if rendered is not None:
            self._last_video_frame_index_served = frame_index
            metrics.inc("video_frames_served")
            metrics.set("last_video_frame_index_served", frame_index)
            self._record_served_video_lag(target_index, frame_index)
            self._record_served_frame_latency(rendered)
            if channel is None:
                metrics.inc("client_render_frames_without_channel")
            else:
                if self._held_frame is not None:
                    metrics.inc("client_render_frames_dropped")
                self._held_frame = self._stamp_frame(rendered, frame_index, frame, pts, time_base)
        # Checked on every callback, not only with a new frame, so a held
        # frame leaves within one audio ptime of the channel draining.
        if channel is not None and self._held_frame is not None and channel.buffered_amount == 0:
            message, lead_ms, stamped_at = self._held_frame
            self._held_frame = None
            waited_ms = (time.perf_counter() - stamped_at) * 1000.0
            STAMP_SENT.pack_into(message, 24, time.time() * 1000.0, lead_ms - waited_ms)
            if channel.send(message):
                metrics.inc("client_render_frames_sent")
                metrics.inc("client_render_bytes_sent", len(message))
            else:
                metrics.inc("client_render_frames_dropped")
        return frame

    def _stamp_frame(self, rendered, frame_index, frame, pts, time_base):
        with self._audio_out_lock:
            synced = self._synced_audio_samples_served
        # Track position of the frame's first sample. The synced count ends
        # where this call's audio ends; after a call that emitted silence it
        # resumes with the next call's audio, which starts at the same place.
        start = pts + frame.samples + frame_index * AUDIO_SAMPLES_PER_VIDEO_FRAME - synced
        start_s = float(start * time_base)
        # streamlit-webrtc's audio track calls back one ptime before it sends
        # the frame with this pts.
        lead_ms = (float((start - pts) * time_base) + AUDIO_OUT_PTIME) * 1000.0
        samples = frame.to_ndarray().astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples)))
        rtp = self.rtp_timestamp(int(start), time_base)
        message = bytearray(rendered.image.tobytes())
        STAMP_INDEX.pack_into(message, 4, frame_index)
        STAMP_TIMING.pack_into(
            message,
            16,
            start_s,
            time.time() * 1000.0,
            lead_ms,
            rms,
            float(pts * time_base),
            rtp or 0,
            rtp is not None,
        )
        return message, lead_ms, time.perf_counter()
