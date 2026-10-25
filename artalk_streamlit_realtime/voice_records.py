"""Opt-in recording of a media session's microphone and reply audio."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, NamedTuple

import av
import numpy as np

logger = logging.getLogger(__name__)

VOICE_DIR = "voice"
VOICE_SCHEMA_VERSION = 1
MIC_SAMPLE_RATE = 16000
# Speech at this rate is close to transparent with Opus; it is 180 kB per
# minute and track, against roughly a megabyte for FLAC at 16 kHz.
OPUS_BIT_RATE = 24000
# The writer thread's backlog, in chunks: the microphone arrives as 20 ms
# frames, so this holds at least ten seconds of it.
QUEUE_MAX_CHUNKS = 500
# A chunk that arrives later than the end of the audio already written,
# by more than this, is preceded by silence. Below it, network and
# scheduling jitter would otherwise fill the track with slivers of silence.
GAP_TOLERANCE_S = 0.2

Track = Literal["mic", "reply"]


class VoiceCodec(NamedTuple):
    name: str
    container: str
    extension: str


@functools.cache
def voice_codec() -> VoiceCodec:
    try:
        av.codec.Codec("libopus", "w")
    except Exception:
        logger.warning(
            "[voice-records] this PyAV build has no libopus encoder; voice "
            "recordings fall back to FLAC, about six times larger"
        )
        return VoiceCodec("flac", "flac", "flac")
    # Ogg rather than WebM: a page written is a page readable, so a file
    # cut short by a crash still decodes up to its last page.
    return VoiceCodec("libopus", "ogg", "opus")


def subject_key(subject_id: str) -> str:
    """The folder name for one subject's recordings: stable across sessions
    of a logged-in visitor, so that their deletion reaches every recording,
    without putting their account id in a path."""
    digest = hashlib.sha256(f"artalk-voice-subject\0{subject_id}".encode("utf-8"))
    return digest.hexdigest()[:32]


class _TrackWriter:
    def __init__(self, path: Path, rate: int, codec: VoiceCodec) -> None:
        self.path = path
        # Written under a name outside the upload patterns, so a scheduled
        # commit never takes a file that is still growing.
        self.part = path.with_name(path.name + ".part")
        self.rate = rate
        self.codec = codec
        self.first_at: float | None = None
        self.samples = 0
        self.silence_samples = 0
        self._container = None
        self._stream = None

    def write(self, samples: np.ndarray, start_at: float) -> None:
        if self._container is None:
            self.part.parent.mkdir(parents=True, exist_ok=True)
            self._container = av.open(str(self.part), mode="w", format=self.codec.container)
            self._stream = self._container.add_stream(
                self.codec.name, rate=self.rate, layout="mono"
            )
            self._stream.codec_context.format = "s16"
            if self.codec.name == "libopus":
                self._stream.bit_rate = OPUS_BIT_RATE
            self.first_at = start_at
        gap = round((start_at - self.first_at) * self.rate) - self.samples
        if gap > GAP_TOLERANCE_S * self.rate:
            self.silence_samples += gap
            silence = np.zeros(self.rate, dtype=np.int16)
            while gap > 0:
                self._encode(silence[: min(gap, self.rate)])
                gap -= self.rate
        self._encode(samples)

    def _encode(self, samples: np.ndarray) -> None:
        frame = av.AudioFrame.from_ndarray(samples[np.newaxis, :], format="s16", layout="mono")
        frame.sample_rate = self.rate
        frame.pts = self.samples
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self.samples += samples.size

    def finish(self, origin: float) -> dict | None:
        if self._container is None:
            return None
        for packet in self._stream.encode(None):
            self._container.mux(packet)
        self._container.close()
        os.replace(self.part, self.path)
        return {
            "file": self.path.name,
            "codec": self.codec.name,
            "sample_rate": self.rate,
            "start_offset_s": round(self.first_at - origin, 3),
            "duration_s": round(self.samples / self.rate, 3),
            "silence_inserted_s": round(self.silence_samples / self.rate, 3),
        }

    def abort(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                logger.debug("[voice-records] closing an aborted track", exc_info=True)
        self.part.unlink(missing_ok=True)


class VoiceSession:
    """Records one media session's tracks into ``directory``.

    The push methods are called from WebRTC and backend threads and never
    block: they hand chunks to a bounded queue that a thread of its own
    encodes, and count what does not fit. Placing each chunk by its arrival
    time keeps a dropped chunk from shifting what follows it.
    """

    def __init__(
        self,
        directory: Path,
        media_session: str,
        *,
        record_reply: bool,
        metadata: dict,
    ) -> None:
        self.directory = directory
        self.media_session = media_session
        self.record_reply = record_reply
        self.metadata = metadata
        self.started = time.monotonic()
        self.started_utc = datetime.now(timezone.utc)
        self.dropped = {"mic": 0, "reply": 0}
        self.end_reason: str | None = None
        self.ended: float | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX_CHUNKS)
        self._closing = threading.Event()
        self._aborting = threading.Event()
        self._mic_resampler = av.AudioResampler(format="s16", layout="mono", rate=MIC_SAMPLE_RATE)
        codec = voice_codec()
        self._tracks: dict[Track, _TrackWriter] = {
            "mic": _TrackWriter(
                directory / f"{media_session}.mic.{codec.extension}", MIC_SAMPLE_RATE, codec
            ),
        }
        self._codec = codec
        self._thread = threading.Thread(
            target=self._run, name=f"voice-records-{media_session[:8]}", daemon=True
        )
        self._thread.start()

    def push_mic(self, frame: av.AudioFrame) -> None:
        if self._closing.is_set():
            return
        # A copy, since the frame's buffer belongs to whoever decoded it.
        chunk = (frame.to_ndarray().copy(), frame.format.name, frame.layout.name, frame.sample_rate)
        self._put("mic", chunk)

    def push_reply(self, samples: np.ndarray, sample_rate: int) -> None:
        if not self.record_reply or self._closing.is_set():
            return
        self._put("reply", (samples, sample_rate))

    def _put(self, track: Track, chunk) -> None:
        try:
            self._queue.put_nowait((track, time.monotonic(), chunk))
        except queue.Full:
            # Each track is pushed from a single thread, so its counter has
            # a single writer.
            self.dropped[track] += 1

    def close(self, reason: str) -> None:
        """Stops taking audio and finalizes the files on the writer thread."""
        if self._closing.is_set():
            return
        self.end_reason = reason
        self.ended = time.monotonic()
        self._closing.set()

    def join(self) -> None:
        self._thread.join()

    def _run(self) -> None:
        try:
            while not self._aborting.is_set():
                try:
                    track, arrived, chunk = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._closing.is_set():
                        break
                    continue
                self._encode(track, arrived, chunk)
        except Exception:
            logger.warning("[voice-records] recording failed; discarding it", exc_info=True)
            self._aborting.set()
        finally:
            try:
                if self._aborting.is_set():
                    for writer in self._tracks.values():
                        writer.abort()
                else:
                    self._finalize()
            except Exception:
                logger.warning("[voice-records] could not finalize a recording", exc_info=True)
                for writer in self._tracks.values():
                    writer.abort()

    def _encode(self, track: Track, arrived: float, chunk) -> None:
        if track == "mic":
            array, fmt, layout, rate = chunk
            frame = av.AudioFrame.from_ndarray(array, format=fmt, layout=layout)
            frame.sample_rate = rate
            for resampled in self._mic_resampler.resample(frame):
                samples = resampled.to_ndarray()[0]
                # A microphone frame arrives once it has been captured, so
                # it began its own length earlier.
                self._tracks["mic"].write(samples, arrived - samples.size / MIC_SAMPLE_RATE)
            return
        samples, rate = chunk
        writer = self._tracks.get("reply")
        if writer is None:
            writer = self._tracks["reply"] = _TrackWriter(
                self.directory / f"{self.media_session}.reply.{self._codec.extension}",
                rate,
                self._codec,
            )
        elif rate != writer.rate:
            raise ValueError(f"reply rate changed from {writer.rate} to {rate} Hz")
        # Reply audio arrives faster than it plays, so a burst is laid end
        # to end rather than at its arrival times.
        writer.write(samples, arrived)

    def _finalize(self) -> None:
        tracks = {name: writer.finish(self.started) for name, writer in self._tracks.items()}
        tracks = {name: meta for name, meta in tracks.items() if meta is not None}
        if not tracks:
            return
        for name, meta in tracks.items():
            meta["dropped_chunks"] = self.dropped[name]
        ended = self.ended if self.ended is not None else time.monotonic()
        sidecar = {
            "schema_version": VOICE_SCHEMA_VERSION,
            "media_session": self.media_session,
            "started_utc": self.started_utc.isoformat(timespec="milliseconds"),
            "duration_s": round(ended - self.started, 3),
            "end_reason": self.end_reason,
            **self.metadata,
            "tracks": tracks,
        }
        path = self.directory / f"{self.media_session}.json"
        part = path.with_name(path.name + ".part")
        part.write_text(json.dumps(sidecar, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(part, path)
