"""A pipeline that plays conversation audio back without an avatar.

Satisfies the surface the bridges call (`push_audio_frame`, `flush_output`,
`request_chunk_flush`, `metrics_snapshot`) and the outbound-track surface
streamlit-webrtc calls (`audio_source_callback`), so a backend can be listened
to on its own. That isolates how a model sounds from the latency the avatar's
prebuffer adds, which is otherwise the dominant term.

Audio is kept at the backend's own rate rather than resampled to the avatar's,
so nothing is lost on the way to the speaker.
"""

from __future__ import annotations

import threading
from fractions import Fraction

import av
import numpy as np

from .config import ARTALK_SAMPLE_RATE

DEFAULT_AUDIO_ONLY_PREBUFFER_SECONDS = 0.20


class AudioOnlyPipeline:
    def __init__(
        self,
        *,
        sample_rate: int,
        ptime: float = 0.020,
        prebuffer_seconds: float = DEFAULT_AUDIO_ONLY_PREBUFFER_SECONDS,
    ) -> None:
        self.sample_rate = sample_rate
        self._samples_per_frame = int(sample_rate * ptime)
        self._prebuffer = int(prebuffer_seconds * sample_rate)
        self._lock = threading.Lock()
        self._buffer = np.zeros(0, dtype=np.int16)
        self._playing = False
        self._served = 0
        self._underruns = 0
        self._flushes = 0
        self._resampler = av.AudioResampler(
            format="s16", layout="mono", rate=sample_rate
        )

    def push_audio_frame(self, frame: av.AudioFrame) -> None:
        for resampled in self._resampler.resample(frame):
            arr = resampled.to_ndarray().reshape(-1).astype(np.int16, copy=False)
            if arr.size == 0:
                continue
            with self._lock:
                self._buffer = np.concatenate((self._buffer, arr))

    def flush_output(self) -> None:
        """Drop audio not yet played, so barge-in silences the reply at once."""
        with self._lock:
            # Credit the discard to the served clock: callers measure what the
            # listener heard against it, and discarded audio was never heard
            # but must not read as still pending.
            self._served += self._buffer.size
            self._buffer = np.zeros(0, dtype=np.int16)
            self._playing = False
            self._flushes += 1

    def request_chunk_flush(self) -> None:
        """No-op: there is no motion model here to flush a partial chunk into."""

    def metrics_snapshot(self) -> dict:
        with self._lock:
            buffered = int(self._buffer.size)
            served = self._served
            underruns = self._underruns
            flushes = self._flushes
        # The bridges read this clock in the avatar's sample rate, so report it
        # in those units whatever rate this pipeline actually runs at.
        served_16k = int(served * ARTALK_SAMPLE_RATE / self.sample_rate)
        return {
            "counters": {
                "synced_audio_samples_served": served_16k,
                "audio_out_buffer_samples": buffered,
                "audio_out_buffer_seconds": buffered / self.sample_rate,
                "audio_playback_underrun_frames": underruns,
                "output_flushes": flushes,
            }
        }

    def audio_source_callback(self, pts: int, time_base: Fraction) -> av.AudioFrame:
        want = self._samples_per_frame
        with self._lock:
            # Hold playback until enough has arrived to ride out jitter, then
            # keep playing: restarting the wait on every gap would stutter.
            if not self._playing and self._buffer.size >= self._prebuffer:
                self._playing = True
            if self._playing and self._buffer.size:
                take = min(want, self._buffer.size)
                samples = self._buffer[:take]
                self._buffer = self._buffer[take:]
                self._served += take
                if take < want:
                    self._underruns += 1
                    samples = np.concatenate(
                        (samples, np.zeros(want - take, dtype=np.int16))
                    )
            else:
                if self._playing:
                    self._underruns += 1
                samples = np.zeros(want, dtype=np.int16)

        frame = av.AudioFrame.from_ndarray(
            samples[np.newaxis, :], format="s16", layout="mono"
        )
        frame.sample_rate = self.sample_rate
        frame.pts = pts
        frame.time_base = time_base
        return frame
