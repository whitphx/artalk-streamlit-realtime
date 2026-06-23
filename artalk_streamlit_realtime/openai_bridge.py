"""OpenAI Realtime transport glue for the Streamlit demo."""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from typing import Callable, Optional

import av
import numpy as np
from artalk.realtime_pipeline import ARTalkPipeline

from .config import OPENAI_REALTIME_SAMPLE_RATE

logger = logging.getLogger(__name__)


class OpenAIRealtimeBridge:
    """Bridge browser mic audio to OpenAI and feed response audio to ARTalk."""

    def __init__(
        self,
        *,
        api_key: str,
        pipeline: ARTalkPipeline,
        on_audio_output: Callable[[], None] | None,
        model: str,
        voice: str,
        instructions: str,
    ) -> None:
        self._api_key = api_key
        self._pipeline = pipeline
        self._on_audio_output = on_audio_output
        self._model = model
        self._voice = voice
        self._instructions = instructions

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._input_queue: Optional["asyncio.Queue[bytes]"] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._resampler = av.AudioResampler(
            format="s16", layout="mono", rate=OPENAI_REALTIME_SAMPLE_RATE
        )

        self._state_lock = threading.Lock()
        self._connected = False
        self._error: Optional[str] = None
        self._user_transcript = ""
        self._assistant_transcript = ""

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._ready_event.clear()
        with self._state_lock:
            self._error = None
        self._thread = threading.Thread(
            target=self._run,
            name="OpenAIRealtimeBridge",
            daemon=True,
        )
        self._thread.start()
        self._ready_event.wait(timeout=3.0)

    def wait_until_connected(self, timeout: float) -> bool:
        stop_at = time.monotonic() + timeout
        while time.monotonic() < stop_at:
            with self._state_lock:
                if self._connected:
                    return True
                if self._error:
                    return False
            time.sleep(0.05)
        return False

    def stop(self) -> None:
        loop, stop_event = self._loop, self._stop_event
        if loop is not None and stop_event is not None and not loop.is_closed():
            loop.call_soon_threadsafe(stop_event.set)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
        self._thread = None
        with self._state_lock:
            self._connected = False

    def push_input(self, frame: av.AudioFrame) -> None:
        loop, q = self._loop, self._input_queue
        if loop is None or q is None or loop.is_closed():
            return
        for resampled in self._resampler.resample(frame):
            arr = resampled.to_ndarray()
            pcm = arr.astype(np.int16, copy=False).tobytes()
            if not pcm:
                continue
            try:
                loop.call_soon_threadsafe(self._queue_input, pcm)
            except RuntimeError:
                return

    def snapshot(self) -> dict:
        with self._state_lock:
            return {
                "connected": self._connected,
                "error": self._error,
                "user": self._user_transcript,
                "assistant": self._assistant_transcript,
            }

    def _queue_input(self, pcm: bytes) -> None:
        if self._input_queue is None:
            return
        try:
            self._input_queue.put_nowait(pcm)
        except asyncio.QueueFull:
            pass

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            self._input_queue = asyncio.Queue(maxsize=256)
            self._stop_event = asyncio.Event()
            self._ready_event.set()
            loop.run_until_complete(self._session())
        except Exception as exc:
            logger.exception("OpenAI Realtime bridge crashed")
            with self._state_lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._state_lock:
                self._connected = False
            try:
                loop.close()
            finally:
                self._loop = None

    async def _session(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the OpenAI SDK to use Interactive mode: pip install openai"
            ) from exc

        if self._stop_event is None or self._input_queue is None:
            raise RuntimeError("Realtime bridge loop is not initialized")

        client = AsyncOpenAI(api_key=self._api_key)
        async with client.realtime.connect(model=self._model) as conn:
            await conn.session.update(
                session={
                    "type": "realtime",
                    "model": self._model,
                    "instructions": self._instructions,
                    "audio": {
                        "input": {"turn_detection": {"type": "server_vad"}},
                        "output": {"voice": self._voice},
                    },
                }
            )
            with self._state_lock:
                self._connected = True

            tasks = [
                asyncio.create_task(self._send_loop(conn), name="openai-send"),
                asyncio.create_task(self._recv_loop(conn), name="openai-recv"),
                asyncio.create_task(self._stop_event.wait(), name="openai-stop"),
            ]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_loop(self, conn) -> None:
        if self._input_queue is None:
            return
        while True:
            pcm = await self._input_queue.get()
            await conn.input_audio_buffer.append(
                audio=base64.b64encode(pcm).decode("ascii")
            )

    async def _recv_loop(self, conn) -> None:
        async for event in conn:
            etype = getattr(event, "type", "")
            if etype == "response.output_audio.delta":
                self._push_response_audio(base64.b64decode(event.delta))
            elif etype == "response.output_audio_transcript.delta":
                with self._state_lock:
                    self._assistant_transcript += getattr(event, "delta", "") or ""
            elif etype == "response.done":
                with self._state_lock:
                    if (
                        self._assistant_transcript
                        and not self._assistant_transcript.endswith("\n")
                    ):
                        self._assistant_transcript += "\n"
            elif etype == "conversation.item.input_audio_transcription.delta":
                with self._state_lock:
                    self._user_transcript += getattr(event, "delta", "") or ""
            elif etype == "conversation.item.input_audio_transcription.completed":
                with self._state_lock:
                    if self._user_transcript and not self._user_transcript.endswith("\n"):
                        self._user_transcript += "\n"
            elif etype == "error":
                err = getattr(event, "error", None)
                msg = getattr(err, "message", None) or repr(err)
                logger.warning("OpenAI Realtime API error: %s", msg)
                with self._state_lock:
                    self._error = msg

    def _push_response_audio(self, pcm: bytes) -> None:
        if len(pcm) < 2:
            return
        if len(pcm) % 2:
            pcm = pcm[:-1]
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return
        frame = av.AudioFrame.from_ndarray(
            samples[np.newaxis, :], format="s16", layout="mono"
        )
        frame.sample_rate = OPENAI_REALTIME_SAMPLE_RATE
        if self._on_audio_output is not None:
            self._on_audio_output()
        self._pipeline.push_audio_frame(frame)

