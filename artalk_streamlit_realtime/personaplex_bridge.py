"""PersonaPlex (moshi) transport glue for the Streamlit demo.

Speaks the moshi server's `/api/chat` protocol: binary WebSocket frames whose
first byte is the opcode, with Opus-coded audio in both directions at Mimi's
sample rate.

Unlike the OpenAI Realtime API there are no turns. The model emits audio
continuously, including its own silence, and yields by going quiet rather than
by ending a response. There is consequently no truncate call, no server-VAD
event, and nothing to trigger a chunk flush, so the barge-in reconciliation
that `OpenAIRealtimeBridge` performs has no counterpart here. `snapshot()`
instead reports how far the listener trails the model, which is the quantity
that decides whether full-duplex is usable behind the avatar's output buffer.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional
from urllib.parse import urlencode

import av
import numpy as np

from .config import ARTALK_SAMPLE_RATE, PERSONAPLEX_SAMPLE_RATE

if TYPE_CHECKING:
    from artalk.realtime_pipeline import ARTalkPipeline

logger = logging.getLogger(__name__)

_OPCODE_HANDSHAKE = 0
_OPCODE_AUDIO = 1
_OPCODE_TEXT = 2

# Opus only accepts whole frames, and mic chunks never line up with one.
# 1920 samples is 80 ms at 24 kHz, matching Mimi's frame size.
_OPUS_FRAME_SAMPLES = 1920


class PersonaPlexBridge:
    """Bridge browser mic audio to a moshi server and feed its audio to ARTalk."""

    def __init__(
        self,
        *,
        url: str,
        pipeline: ARTalkPipeline,
        on_audio_output: Callable[[], None] | None,
        text_prompt: str = "",
        voice_prompt: str = "",
        seed: int | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._pipeline = pipeline
        self._on_audio_output = on_audio_output
        self._text_prompt = text_prompt
        self._voice_prompt = voice_prompt
        self._seed = seed

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._input_queue: Optional["asyncio.Queue[np.ndarray]"] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._handshake: Optional[asyncio.Event] = None
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stop_lock = threading.Lock()
        self._resampler = av.AudioResampler(
            format="s16", layout="mono", rate=PERSONAPLEX_SAMPLE_RATE
        )
        self._opus_writer = None
        self._opus_reader = None

        self._state_lock = threading.Lock()
        self._connected = False
        self._error: Optional[str] = None
        self._assistant_transcript = ""

        # Counted on the receive side only. play_start pins the pipeline's
        # served clock at the moment the model's first audio arrived, so the
        # trailing distance can be read off it without a turn boundary.
        self._model_samples_out = 0
        self._play_start_16k: Optional[int] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        if self._thread is not None and not self._thread.is_alive():
            self._thread = None
        self._ready_event.clear()
        with self._state_lock:
            self._error = None
            self._connected = False
            self._assistant_transcript = ""
            self._model_samples_out = 0
            self._play_start_16k = None
        self._thread = threading.Thread(
            target=self._run,
            name="PersonaPlexBridge",
            daemon=True,
        )
        self._thread.start()
        self._ready_event.wait(timeout=3.0)

    def wait_until_connected(self, timeout: float) -> bool:
        # The server withholds its handshake until the system prompts have been
        # stepped through the model, which takes seconds on a cold session.
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
        with self._stop_lock:
            loop = self._loop
            thread = self._thread
            if loop is not None and not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(self._request_stop)
                except RuntimeError:
                    pass
            if thread is not None and threading.current_thread() is not thread:
                thread.join(timeout=3.0)
            if thread is None or not thread.is_alive():
                self._thread = None
            else:
                logger.warning("PersonaPlex bridge thread did not stop cleanly")
        with self._state_lock:
            self._connected = False

    def push_input(self, frame: av.AudioFrame) -> None:
        loop, q = self._loop, self._input_queue
        if loop is None or q is None or loop.is_closed():
            return
        for resampled in self._resampler.resample(frame):
            arr = resampled.to_ndarray()
            if arr.size == 0:
                continue
            pcm = (arr.astype(np.float32) / 32768.0).reshape(-1)
            try:
                loop.call_soon_threadsafe(self._queue_input, pcm)
            except RuntimeError:
                return

    def snapshot(self) -> dict:
        # Read the pipeline before taking the lock: metrics_snapshot() takes
        # the pipeline's own lock, and holding both invites a deadlock.
        served_16k = self._pipeline.metrics_snapshot()["counters"].get(
            "synced_audio_samples_served", 0
        )
        with self._state_lock:
            model_s = self._model_samples_out / PERSONAPLEX_SAMPLE_RATE
            if self._play_start_16k is None:
                trailing_s = 0.0
            else:
                heard_s = max(served_16k - self._play_start_16k, 0) / ARTALK_SAMPLE_RATE
                trailing_s = max(model_s - heard_s, 0.0)
            return {
                "connected": self._connected,
                "error": self._error,
                # Full-duplex models transcribe only their own speech, so the
                # user side stays empty; the app renders both fields.
                "user": "",
                "assistant": self._assistant_transcript,
                "model_audio_s": model_s,
                "trailing_s": trailing_s,
            }

    def _request_stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

    def _queue_input(self, pcm: np.ndarray) -> None:
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
            self._handshake = asyncio.Event()
            self._ready_event.set()
            loop.run_until_complete(self._session())
        except Exception as exc:
            logger.exception("PersonaPlex bridge crashed")
            with self._state_lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._state_lock:
                self._connected = False
            try:
                loop.close()
            finally:
                self._loop = None
                self._input_queue = None
                self._stop_event = None
                self._handshake = None

    def _chat_url(self) -> str:
        # The server indexes both prompt parameters unconditionally, so they
        # have to be present even when empty.
        query = {"text_prompt": self._text_prompt, "voice_prompt": self._voice_prompt}
        if self._seed is not None:
            query["seed"] = str(self._seed)
        return f"{self._url}/api/chat?{urlencode(query)}"

    async def _session(self) -> None:
        try:
            import aiohttp
            import sphn
        except ImportError as exc:
            raise RuntimeError(
                "PersonaPlex mode needs aiohttp and sphn: pip install aiohttp 'sphn<0.2'"
            ) from exc

        if self._stop_event is None or self._input_queue is None:
            raise RuntimeError("PersonaPlex bridge loop is not initialized")

        self._opus_writer = sphn.OpusStreamWriter(PERSONAPLEX_SAMPLE_RATE)
        self._opus_reader = sphn.OpusStreamReader(PERSONAPLEX_SAMPLE_RATE)

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self._chat_url()) as ws:
                tasks = [
                    asyncio.create_task(self._send_loop(ws), name="personaplex-send"),
                    asyncio.create_task(self._recv_loop(ws), name="personaplex-recv"),
                    asyncio.create_task(self._decode_loop(), name="personaplex-decode"),
                    asyncio.create_task(self._stop_event.wait(), name="personaplex-stop"),
                ]
                try:
                    done, _ = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    # A loop that dies takes the session with it, so surface
                    # its exception instead of reporting a clean disconnect.
                    for task in done:
                        exc = task.exception()
                        if exc is not None:
                            logger.error("%s failed", task.get_name(), exc_info=exc)
                            with self._state_lock:
                                self._error = f"{type(exc).__name__}: {exc}"
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_loop(self, ws) -> None:
        if self._input_queue is None or self._handshake is None:
            raise RuntimeError("PersonaPlex bridge loop is not initialized")
        # Nothing may go out before the handshake. While the server steps its
        # system prompts it polls liveness with a destructive ws.receive(), so
        # early frames are silently dropped; losing the first one is fatal
        # because it carries the Opus stream header, after which the server's
        # decoder yields None and its opus_loop dies.
        await self._handshake.wait()
        # Mic audio captured during that wait is stale. Feeding it now would
        # hand a realtime model several seconds at once and leave the session
        # permanently behind, so start from live audio instead.
        while not self._input_queue.empty():
            self._input_queue.get_nowait()
        pending = np.zeros(0, dtype=np.float32)
        while True:
            try:
                pcm = await asyncio.wait_for(self._input_queue.get(), timeout=0.02)
                pending = np.concatenate((pending, pcm))
            except asyncio.TimeoutError:
                pass
            while pending.size >= _OPUS_FRAME_SAMPLES:
                self._opus_writer.append_pcm(pending[:_OPUS_FRAME_SAMPLES])
                pending = pending[_OPUS_FRAME_SAMPLES:]
            data = self._opus_writer.read_bytes()
            if data:
                await ws.send_bytes(bytes([_OPCODE_AUDIO]) + data)

    async def _recv_loop(self, ws) -> None:
        import aiohttp

        async for message in ws:
            if message.type != aiohttp.WSMsgType.BINARY:
                if message.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    break
                continue
            data = message.data
            if not data:
                continue
            kind = data[0]
            if kind == _OPCODE_HANDSHAKE:
                with self._state_lock:
                    self._connected = True
                if self._handshake is not None:
                    self._handshake.set()
            elif kind == _OPCODE_AUDIO:
                self._opus_reader.append_bytes(data[1:])
            elif kind == _OPCODE_TEXT:
                piece = data[1:].decode("utf8", errors="replace")
                with self._state_lock:
                    self._assistant_transcript += piece
            else:
                logger.warning("Unknown PersonaPlex opcode %s", kind)

    async def _decode_loop(self) -> None:
        # sphn decodes on its own thread, so the PCM shows up a little after
        # the bytes go in; the server polls this way too.
        while True:
            await asyncio.sleep(0.005)
            pcm = self._opus_reader.read_pcm()
            if pcm.shape[-1] == 0:
                continue
            self._push_output(pcm)

    def _push_output(self, pcm: np.ndarray) -> None:
        samples = np.clip(pcm * 32768.0, -32768, 32767).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(
            samples[np.newaxis, :], format="s16", layout="mono"
        )
        frame.sample_rate = PERSONAPLEX_SAMPLE_RATE
        need_start = self._play_start_16k is None
        served_16k = (
            self._pipeline.metrics_snapshot()["counters"].get(
                "synced_audio_samples_served", 0
            )
            if need_start
            else 0
        )
        with self._state_lock:
            if self._play_start_16k is None:
                self._play_start_16k = int(served_16k)
            self._model_samples_out += samples.size
        if self._on_audio_output is not None:
            self._on_audio_output()
        self._pipeline.push_audio_frame(frame)
