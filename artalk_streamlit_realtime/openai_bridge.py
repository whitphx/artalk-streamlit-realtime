"""OpenAI Realtime transport glue for the Streamlit demo."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional

import av
import numpy as np

from .config import ARTALK_SAMPLE_RATE, OPENAI_REALTIME_SAMPLE_RATE
from .event_log import SessionEventLog

if TYPE_CHECKING:
    from artalk.realtime_pipeline import ARTalkPipeline

logger = logging.getLogger(__name__)


class OpenAIRealtimeBridge:
    """Bridge browser mic audio to a Realtime API and feed its audio to ARTalk."""

    def __init__(
        self,
        *,
        api_key: str,
        pipeline: ARTalkPipeline,
        on_audio_output: Callable[[], None] | None,
        model: str,
        voice: str,
        instructions: str,
        websocket_base_url: str = "",
        event_log: SessionEventLog | None = None,
    ) -> None:
        self._api_key = api_key
        self._pipeline = pipeline
        self._on_audio_output = on_audio_output
        self._event_log = event_log
        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._websocket_base_url = websocket_base_url

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._input_queue: Optional["asyncio.Queue[bytes]"] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._conn = None
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stop_lock = threading.Lock()
        self._resampler = av.AudioResampler(
            format="s16", layout="mono", rate=OPENAI_REALTIME_SAMPLE_RATE
        )

        self._state_lock = threading.Lock()
        self._connected = False
        self._error: Optional[str] = None
        self._user_transcript = ""
        self._assistant_transcript = ""

        # Current assistant audio item, tracked on the receive loop only.
        # play_start is the pipeline served-samples position at which this
        # item's audio begins playing (served + everything queued ahead of it
        # when its first delta arrived), so heard time can be read off the
        # served clock at barge-in.
        self._assistant_item_id: Optional[str] = None
        self._assistant_item_content_index = 0
        self._assistant_item_pushed_ms = 0.0
        self._assistant_item_play_start_16k = 0

    def _record_event(self, category: str, message: str, detail: str = "") -> None:
        if self._event_log is not None:
            self._event_log.record(category, message, detail)

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
        with self._stop_lock:
            loop = self._loop
            thread = self._thread
            if threading.current_thread() is thread:
                if self._stop_event is not None:
                    self._stop_event.set()
            elif loop is not None and not loop.is_closed():
                try:
                    close_future = asyncio.run_coroutine_threadsafe(
                        self._close_realtime_session(),
                        loop,
                    )
                    close_future.result(timeout=3.0)
                except concurrent.futures.TimeoutError:
                    logger.warning("Timed out while closing OpenAI Realtime session")
                except Exception:
                    logger.debug("Failed to request OpenAI Realtime close", exc_info=True)
            if thread is not None and threading.current_thread() is not thread:
                thread.join(timeout=3.0)
            if thread is None or not thread.is_alive():
                self._thread = None
            else:
                logger.warning("OpenAI Realtime bridge thread did not stop cleanly")
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
                self._input_queue = None
                self._stop_event = None
                self._conn = None

    async def _session(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the OpenAI SDK to use Interactive mode: pip install openai"
            ) from exc

        if self._stop_event is None or self._input_queue is None:
            raise RuntimeError("Realtime bridge loop is not initialized")

        client = AsyncOpenAI(
            # Local Realtime-compatible servers ignore the Authorization header,
            # but the SDK requires a key to construct the client.
            api_key=self._api_key or "local-server",
            websocket_base_url=self._websocket_base_url or None,
        )
        try:
            async with client.realtime.connect(model=self._model) as conn:
                self._conn = conn
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
                endpoint = self._websocket_base_url or "openai"
                self._record_event(
                    "openai",
                    "session connected",
                    f"model={self._model} endpoint={endpoint}",
                )

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
        finally:
            self._conn = None
            self._record_event("openai", "session closed")
            await client.close()

    async def _close_realtime_session(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        conn = self._conn
        if conn is not None:
            try:
                await conn.close()
            except Exception:
                logger.debug("OpenAI Realtime websocket close failed", exc_info=True)

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
                pcm = base64.b64decode(event.delta)
                self._track_response_item(event, pcm)
                self._push_response_audio(pcm)
            elif etype == "response.output_audio.done":
                self._record_event(
                    "openai",
                    "response audio stream ended",
                    f"pushed={self._assistant_item_pushed_ms / 1000.0:.2f}s "
                    f"item={self._assistant_item_id}",
                )
                # The response's audio is complete; flush the sub-chunk tail
                # immediately instead of waiting for the silence pump's
                # realtime-paced fill, which lands seconds too late and
                # pauses playback right before the final words.
                self._pipeline.request_chunk_flush()
            elif etype == "input_audio_buffer.speech_started":
                self._record_event("openai", "user speech started (server VAD)")
                # Barge-in: server VAD detected the user talking over the
                # assistant. OpenAI cancels its in-flight response; drop the
                # already-buffered remainder on our side too so the avatar
                # stops speaking instead of playing it out, and truncate the
                # conversation item to what was actually heard.
                await self._handle_barge_in(conn)
            elif etype == "input_audio_buffer.speech_stopped":
                self._record_event("openai", "user speech stopped (server VAD)")
            elif etype == "response.created":
                self._record_event("openai", "response created")
            elif etype == "response.output_audio_transcript.delta":
                with self._state_lock:
                    self._assistant_transcript += getattr(event, "delta", "") or ""
            elif etype == "response.done":
                self._record_event("openai", "response done")
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
                self._record_event("openai", "API error", msg)
                with self._state_lock:
                    self._error = msg

    def _track_response_item(self, event, pcm: bytes) -> None:
        item_id = getattr(event, "item_id", None)
        if item_id is None:
            return
        if item_id != self._assistant_item_id:
            counters = self._pipeline.metrics_snapshot()["counters"]
            # The output buffer is read in seconds because the audio-only
            # pipeline holds it at the backend's rate rather than ARTalk's;
            # the stages behind it are ARTalk's own and always count at its.
            queued_ahead_16k = int(
                counters.get("audio_out_buffer_seconds", 0.0) * ARTALK_SAMPLE_RATE
                + counters.get("pending_audio_for_output_samples", 0)
                + counters.get("streamer_buffer_samples", 0)
            )
            self._assistant_item_id = item_id
            self._assistant_item_content_index = int(
                getattr(event, "content_index", 0) or 0
            )
            self._assistant_item_pushed_ms = 0.0
            self._assistant_item_play_start_16k = int(
                counters.get("synced_audio_samples_served", 0) + queued_ahead_16k
            )
            self._record_event(
                "openai",
                "response audio stream started -> feeding model",
                f"item={item_id} queued_ahead={queued_ahead_16k / ARTALK_SAMPLE_RATE:.2f}s",
            )
        self._assistant_item_pushed_ms += (
            (len(pcm) // 2) * 1000.0 / OPENAI_REALTIME_SAMPLE_RATE
        )

    async def _handle_barge_in(self, conn) -> None:
        item_id = self._assistant_item_id
        heard_ms = 0.0
        if item_id is not None:
            # Read the served clock before flush_output(): the flush credits
            # the discarded samples to that clock.
            counters = self._pipeline.metrics_snapshot()["counters"]
            heard_16k = (
                counters.get("synced_audio_samples_served", 0)
                - self._assistant_item_play_start_16k
            )
            heard_ms = min(
                max(heard_16k * 1000.0 / ARTALK_SAMPLE_RATE, 0.0),
                self._assistant_item_pushed_ms,
            )
        self._record_event(
            "openai",
            "barge-in: flushing queued response",
            f"heard={heard_ms / 1000.0:.2f}s of "
            f"{self._assistant_item_pushed_ms / 1000.0:.2f}s pushed",
        )
        self._pipeline.flush_output()
        if (
            item_id is not None
            and heard_ms < self._assistant_item_pushed_ms - 100.0
        ):
            try:
                await conn.conversation.item.truncate(
                    item_id=item_id,
                    content_index=self._assistant_item_content_index,
                    audio_end_ms=int(heard_ms),
                )
            except Exception:
                logger.warning(
                    "conversation.item.truncate failed for %s", item_id,
                    exc_info=True,
                )
        self._assistant_item_id = None
        self._assistant_item_pushed_ms = 0.0

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
