"""OpenAI GPT-Live transport glue for the Streamlit demo.

Live is a separate protocol from the Realtime API rather than another model
name for it: the connection carries no model query parameter, the session is
configured by `session.start`, audio moves as `session.input_audio.append` and
`session.output_audio.delta`, and the voice frontend delegates task work to a
backend instead of doing it itself.

It is also full duplex. There is no turn boundary, no end-of-speech event and
no truncate call, so the barge-in reconciliation `OpenAIRealtimeBridge`
performs has no counterpart here; as in `PersonaPlexBridge`, `snapshot()`
reports how far the listener trails the model instead.
"""

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

from .config import (
    ARTALK_SAMPLE_RATE,
    OPENAI_LIVE_CLOSE_DRAIN_SECONDS,
    OPENAI_LIVE_OUTPUT_IDLE_FLUSH_SECONDS,
    OPENAI_LIVE_SAMPLE_RATE,
)
from .event_log import SessionEventLog

if TYPE_CHECKING:
    from artalk.realtime_pipeline import ARTalkPipeline

logger = logging.getLogger(__name__)


class OpenAILiveBridge:
    """Bridge browser mic audio to GPT-Live and feed its audio to ARTalk."""

    def __init__(
        self,
        *,
        api_key: str,
        pipeline: ARTalkPipeline,
        on_audio_output: Callable[[], None] | None,
        model: str,
        voice: str,
        instructions: str,
        delegation_model: str = "",
        event_log: SessionEventLog | None = None,
    ) -> None:
        self._api_key = api_key
        self._pipeline = pipeline
        self._on_audio_output = on_audio_output
        self._event_log = event_log
        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._delegation_model = delegation_model

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._input_queue: Optional["asyncio.Queue[bytes]"] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._closed_event: Optional[asyncio.Event] = None
        self._conn = None
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stop_lock = threading.Lock()
        self._resampler = av.AudioResampler(
            format="s16", layout="mono", rate=OPENAI_LIVE_SAMPLE_RATE
        )

        self._state_lock = threading.Lock()
        self._connected = False
        self._error: Optional[str] = None
        self._user_transcript = ""
        self._assistant_transcript = ""

        self._model_samples_out = 0

        # Touched on the event loop only.
        self._last_output_delta_s = 0.0
        self._chunk_flush_pending = False

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
            self._user_transcript = ""
            self._assistant_transcript = ""
            self._model_samples_out = 0
        self._thread = threading.Thread(
            target=self._run,
            name="OpenAILiveBridge",
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
                        self._close_live_session(),
                        loop,
                    )
                    close_future.result(timeout=3.0)
                except concurrent.futures.TimeoutError:
                    logger.warning("Timed out while closing GPT-Live session")
                except Exception:
                    logger.debug("Failed to request GPT-Live close", exc_info=True)
            if thread is not None and threading.current_thread() is not thread:
                thread.join(timeout=3.0)
            if thread is None or not thread.is_alive():
                self._thread = None
            else:
                logger.warning("GPT-Live bridge thread did not stop cleanly")
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
        # Read the pipeline before taking the lock: metrics_snapshot() takes
        # the pipeline's own lock, and holding both invites a deadlock.
        counters = self._pipeline.metrics_snapshot()["counters"]
        # Audio that has been pushed but not yet served is what the listener
        # still has to sit through before hearing anything said after it, so
        # it stands in for the turn boundary a full-duplex session never draws.
        queued_16k = (
            counters.get("audio_out_buffer_samples", 0)
            + counters.get("pending_audio_for_output_samples", 0)
            + counters.get("streamer_buffer_samples", 0)
        )
        with self._state_lock:
            return {
                "connected": self._connected,
                "error": self._error,
                "user": self._user_transcript,
                "assistant": self._assistant_transcript,
                "model_audio_s": self._model_samples_out / OPENAI_LIVE_SAMPLE_RATE,
                "trailing_s": queued_16k / ARTALK_SAMPLE_RATE,
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
            self._closed_event = asyncio.Event()
            self._ready_event.set()
            loop.run_until_complete(self._session())
        except Exception as exc:
            logger.exception("GPT-Live bridge crashed")
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
                self._closed_event = None
                self._conn = None

    def _session_config(self) -> dict:
        session: dict = {
            "model": self._model,
            "instructions": self._instructions,
            "audio": {
                "format": {"type": "audio/pcm", "rate": OPENAI_LIVE_SAMPLE_RATE},
                "output": {"voice": self._voice},
            },
        }
        if self._delegation_model:
            session["delegation"] = {
                "type": "responses",
                "responses": {
                    "model": self._delegation_model,
                    "tools": [{"type": "web_search"}],
                    "tool_choice": "auto",
                },
            }
        return session

    async def _session(self) -> None:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "Install the OpenAI SDK to use Interactive mode: pip install openai"
            ) from exc

        if self._stop_event is None or self._input_queue is None:
            raise RuntimeError("Live bridge loop is not initialized")

        client = openai.AsyncOpenAI(api_key=self._api_key)
        if not hasattr(client, "live"):
            raise RuntimeError(
                "GPT-Live needs openai 3.12 or newer, which is where "
                f"`client.live` arrives; this environment has {openai.__version__}."
            )
        try:
            async with client.live.connect() as conn:
                self._conn = conn
                await conn.session.start(session=self._session_config())

                tasks = [
                    asyncio.create_task(self._send_loop(conn), name="live-send"),
                    asyncio.create_task(self._recv_loop(conn), name="live-recv"),
                    asyncio.create_task(self._flush_loop(), name="live-flush"),
                    asyncio.create_task(self._stop_event.wait(), name="live-stop"),
                ]
                try:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._conn = None
            self._record_event("live", "session closed")
            await client.close()

    async def _close_live_session(self) -> None:
        conn = self._conn
        if conn is not None and self._closed_event is not None:
            # Live finalizes the session asynchronously, and only the
            # session.closed event confirms it; dropping the socket first
            # forfeits the final usage report.
            try:
                await conn.session.close()
                await asyncio.wait_for(
                    self._closed_event.wait(), timeout=OPENAI_LIVE_CLOSE_DRAIN_SECONDS
                )
            except asyncio.TimeoutError:
                logger.warning("GPT-Live session did not confirm close")
            except Exception:
                logger.debug("GPT-Live close request failed", exc_info=True)
        if self._stop_event is not None:
            self._stop_event.set()

    async def _send_loop(self, conn) -> None:
        if self._input_queue is None:
            return
        while True:
            pcm = await self._input_queue.get()
            await conn.session.input_audio.append(
                audio=base64.b64encode(pcm).decode("ascii")
            )

    async def _flush_loop(self) -> None:
        """Flush the partial ARTalk chunk once the model stops sending audio.

        Live has no end-of-speech event; the deltas simply stop. Without this
        the tail of a response waits for the silence pump's realtime-paced
        fill, which lands seconds late and pauses playback right before the
        final words.
        """
        while True:
            await asyncio.sleep(OPENAI_LIVE_OUTPUT_IDLE_FLUSH_SECONDS / 2)
            if not self._chunk_flush_pending:
                continue
            idle_s = time.monotonic() - self._last_output_delta_s
            if idle_s < OPENAI_LIVE_OUTPUT_IDLE_FLUSH_SECONDS:
                continue
            self._chunk_flush_pending = False
            self._record_event("live", "output audio went idle", f"idle={idle_s:.2f}s")
            self._pipeline.request_chunk_flush()

    async def _recv_loop(self, conn) -> None:
        async for event in conn:
            etype = getattr(event, "type", "")
            if etype == "session.output_audio.delta":
                self._last_output_delta_s = time.monotonic()
                self._chunk_flush_pending = True
                self._push_output_audio(base64.b64decode(event.delta))
            elif etype == "session.started":
                with self._state_lock:
                    self._connected = True
                self._record_event(
                    "live",
                    "session started",
                    f"id={event.session.id} model={self._model} "
                    f"delegation={self._delegation_model or 'client'}",
                )
            elif etype == "session.output_transcript.delta":
                with self._state_lock:
                    self._assistant_transcript += event.delta
            elif etype == "session.input_transcript.delta":
                with self._state_lock:
                    self._user_transcript += event.delta
            elif etype == "session.delegation.created":
                await self._handle_delegation(conn, event.delegation)
            elif etype == "session.usage.updated":
                self._record_event(
                    "live", "usage updated", f"{event.usage.seconds:.0f}s"
                )
            elif etype == "session.closed":
                self._record_event(
                    "live",
                    "session finalized",
                    f"reason={event.reason} billed={event.usage.seconds:.0f}s",
                )
                if self._closed_event is not None:
                    self._closed_event.set()
            elif etype == "error":
                message = getattr(event.error, "message", None) or repr(event.error)
                logger.warning("GPT-Live API error: %s", message)
                self._record_event("live", "API error", message)
                with self._state_lock:
                    self._error = message

    async def _handle_delegation(self, conn, delegation) -> None:
        self._record_event(
            "live",
            "model delegated work",
            f"id={delegation.id} target={delegation.target}",
        )
        if delegation.target != "client":
            return
        # Nothing in this demo answers delegated work, and a delegation left
        # unanswered keeps the model waiting on it.
        await conn.session.commentary.append(
            content="No backend is connected, so that request cannot be completed.",
            delegation_id=delegation.id,
        )

    def _push_output_audio(self, pcm: bytes) -> None:
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
        frame.sample_rate = OPENAI_LIVE_SAMPLE_RATE
        with self._state_lock:
            self._model_samples_out += samples.size
        if self._on_audio_output is not None:
            self._on_audio_output()
        self._pipeline.push_audio_frame(frame)
