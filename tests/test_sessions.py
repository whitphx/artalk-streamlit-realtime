import unittest
from unittest import mock

from artalk_streamlit_realtime import sessions
from artalk_streamlit_realtime.sessions import (
    DISCONNECT_GRACE_SECONDS,
    ENDED_TIME_LIMIT,
    FINAL_COUNTDOWN_SECONDS,
    PipelineSlots,
)

LIMIT = 120.0


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


class StoppedObject:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class TimeLimitTest(unittest.TestCase):
    """Outside a Streamlit server every session counts as connected, and the
    reaper thread is kept from starting so that each test reaps by hand
    against a fake clock."""

    def setUp(self):
        self.clock = FakeClock()
        patches = [
            mock.patch.object(sessions.time, "monotonic", self.clock.monotonic),
            mock.patch.object(PipelineSlots, "_start_reaper", lambda self: None),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def make_slots(self, limit):
        slots = PipelineSlots()
        slots.limit = limit
        slots.session_seconds = LIMIT
        return slots

    def play(self, slots, session_id):
        self.assertTrue(slots.acquire(session_id))
        slots.start_clock(session_id)
        pipeline = StoppedObject()
        slots.attach(session_id, "pipeline", pipeline)
        return pipeline

    def advance(self, slots, seconds):
        self.clock.now += seconds
        slots._reap_once()

    def close_tabs_except(self, slots, *session_ids):
        """Disconnect every other session for long enough that the reaper
        drops it from the queue."""
        with mock.patch.object(
            sessions, "_active_session_ids", return_value=set(session_ids)
        ):
            self.advance(slots, 0.0)
            self.advance(slots, DISCONNECT_GRACE_SECONDS)

    def test_no_limit_while_nobody_waits(self):
        slots = self.make_slots(limit=1)
        pipeline = self.play(slots, "a")
        self.advance(slots, 10 * LIMIT)
        self.assertTrue(slots.holds("a"))
        self.assertFalse(pipeline.stopped)
        self.assertIsNone(slots.deadline("a"))

    def test_queue_join_limits_session_under_the_limit_to_its_playing_time(self):
        slots = self.make_slots(limit=1)
        pipeline = self.play(slots, "a")
        self.advance(slots, 50.0)
        self.assertFalse(slots.acquire("b"))
        self.assertEqual(slots.deadline("a"), self.clock.now + LIMIT - 50.0)
        self.advance(slots, LIMIT - 50.0 - 1.0)
        self.assertTrue(slots.holds("a"))
        self.advance(slots, 1.0)
        self.assertTrue(pipeline.stopped)
        self.assertEqual(slots.pop_ended("a"), ENDED_TIME_LIMIT)
        self.assertTrue(slots.holds("b"))

    def test_stopped_time_does_not_count(self):
        slots = self.make_slots(limit=1)
        self.play(slots, "a")
        self.advance(slots, 50.0)
        slots.mark_stopped("a")
        self.advance(slots, 500.0)
        slots.start_clock("a")
        self.assertFalse(slots.acquire("b"))
        self.assertEqual(slots.deadline("a"), self.clock.now + LIMIT - 50.0)

    def test_session_over_the_limit_gets_a_final_countdown(self):
        slots = self.make_slots(limit=1)
        pipeline = self.play(slots, "a")
        self.advance(slots, 3 * LIMIT)
        self.assertFalse(slots.acquire("b"))
        self.assertEqual(slots.deadline("a"), self.clock.now + FINAL_COUNTDOWN_SECONDS)
        self.advance(slots, FINAL_COUNTDOWN_SECONDS - 1.0)
        self.assertFalse(pipeline.stopped)
        self.advance(slots, 1.0)
        self.assertTrue(pipeline.stopped)
        self.assertTrue(slots.holds("b"))

    def test_session_near_the_limit_gets_the_final_countdown(self):
        slots = self.make_slots(limit=1)
        self.play(slots, "a")
        self.advance(slots, LIMIT - 1.0)
        self.assertFalse(slots.acquire("b"))
        self.assertEqual(slots.deadline("a"), self.clock.now + FINAL_COUNTDOWN_SECONDS)

    def test_limit_lifted_when_the_queue_empties(self):
        slots = self.make_slots(limit=1)
        pipeline = self.play(slots, "a")
        self.advance(slots, 3 * LIMIT)
        self.assertFalse(slots.acquire("b"))
        self.advance(slots, FINAL_COUNTDOWN_SECONDS / 2)
        self.close_tabs_except(slots, "a")
        self.assertIsNone(slots.deadline("a"))
        self.advance(slots, 10 * LIMIT)
        self.assertFalse(pipeline.stopped)

    def test_next_waiter_gets_a_new_final_countdown(self):
        slots = self.make_slots(limit=1)
        self.play(slots, "a")
        self.advance(slots, 3 * LIMIT)
        self.assertFalse(slots.acquire("b"))
        self.advance(slots, FINAL_COUNTDOWN_SECONDS - 1.0)
        self.close_tabs_except(slots, "a")
        self.advance(slots, LIMIT)
        self.assertFalse(slots.acquire("c"))
        self.assertEqual(slots.deadline("a"), self.clock.now + FINAL_COUNTDOWN_SECONDS)

    def test_stopped_session_keeps_its_remaining_time(self):
        slots = self.make_slots(limit=1)
        self.play(slots, "a")
        self.assertFalse(slots.acquire("b"))
        self.advance(slots, 100.0)
        slots.mark_stopped("a")
        self.assertIsNone(slots.deadline("a"))
        self.assertEqual(slots.time_left("a"), LIMIT - 100.0)
        self.advance(slots, 5.0)
        slots.start_clock("a")
        self.assertEqual(slots.deadline("a"), self.clock.now + LIMIT - 100.0)

    def test_one_waiter_ends_only_the_longest_playing_session(self):
        slots = self.make_slots(limit=2)
        first = self.play(slots, "a")
        self.advance(slots, 100.0)
        second = self.play(slots, "b")
        self.advance(slots, 2 * LIMIT)
        self.assertFalse(slots.acquire("c"))
        self.assertEqual(slots.deadline("a"), slots.deadline("b"))
        self.advance(slots, FINAL_COUNTDOWN_SECONDS)
        self.assertTrue(first.stopped)
        self.assertFalse(second.stopped)
        self.assertTrue(slots.holds("c"))
        self.assertIsNone(slots.deadline("b"))

    def test_estimated_wait_follows_the_holders_deadline(self):
        slots = self.make_slots(limit=1)
        self.play(slots, "a")
        self.advance(slots, 3 * LIMIT)
        self.assertFalse(slots.acquire("b"))
        self.assertFalse(slots.acquire("c"))
        self.assertEqual(slots.estimated_wait("b"), FINAL_COUNTDOWN_SECONDS)
        self.assertEqual(slots.estimated_wait("c"), FINAL_COUNTDOWN_SECONDS + LIMIT)


if __name__ == "__main__":
    unittest.main()
