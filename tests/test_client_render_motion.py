import json
import os
import shutil
import subprocess
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import torch

from artalk_streamlit_realtime.client_render import (
    CAMERA_OFFSET,
    HEAD_VERTICES,
    HEADER,
    OFFSET_LIMIT,
    OFFSETS_HEADER,
    OFFSETS_HEADER_OFFSET,
    PAYLOAD_OFFSET,
    QUANT_STEP,
    ClientRenderAdapter,
    ClientRenderAssets,
    nearest_earlier_vertices,
)

NODE = shutil.which("node")
JS_CHECK = Path(__file__).with_name("client_render_motion_check.mjs")


def synthetic_head(seed=0):
    """Neutral vertices on a jittered ellipsoid, with a few exact ties: a
    duplicated vertex and vertices mirrored across x = 0."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(HEAD_VERTICES, 3))
    ref = (u / np.linalg.norm(u, axis=1, keepdims=True) * [0.08, 0.1, 0.09]).astype(np.float32)
    ref[10] = ref[3]
    ref[20:40] = ref[0:20] * np.array([-1, 1, 1], dtype=np.float32)
    return ref


def synthetic_points(ref, frames, seed=0):
    """Smooth motion over the head, largest near the "mouth", plus a vertex
    pushed past the offset range so the clamp is exercised."""
    rng = np.random.default_rng(seed)
    points = []
    for t in range(frames):
        phase = rng.uniform(0, 2 * np.pi, size=3)
        wave = np.sin(ref * 40 + phase) * 0.01 * np.exp(-((ref[:, 1:2] + 0.05) ** 2) / 0.002)
        p = ref + wave.astype(np.float32)
        p[7] = ref[7] + (OFFSET_LIMIT + 50 + t) * QUANT_STEP
        points.append(p)
    return np.stack(points)


def messages_for(points, ref, nearest, camera, jaw, generation=3):
    """forward_expression's records, cut to length as the pipeline sends them."""
    batch = {
        "points": torch.from_numpy(points),
        "head_ref": torch.from_numpy(ref),
        "nearest": nearest,
        "camera": torch.from_numpy(camera),
        "jaw": torch.from_numpy(jaw),
        "generation": generation,
    }
    records = ClientRenderAdapter.__new__(ClientRenderAdapter).forward_expression(batch).numpy()
    out = []
    for row in records:
        _, length = OFFSETS_HEADER.unpack_from(row, OFFSETS_HEADER_OFFSET)
        out.append(row[: PAYLOAD_OFFSET + length].tobytes())
    return out


def decode_steps(message, nearest):
    """Reference decoder: the offsets of a message, in integer steps."""
    _, length = OFFSETS_HEADER.unpack_from(message, OFFSETS_HEADER_OFFSET)
    planes = np.frombuffer(zlib.decompress(message[PAYLOAD_OFFSET : PAYLOAD_OFFSET + length], -15), np.uint8)
    lo, hi = planes.reshape(2, -1).astype(np.int32)
    zigzag = lo | (hi << 8)
    residual = ((zigzag >> 1) ^ -(zigzag & 1)).reshape(3, -1).T
    steps = np.zeros_like(residual)
    for i, p in enumerate(nearest):
        steps[i] = residual[i] + (steps[p] if p >= 0 else 0)
    return steps


def expected_steps(points, ref):
    return np.clip(np.round((points - ref).astype(np.float64) / QUANT_STEP), -OFFSET_LIMIT, OFFSET_LIMIT)


class NearestEarlierVerticesTest(unittest.TestCase):
    def test_matches_brute_force_with_ties(self):
        ref = synthetic_head()
        r = ref.astype(np.float64)
        nearest = nearest_earlier_vertices(ref)
        self.assertEqual(nearest[0], -1)
        for i in range(1, HEAD_VERTICES, 7):
            d = (r[i, 0] - r[:i, 0]) ** 2 + (r[i, 1] - r[:i, 1]) ** 2 + (r[i, 2] - r[:i, 2]) ** 2
            self.assertEqual(nearest[i], np.argmin(d), i)
        self.assertEqual(nearest[10], 3)


class EncodeOffsetsTest(unittest.TestCase):
    def setUp(self):
        self.ref = synthetic_head()
        self.nearest = nearest_earlier_vertices(self.ref)
        self.points = synthetic_points(self.ref, frames=4)
        self.camera = np.arange(4 * 12, dtype=np.float32).reshape(4, 3, 4)
        self.jaw = np.linspace(0, 0.1, 4, dtype=np.float32)
        self.messages = messages_for(self.points, self.ref, self.nearest, self.camera, self.jaw)

    def test_round_trip(self):
        for k, message in enumerate(self.messages):
            np.testing.assert_array_equal(decode_steps(message, self.nearest), expected_steps(self.points[k], self.ref))

    def test_header_fields(self):
        for k, message in enumerate(self.messages):
            magic, _, generation, jaw, *_ = HEADER.unpack_from(message, 0)
            self.assertEqual((magic, generation), (b"ARV2", 3))
            self.assertEqual(jaw, self.jaw[k])
            camera = np.frombuffer(message, "<f4", 12, CAMERA_OFFSET)
            np.testing.assert_array_equal(camera, self.camera[k].ravel())
            step, _ = OFFSETS_HEADER.unpack_from(message, OFFSETS_HEADER_OFFSET)
            self.assertEqual(step, QUANT_STEP)

    def test_smaller_than_int16_records(self):
        for message in self.messages:
            self.assertLess(len(message), HEAD_VERTICES * 3 * 2 / 4)


@unittest.skipIf(NODE is None, "node is not installed")
class PageDecoderTest(unittest.TestCase):
    """The page's motion.js reads what the server writes."""

    def run_check(self, ref, messages):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "head_ref.f32").write_bytes(ref.astype("<f4").tobytes())
            for k, m in enumerate(messages):
                (d / f"{k}.msg").write_bytes(m)
            subprocess.run([NODE, str(JS_CHECK), tmp], check=True, capture_output=True)
            nearest = np.frombuffer((d / "nearest.i32").read_bytes(), "<i4")
            vertices = [np.frombuffer((d / f"{k}.f32").read_bytes(), "<f4").reshape(-1, 3) for k in range(len(messages))]
            headers = [json.loads((d / f"{k}.json").read_text()) for k in range(len(messages))]
        return nearest, vertices, headers

    def test_reads_v2(self):
        ref = synthetic_head(seed=1)
        nearest = nearest_earlier_vertices(ref)
        points = synthetic_points(ref, frames=3, seed=1)
        camera = np.ones((3, 3, 4), np.float32)
        jaw = np.zeros(3, np.float32)
        page_nearest, vertices, headers = self.run_check(ref, messages_for(points, ref, nearest, camera, jaw))
        np.testing.assert_array_equal(page_nearest, nearest)
        for k in range(3):
            expected = (ref.astype(np.float64) + expected_steps(points[k], ref) * QUANT_STEP).astype(np.float32)
            np.testing.assert_array_equal(vertices[k], expected)
            self.assertEqual((headers[k]["version"], headers[k]["generation"], headers[k]["step"]), (2, 3, QUANT_STEP))

    def test_reads_v1(self):
        ref = synthetic_head(seed=2)
        offsets = np.random.default_rng(2).integers(-30000, 30000, size=(HEAD_VERTICES, 3)).astype("<i2")
        message = bytearray(CAMERA_OFFSET + 48)
        HEADER.pack_into(message, 0, b"ARV1", 9, 1, 0.5, 0, 0, 0, 0, 0, 0, 0)
        message += offsets.tobytes()
        _, vertices, headers = self.run_check(ref, [bytes(message)])
        expected = (ref.astype(np.float64) + offsets * 2.0**-17).astype(np.float32)
        np.testing.assert_array_equal(vertices[0], expected)
        self.assertEqual((headers[0]["version"], headers[0]["index"]), (1, 9))

    @unittest.skipUnless(os.environ.get("ARTALK_CLIENT_RENDER_DIR"), "ARTALK_CLIENT_RENDER_DIR is not set")
    def test_same_nearest_vertices_for_real_avatars(self):
        assets = ClientRenderAssets(os.environ["ARTALK_CLIENT_RENDER_DIR"])
        for avatar_id in assets.avatar_ids():
            ref = assets.head_ref(avatar_id)
            page_nearest, _, _ = self.run_check(ref, [])
            np.testing.assert_array_equal(page_nearest, nearest_earlier_vertices(ref), avatar_id)


if __name__ == "__main__":
    unittest.main()
