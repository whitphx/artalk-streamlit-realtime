// Reads motion messages (layout in client_render.py) into the camera and
// head-vertex positions the renderer draws. Free of DOM and GPU calls, so it
// also runs under Node.

const HEADER_BYTES = 56;
const CAMERA_OFFSET = 56;
const MAGIC_V1 = 0x31565241; // "ARV1"
const MAGIC_V2 = 0x32565241; // "ARV2"
const V1_OFFSETS = 104, V1_QUANT_STEP = 2 ** -17;
const V2_STEP = 104, V2_LENGTH = 108, V2_PAYLOAD = 112;

// The header fields, or null when the message is not a motion frame.
export function parseHeader(buf) {
  if (!(buf instanceof ArrayBuffer) || buf.byteLength <= HEADER_BYTES) return null;
  const v = new DataView(buf);
  const magic = v.getUint32(0, true);
  if (magic !== MAGIC_V1 && magic !== MAGIC_V2) return null;
  return {
    version: magic === MAGIC_V1 ? 1 : 2,
    index: v.getUint32(4, true), generation: v.getUint32(8, true), jaw: v.getFloat32(12, true),
    mediaMs: v.getFloat64(16, true) * 1000, sentWallMs: v.getFloat64(24, true), leadMs: v.getFloat32(32, true),
    rms: v.getFloat32(36, true), audioFrameMs: v.getFloat64(40, true) * 1000,
    rtp: v.getUint32(52, true) ? v.getUint32(48, true) : null,
  };
}

async function inflateRaw(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// Adds h.camera and the offsets to a parsed header: h.offsets (int16, one
// x, y, z per vertex) for a version 1 message, h.residuals (encode_offsets'
// planes, inflated) for version 2. Throws on a truncated message.
export async function readMotion(buf, h) {
  h.camera = new Float32Array(buf.slice(CAMERA_OFFSET, CAMERA_OFFSET + 48));
  if (h.version === 1) {
    h.step = V1_QUANT_STEP;
    h.offsets = new Int16Array(buf, V1_OFFSETS, (buf.byteLength - V1_OFFSETS) >> 1);
    return h;
  }
  const v = new DataView(buf);
  h.step = v.getFloat32(V2_STEP, true);
  const length = v.getUint32(V2_LENGTH, true);
  if (V2_PAYLOAD + length > buf.byteLength) throw new Error('truncated motion message');
  h.residuals = await inflateRaw(new Uint8Array(buf, V2_PAYLOAD, length));
  return h;
}

// Mirrors client_render.nearest_earlier_vertices, which says why the two
// must agree exactly: same float64 sums in the same order, ties to the
// lower index.
export function nearestEarlierVertices(headRef) {
  const n = headRef.length / 3, nearest = new Int32Array(n);
  nearest[0] = -1;
  for (let i = 1; i < n; i++) {
    const x = headRef[3 * i], y = headRef[3 * i + 1], z = headRef[3 * i + 2];
    let best = Infinity, arg = 0;
    for (let j = 0; j < i; j++) {
      const dx = x - headRef[3 * j], dy = y - headRef[3 * j + 1], dz = z - headRef[3 * j + 2];
      const d = dx * dx + dy * dy + dz * dz;
      if (d < best) { best = d; arg = j; }
    }
    nearest[i] = arg;
  }
  return nearest;
}

// Writes the head vertices of a message read by readMotion into out
// (Float32Array, x, y, z per vertex). steps is scratch, an Int32Array as
// long as out.
export function headVertices(h, headRef, nearest, out, steps) {
  const n = headRef.length / 3;
  if (h.offsets) {
    if (h.offsets.length !== 3 * n) throw new Error(`motion has ${h.offsets.length / 3} vertices, avatar ${n}`);
    for (let k = 0; k < 3 * n; k++) out[k] = headRef[k] + h.offsets[k] * h.step;
    return out;
  }
  const r = h.residuals;
  if (r.length !== 6 * n) throw new Error(`motion has ${r.length / 6} vertices, avatar ${n}`);
  const hi = 3 * n;
  for (let axis = 0; axis < 3; axis++) {
    for (let i = 0; i < n; i++) {
      const k = axis * n + i, zigzag = r[k] | (r[hi + k] << 8);
      const p = nearest[i];
      const q = ((zigzag >>> 1) ^ -(zigzag & 1)) + (p >= 0 ? steps[3 * p + axis] : 0);
      steps[3 * i + axis] = q;
      out[3 * i + axis] = headRef[3 * i + axis] + q * h.step;
    }
  }
  return out;
}
