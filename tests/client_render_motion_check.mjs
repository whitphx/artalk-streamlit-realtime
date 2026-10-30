// Reads motion messages with the page's motion.js, for test_client_render_motion.py.
//   node client_render_motion_check.mjs <dir>
// <dir>/head_ref.f32 holds the neutral vertices and <dir>/<k>.msg the
// messages. Writes <dir>/nearest.i32 and, per message, <dir>/<k>.f32 with
// the head vertices and <dir>/<k>.json with the header fields.
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseHeader, readMotion, nearestEarlierVertices, headVertices } from '../artalk_streamlit_realtime/client_render_js/motion.js';

const dir = process.argv[2];
const bytes = (name) => { const b = readFileSync(join(dir, name)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const headRef = new Float32Array(bytes('head_ref.f32'));
const nearest = nearestEarlierVertices(headRef);
writeFileSync(join(dir, 'nearest.i32'), new Uint8Array(nearest.buffer));
for (const name of readdirSync(dir).filter((f) => f.endsWith('.msg'))) {
  const buf = bytes(name);
  const h = parseHeader(buf);
  await readMotion(buf, h);
  const out = headVertices(h, headRef, nearest, new Float32Array(headRef.length), new Int32Array(headRef.length));
  const stem = name.slice(0, -4);
  writeFileSync(join(dir, `${stem}.f32`), new Uint8Array(out.buffer));
  const { version, index, generation, jaw, step } = h;
  writeFileSync(join(dir, `${stem}.json`), JSON.stringify({ version, index, generation, jaw, step, camera: [...h.camera] }));
}
