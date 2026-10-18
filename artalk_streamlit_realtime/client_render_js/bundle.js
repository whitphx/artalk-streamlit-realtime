// Reader for the .gsb avatar bundle: Gaussians, PCA features and neutral head vertices.

const ctors = { float32: Float32Array, float16: Uint16Array, uint32: Uint32Array, int32: Int32Array, uint8: Uint8Array };

export function parseBundle(buffer) {
  const u8 = new Uint8Array(buffer);
  const magic = String.fromCharCode(...u8.subarray(0, 4));
  if (magic !== 'GSB1') throw new Error(`not a GSB1 bundle (magic ${magic})`);
  const headerLen = new DataView(buffer).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(16, 16 + headerLen)));
  const dataStart = 16 + headerLen;
  const arrays = {};
  for (const a of header.arrays) {
    const C = ctors[a.dtype];
    if (!C) throw new Error(`unsupported dtype ${a.dtype}`);
    arrays[a.name] = { data: new C(buffer, dataStart + a.offset, a.nbytes / C.BYTES_PER_ELEMENT), shape: a.shape };
  }
  return { meta: header.meta, arrays };
}

// Packs the requested channel count as f16 pairs: 16 uses the PCA features,
// 32 the raw ones, 3 the first three raw channels padded to four.
export function packFeatures(bundle, channels) {
  const n = bundle.meta.count;
  const words = Math.ceil(channels / 2);
  const out = new Uint16Array(n * words * 2);
  if (channels === 16) {
    out.set(bundle.arrays.feat16.data);
  } else if (channels === 32) {
    out.set(bundle.arrays.feat32.data);
  } else if (channels === 3) {
    const src = bundle.arrays.feat32.data;
    for (let i = 0; i < n; i++) {
      out[i * 4] = src[i * 32];
      out[i * 4 + 1] = src[i * 32 + 1];
      out[i * 4 + 2] = src[i * 32 + 2];
    }
  } else {
    throw new Error(`channels must be 3, 16 or 32 (got ${channels})`);
  }
  return out;
}

export function f16ToF32(h) {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
