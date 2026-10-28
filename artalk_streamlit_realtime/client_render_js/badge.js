// The "AI-generated" badge, drawn into the pixels of every frame the page
// shows rather than laid over the canvas, so that a screenshot, a recording
// or a capture of the canvas keeps it. ai_badge.py draws the same badge into
// server-rendered frames.

const TEXT = 'AI-generated';

// The badge as an image, and where it goes in a square frame of frameSize px.
export function makeBadge(frameSize) {
  const fontPx = Math.max(10, Math.round(frameSize * 0.036));
  const font = `${fontPx}px system-ui, sans-serif`;
  const probe = new OffscreenCanvas(1, 1).getContext('2d');
  probe.font = font;
  const m = probe.measureText(TEXT);
  const padX = Math.round(fontPx * 0.5), padY = Math.round(fontPx * 0.3);
  const w = Math.ceil(m.width) + 2 * padX;
  const h = Math.ceil(m.actualBoundingBoxAscent + m.actualBoundingBoxDescent) + 2 * padY;
  const image = new OffscreenCanvas(w, h);
  const ctx = image.getContext('2d');
  // Black at this opacity under white text keeps the text readable on a
  // pale face as well as on the black background.
  ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
  ctx.beginPath();
  ctx.roundRect(0, 0, w, h, h * 0.3);
  ctx.fill();
  ctx.font = font;
  ctx.fillStyle = '#fff';
  ctx.fillText(TEXT, padX, padY + m.actualBoundingBoxAscent);
  const margin = Math.round(frameSize * 0.02);
  return { image, x: frameSize - margin - w, y: frameSize - margin - h, w, h };
}
