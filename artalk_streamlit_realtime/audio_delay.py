"""Control for the delay a listener's audio device adds after the browser."""

from __future__ import annotations

import streamlit as st

BLUETOOTH_MS = 200
MAX_MS = 1000

_HTML = f"""
<div class="delay">
  <label>Audio device delay
    <select aria-describedby="delay-hint">
      <option value="none">None</option>
      <option value="bluetooth">Bluetooth headphones (about {BLUETOOTH_MS} ms)</option>
      <option value="custom">Custom</option>
    </select>
  </label>
  <label class="custom" hidden>Delay in milliseconds
    <input type="number" min="0" max="{MAX_MS}" step="10" inputmode="numeric">
  </label>
  <p id="delay-hint">Holds the face back to match sound your speakers or
  headphones play late. Browsers do not report the delay Bluetooth adds.</p>
</div>
"""

_CSS = """
.delay {
  display: flex; flex-direction: column; gap: 0.5rem;
  font-family: var(--st-font, sans-serif); font-size: 0.875rem;
  color: var(--st-text-color, inherit);
}
.delay label { display: flex; flex-direction: column; gap: 0.25rem; }
.delay label[hidden] { display: none; }
.delay select, .delay input {
  box-sizing: border-box; width: 100%; max-width: 20rem; padding: 0.375rem 0.5rem;
  font: inherit; color: inherit;
  background: var(--st-secondary-background-color, transparent);
  border: 1px solid var(--st-border-color, #ccc);
  border-radius: var(--st-base-radius, 0.5rem);
}
.delay select:focus-visible, .delay input:focus-visible {
  outline: 2px solid var(--st-primary-color, #ff4b4b); outline-offset: 2px;
}
.delay p { margin: 0; opacity: 0.75; font-size: 0.8125rem; }
"""

# The choice is kept in the browser, since it belongs to the listener's
# device rather than to an account.
_JS = """
const STORE_KEY = "artalk.audioDeviceDelay";

export default function ({ data, parentElement, setStateValue }) {
  const select = parentElement.querySelector("select");
  const customLabel = parentElement.querySelector(".custom");
  const input = customLabel.querySelector("input");
  const delayMs = () => {
    if (select.value === "bluetooth") return data.bluetoothMs;
    if (select.value === "custom") return Math.min(data.maxMs, Math.max(0, Math.round(Number(input.value) || 0)));
    return 0;
  };
  const sync = () => {
    customLabel.hidden = select.value !== "custom";
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({ choice: select.value, customMs: Number(input.value) || 0 }));
    } catch {}
  };
  if (select.dataset.ready) return;
  select.dataset.ready = "1";
  let saved = null;
  try {
    saved = JSON.parse(localStorage.getItem(STORE_KEY));
  } catch {}
  if (saved && [...select.options].some((o) => o.value === saved.choice)) {
    select.value = saved.choice;
    input.value = String(saved.customMs ?? 0);
  } else if (data.delayMs) {
    // Storage can be blocked, as in a private window, while the session
    // still holds the choice made before the control was last mounted.
    select.value = data.delayMs === data.bluetoothMs ? "bluetooth" : "custom";
    input.value = String(data.delayMs);
  }
  sync();
  // Only a remembered choice that differs needs a rerun on mount.
  if (delayMs() !== data.delayMs) setStateValue("delay_ms", delayMs());
  const onChange = () => {
    sync();
    setStateValue("delay_ms", delayMs());
  };
  select.addEventListener("change", onChange);
  input.addEventListener("change", onChange);
}
"""

_control = st.components.v2.component("audio_device_delay", html=_HTML, css=_CSS, js=_JS)


def audio_device_delay(*, key: str) -> int:
    """Render the control and return the delay the listener chose, in ms."""
    delay_ms = int((st.session_state.get(key) or {}).get("delay_ms") or 0)
    _control(
        key=key,
        data={"delayMs": delay_ms, "bluetoothMs": BLUETOOTH_MS, "maxMs": MAX_MS},
        default={"delay_ms": 0},
        on_delay_ms_change=lambda: None,
    )
    return delay_ms
