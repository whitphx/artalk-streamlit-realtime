"""Avatar picker that shows each face instead of its id."""

from __future__ import annotations

import base64
import io
from collections.abc import Callable

import streamlit as st
import torch
from PIL import Image

THUMBNAIL_PX = 96

# A native radio group rather than st.image tiles under st.button: buttons
# cannot expose which avatar is selected to assistive technology, and only
# the button, not the face, would take the click. The radios bring arrow-key
# navigation and the checked state for free.
_HTML = (
    '<fieldset class="gallery"><legend></legend><div class="tiles"></div></fieldset>'
)

_CSS = """
.gallery { border: 0; margin: 0; padding: 0; min-width: 0; }
.gallery legend {
  padding: 0; margin-bottom: 0.5rem;
  font-family: var(--st-font, sans-serif); font-size: 0.875rem;
  color: var(--st-text-color, inherit);
}
.tiles {
  display: grid; gap: 0.5rem;
  grid-template-columns: repeat(auto-fill, minmax(5.5rem, 1fr));
}
.tile {
  position: relative; display: flex; flex-direction: column; align-items: center;
  gap: 0.25rem; padding: 0.25rem; cursor: pointer;
  border: 2px solid transparent; border-radius: var(--st-base-radius, 0.5rem);
  font-family: var(--st-font, sans-serif); font-size: 0.75rem;
  color: var(--st-text-color, inherit);
}
.tile input { position: absolute; opacity: 0; pointer-events: none; }
.tile img, .tile .placeholder {
  width: 100%; aspect-ratio: 1; object-fit: cover;
  border-radius: calc(var(--st-base-radius, 0.5rem) - 2px);
}
.tile .placeholder {
  display: flex; align-items: center; justify-content: center;
  background: var(--st-secondary-background-color, #eee);
}
.tile .caption {
  max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.tile:has(input:checked) {
  border-color: var(--st-primary-color, #ff4b4b);
  font-weight: 600;
}
.tile:has(input:focus-visible) {
  outline: 2px solid var(--st-primary-color, #ff4b4b); outline-offset: 2px;
}
"""

_JS = """
export default function ({ data, parentElement, setTriggerValue }) {
  const fieldset = parentElement.querySelector(".gallery");
  fieldset.querySelector("legend").textContent = data.legend;
  const tiles = fieldset.querySelector(".tiles");
  const signature = JSON.stringify(data.items);
  if (tiles.dataset.signature !== signature) {
    tiles.dataset.signature = signature;
    tiles.replaceChildren(...data.items.map((item) => {
      const tile = document.createElement("label");
      tile.className = "tile";
      const input = document.createElement("input");
      input.type = "radio";
      input.name = data.name;
      input.value = item.value;
      input.addEventListener("change", () => setTriggerValue("picked", item.value));
      const caption = document.createElement("span");
      caption.className = "caption";
      caption.textContent = item.label;
      let preview;
      if (item.src) {
        preview = document.createElement("img");
        preview.src = item.src;
        preview.alt = item.alt;
        // The alt text already holds the caption.
        caption.setAttribute("aria-hidden", "true");
      } else {
        preview = document.createElement("span");
        preview.className = "placeholder";
        preview.setAttribute("aria-hidden", "true");
        preview.textContent = "3D";
      }
      tile.append(input, preview, caption);
      return tile;
    }));
  }
  for (const input of tiles.querySelectorAll("input")) {
    input.checked = input.value === data.selected;
  }
}
"""

_gallery = st.components.v2.component("avatar_gallery", html=_HTML, css=_CSS, js=_JS)


def avatar_gallery(
    legend: str,
    items: list[dict],
    selected: str,
    *,
    key: str,
    on_pick: Callable[[], None],
) -> None:
    """Render the picker; ``on_pick`` runs before the next script run, with
    the picked value at ``st.session_state[key].picked``.

    Selection lives with the caller, not the component, so the caller can
    change it (to a newly registered avatar, or away from a deleted one).
    """
    _gallery(
        key=key,
        data={"legend": legend, "name": key, "items": items, "selected": selected},
        on_picked_change=on_pick,
    )


def _data_uri(image) -> str:
    """A small JPEG of a tracked entry's ``image``, (3, H, W) in [0, 1].

    The built-in tracked.pt stores it as a NumPy array, the tracker that
    registers uploads as a tensor.
    """
    pixels = (torch.as_tensor(image).clamp(0, 1) * 255).round().to(torch.uint8)
    thumbnail = Image.fromarray(pixels.permute(1, 2, 0).numpy())
    thumbnail.thumbnail((THUMBNAIL_PX, THUMBNAIL_PX))
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


@st.cache_data(show_spinner=False)
def builtin_thumbnails(tracked_path: str | None) -> dict[str, str]:
    """Thumbnails of the built-in avatars, from the aligned crops that
    tracked.pt stores beside each avatar's codes."""
    if not tracked_path:
        return {}
    try:
        tracked = torch.load(tracked_path, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        return {}
    return {
        avatar_id: _data_uri(entry["image"])
        for avatar_id, entry in tracked.items()
        if "image" in entry
    }


# Keyed by modification time too, since a deleted id can be registered
# again with a different face.
@st.cache_data(show_spinner=False, max_entries=256)
def user_thumbnail(tracked_path: str, mtime: float) -> str:
    return _data_uri(
        torch.load(tracked_path, map_location="cpu", weights_only=True)["image"]
    )
