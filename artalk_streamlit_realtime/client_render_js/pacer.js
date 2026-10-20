// Frame scheduling for the client renderer, apart from the DOM and the GPU
// so it can run in a simulation: frames are rendered as soon as they arrive
// into a ring of slots, and each animation frame presents the one that is
// due.

export const RING_SLOTS = 8;
export const MAX_QUEUE = 250;
// How fast the hold may move, in ms per ms of wall time. A jump in the hold
// would show as a jump in the face; 5% is a speed change nobody sees.
export const HOLD_SLEW = 0.05;

export function slewToward(current, target, dtMs, rate = HOLD_SLEW) {
  const step = rate * Math.max(0, dtMs);
  return current + Math.max(-step, Math.min(step, target - current));
}

function insertByMedia(list, item, media) {
  let i = list.length;
  while (i > 0 && media(list[i - 1]) > media(item)) i--;
  list.splice(i, 0, item);
}

// counts: the session's counter object, updated in place.
export function createPacer(counts, { slots = RING_SLOTS, periodMs = 40, maxQueue = MAX_QUEUE } = {}) {
  const queue = [];
  const ring = [];
  let free = [...Array(slots).keys()];
  let shownSlot = null, lastShownMedia = -Infinity, ringWasFull = false;

  const releaseSlot = (slot) => { if (slot !== shownSlot) free.push(slot); };

  return {
    get queued() { return queue.length; },
    get ready() { return ring.length; },

    add(h) {
      if (h.mediaMs <= lastShownMedia) { counts.skippedLate++; return; }
      insertByMedia(queue, h, (x) => x.mediaMs);
      if (queue.length > maxQueue) { queue.shift(); counts.queueOverflow++; }
    },

    // The next frame to render and the slot it goes into, or null when there
    // is nothing to render or no free slot.
    takeNext(now, dueAt, generation) {
      while (queue.length) {
        const head = queue[0];
        if (head.generation !== generation) { queue.shift(); counts.skippedOtherAvatar++; continue; }
        // A newer frame is queued behind it, so rendering this one would only
        // delay that one.
        if (queue.length > 1 && dueAt(head) < now - periodMs) {
          queue.shift();
          if (ringWasFull) counts.droppedRingFull++; else counts.skippedLate++;
          continue;
        }
        if (!free.length) {
          // A full ring holds the frames due soonest, so rendering further
          // ahead can wait. Dropping one instead would lose a frame that is
          // on time, since frames often arrive further ahead than the ring
          // reaches. Only a late arrival due before the ring's last frame
          // takes that frame's slot, and that frame is rendered again later.
          const last = ring.at(-1);
          if (last && last.h.mediaMs > head.mediaMs) {
            ring.pop();
            free.push(last.slot);
            insertByMedia(queue, last.h, (x) => x.mediaMs);
            counts.requeuedRingFull++;
            continue;
          }
          ringWasFull = true;
          return null;
        }
        ringWasFull = false;
        queue.shift();
        return { h: head, slot: free.pop() };
      }
      return null;
    },

    finish(job, now, dueAt) {
      counts.rendered++;
      if (job.h.mediaMs <= lastShownMedia) { counts.skippedLate++; releaseSlot(job.slot); return; }
      if (dueAt(job.h) > now) counts.renderedAhead++;
      insertByMedia(ring, job, (x) => x.h.mediaMs);
    },

    cancel(job) { releaseSlot(job.slot); },

    // The frame to show on the animation frame at `now`, or null to keep the
    // current one.
    present(now, dueAt) {
      while (ring.length > 1 && dueAt(ring[0].h) < now - periodMs && dueAt(ring[1].h) <= now) {
        releaseSlot(ring.shift().slot);
        counts.skippedOverdue++;
      }
      if (!ring.length || dueAt(ring[0].h) > now) return null;
      const job = ring.shift();
      if (shownSlot !== null) free.push(shownSlot);
      shownSlot = job.slot;
      lastShownMedia = job.h.mediaMs;
      counts.presented++;
      return job;
    },

    // Drops every frame not yet shown; returns how many.
    clear() {
      const n = queue.length + ring.length;
      queue.length = 0;
      for (const job of ring.splice(0)) releaseSlot(job.slot);
      return n;
    },

    // For a new set of slots: rendered frames go back to be rendered again.
    reset() {
      for (const job of ring.splice(0)) insertByMedia(queue, job.h, (x) => x.mediaMs);
      free = [...Array(slots).keys()];
      shownSlot = null;
      ringWasFull = false;
    },
  };
}
