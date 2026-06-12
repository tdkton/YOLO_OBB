"""
One-shot trigger-line crossing detector.

A board fires exactly once: when its tracked centroid transitions from the
UPSTREAM side of the fixed line to the DOWNSTREAM side. Identity comes from the
centroid tracker, so the slow belt cannot produce duplicate packages.
"""
from __future__ import annotations
from src.tracker import Track


class TriggerLine:
    def __init__(self, y_px: int, direction: str = "down"):
        self.y_px = int(y_px)
        # +1 means downstream is the side with LARGER py (belt moves top->bottom)
        self.sign = 1 if direction == "down" else -1

    def _side(self, cy: float) -> int:
        """-1 upstream, +1 downstream, relative to belt direction."""
        delta = (cy - self.y_px) * self.sign
        return 1 if delta > 0 else -1

    def crossed(self, track: Track) -> bool:
        """Update the track's side; return True on the upstream->downstream edge."""
        side = self._side(track.cy)
        crossed = (
            not track.triggered
            and track.prev_side == -1
            and side == 1
        )
        track.prev_side = side
        if crossed:
            track.triggered = True
        return crossed
