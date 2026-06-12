"""
Lightweight nearest-neighbour centroid tracker.

Justified for this rig: the belt is slow, single-lane, unidirectional and
non-occluding, so the heavy machinery of ByteTrack/BoT-SORT (motion models,
occlusion handling, re-ID) is unnecessary. We just need a stable id per board
so the trigger fires exactly once per physical PCB.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Track:
    id: int
    cx: float
    cy: float
    missing: int = 0
    # 'side' is set by the trigger module; -1 upstream, +1 downstream, 0 unknown
    prev_side: int = 0
    triggered: bool = False


@dataclass
class CentroidTracker:
    max_match_dist: float = 80.0
    max_missing: int = 15
    _next_id: int = 1
    tracks: dict[int, Track] = field(default_factory=dict)

    def update(self, detections: list[tuple[float, float]]) -> dict[int, Track]:
        """detections: list of (cx, cy) pixel centroids for the current frame.

        Returns the current id -> Track map. Detection i is matched to its
        nearest unclaimed track within max_match_dist, else starts a new track.
        """
        unmatched_ids = set(self.tracks.keys())
        matched_ids: set[int] = set()

        for cx, cy in detections:
            best_id, best_d = None, self.max_match_dist
            for tid in unmatched_ids:
                t = self.tracks[tid]
                d = ((t.cx - cx) ** 2 + (t.cy - cy) ** 2) ** 0.5
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is None:
                t = Track(self._next_id, cx, cy)
                self.tracks[self._next_id] = t
                matched_ids.add(self._next_id)
                self._next_id += 1
            else:
                t = self.tracks[best_id]
                t.cx, t.cy, t.missing = cx, cy, 0
                unmatched_ids.discard(best_id)
                matched_ids.add(best_id)

        # Age / retire tracks that were not matched this frame.
        for tid in list(unmatched_ids):
            t = self.tracks[tid]
            t.missing += 1
            if t.missing > self.max_missing:
                del self.tracks[tid]

        return {tid: self.tracks[tid] for tid in matched_ids}
