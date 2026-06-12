"""
360-degree heading + marker-based type cross-check for PCB boards.

A rectangular OBB only fixes orientation modulo 180 deg (a rectangle is
symmetric under a half-turn). Each PCB type carries a DISTINCT white silkscreen
box, detected as its own 'marker_*' class by the same YOLO26-OBB model. That
marker is used for two things:

  1. 360-deg heading: the vector board_center -> marker_center tells which half
     is the 'head', so we pick {theta, theta+180} that points toward the marker.
  2. Type cross-check: because each marker class maps to exactly one PCB type,
     a matched marker confirms (or corrects) the board's class.
"""
from __future__ import annotations
import math
import cv2
import numpy as np


def obb_corners(cx: float, cy: float, w: float, h: float, theta_rad: float) -> np.ndarray:
    """Return the 4 corner points (float32, shape (4,2)) of an OBB."""
    return cv2.boxPoints(((cx, cy), (w, h), math.degrees(theta_rad))).astype(np.float32)


def pick_marker(board: tuple, markers: list[tuple], names: dict, marker_map: dict,
                max_dist_px: float | None = None):
    """Pick the marker belonging to `board` and the PCB type it implies.

    board / markers items are (cx, cy, w, h, theta, cls_id, conf).
    Returns (marker_tuple_or_None, inferred_type_or_None).

    Selection rule: among markers whose center lies inside the board OBB, take
    the closest to the board center; fall back to the globally closest marker.
    If max_dist_px is given, discard any candidate farther than that distance
    (prevents cross-board marker association when boards are close together).
    """
    if not markers:
        return None, None
    bx, by, bw, bh, bth = board[0], board[1], board[2], board[3], board[4]
    corners = obb_corners(bx, by, bw, bh, bth)
    inside = [m for m in markers
              if cv2.pointPolygonTest(corners, (float(m[0]), float(m[1])), False) >= 0]
    pool = inside if inside else markers
    m = min(pool, key=lambda mk: (mk[0] - bx) ** 2 + (mk[1] - by) ** 2)
    if max_dist_px is not None:
        dist = math.hypot(m[0] - bx, m[1] - by)
        if dist > max_dist_px:
            return None, None
    inferred = marker_map.get(names[m[5]])
    return m, inferred


def heading_from_marker_vector(board: tuple, marker: tuple | None,
                               offset_deg: float = 0.0) -> float | None:
    """Return heading in [0, 360) as the angle between the board→marker vector
    and the downward reference vector (0, 1) — i.e. top-to-bottom of the camera.

    Convention (clockwise from downward):
        0°   -> marker is directly below board center
        90°  -> marker is to the right
        180° -> marker is directly above board center
        270° -> marker is to the left

    Returns None when marker is None (caller decides on fallback).
    """
    if marker is None:
        return None
    dx = marker[0] - board[0]   # mx - cx  (positive = right)
    dy = marker[1] - board[1]   # my - cy  (positive = down in image coords)
    angle = math.degrees(math.atan2(dx, dy))   # clockwise from down
    return (angle + offset_deg) % 360.0


def resolve_heading_360(board: tuple, marker: tuple | None,
                        offset_deg: float = 0.0, symmetry_deg: float = 180.0) -> float:
    """Return board heading in [0, 360) using the marker to break shape symmetry.

    `symmetry_deg` is the board's rotational symmetry:
        180 -> rectangular board (2 candidate orientations: theta, theta+180)
         90 -> square board       (4 candidates: theta, +90, +180, +270)
    The OBB supplies the precise edge angle; the marker direction selects which
    candidate is the true heading.

    If marker is None, no disambiguation is possible: we fall back to the OBB
    angle folded into [0, symmetry_deg) (then shifted by offset, mod 360).
    """
    bx, by, bth = board[0], board[1], board[4]
    theta = math.degrees(bth)

    if marker is None:
        h = theta % symmetry_deg
        return (h + offset_deg) % 360.0

    to_text = math.degrees(math.atan2(marker[1] - by, marker[0] - bx))  # [-180,180]
    n = max(1, round(360.0 / symmetry_deg))
    best, best_d = theta, 1e9
    for k in range(n):
        cand = theta + k * symmetry_deg
        d = abs(((cand - to_text + 180.0) % 360.0) - 180.0)   # angular distance
        if d < best_d:
            best_d, best = d, cand
    return (best + offset_deg) % 360.0
