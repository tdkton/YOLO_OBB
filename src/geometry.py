"""
Coordinate / angle helpers: pixel -> millimetre transform and OBB angle
normalization.
"""
from __future__ import annotations
import numpy as np


class PixelToMM:
    """Maps image pixels to table millimetres via a 3x3 homography.

    If the homography is (near) identity the system is treated as uncalibrated
    and a simple scalar `pixels_per_mm` fallback is used instead.
    """

    def __init__(self, homography, pixels_per_mm: float = 12.0):
        self.H = np.array(homography, dtype=np.float64).reshape(3, 3)
        self.pixels_per_mm = float(pixels_per_mm)
        self._identity = np.allclose(self.H, np.eye(3))

    def transform(self, px: float, py: float) -> tuple[float, float]:
        """Return (X_mm, Y_mm) for an image point (px, py)."""
        if self._identity:
            # Uncalibrated fallback: pure scale, origin at top-left.
            return px / self.pixels_per_mm, py / self.pixels_per_mm
        v = self.H @ np.array([px, py, 1.0])
        w = v[2] if abs(v[2]) > 1e-12 else 1e-12
        return float(v[0] / w), float(v[1] / w)


def normalize_angle_deg(theta_rad: float) -> float:
    """Map an OBB angle (radians) to degrees in the range [-90, 90).

    A rectangular board is symmetric under 180 deg rotation, so we fold the
    angle into a half-open 180 deg window centered on 0 - the smallest rotation
    a gripper must apply.
    """
    deg = np.degrees(theta_rad)
    deg = (deg + 90.0) % 180.0 - 90.0
    return float(deg)
