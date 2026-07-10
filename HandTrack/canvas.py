"""Drawing canvas driven by the index fingertip pen."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


PEN_COLORS_BGR: list[tuple[int, int, int]] = [
    (0, 220, 255),    # gold/yellow
    (255, 120, 40),   # blue
    (80, 200, 80),    # green
    (220, 80, 180),   # purple
    (40, 40, 255),    # red
]


class PenCanvas:
    """Transparent stroke layer composited over the camera frame."""

    def __init__(self, width: int, height: int) -> None:
        self.w = width
        self.h = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._prev: Optional[tuple[int, int]] = None
        self.thickness = 4

    def resize(self, width: int, height: int) -> None:
        if width == self.w and height == self.h:
            return
        self.canvas = cv2.resize(self.canvas, (width, height), interpolation=cv2.INTER_NEAREST)
        self.w, self.h = width, height
        self._prev = None

    def clear(self) -> None:
        self.canvas[:] = 0
        self._prev = None

    def lift(self) -> None:
        """Break the current stroke."""
        self._prev = None

    def draw_to(
        self,
        norm_xy: tuple[float, float],
        color_index: int,
    ) -> None:
        x = int(np.clip(norm_xy[0], 0.0, 1.0) * (self.w - 1))
        y = int(np.clip(norm_xy[1], 0.0, 1.0) * (self.h - 1))
        color = PEN_COLORS_BGR[color_index % len(PEN_COLORS_BGR)]
        if self._prev is not None:
            cv2.line(self.canvas, self._prev, (x, y), color, self.thickness, cv2.LINE_AA)
            cv2.circle(self.canvas, (x, y), max(2, self.thickness // 2), color, -1, cv2.LINE_AA)
        else:
            cv2.circle(self.canvas, (x, y), max(2, self.thickness // 2), color, -1, cv2.LINE_AA)
        self._prev = (x, y)

    def erase_at(self, norm_xy: tuple[float, float], radius: int = 28) -> None:
        x = int(np.clip(norm_xy[0], 0.0, 1.0) * (self.w - 1))
        y = int(np.clip(norm_xy[1], 0.0, 1.0) * (self.h - 1))
        cv2.circle(self.canvas, (x, y), radius, (0, 0, 0), -1)
        self._prev = None

    def composite(self, frame: np.ndarray) -> np.ndarray:
        """Overlay non-black canvas pixels onto frame."""
        if self.canvas.shape[:2] != frame.shape[:2]:
            self.resize(frame.shape[1], frame.shape[0])
        mask = np.any(self.canvas > 0, axis=2)
        out = frame.copy()
        out[mask] = cv2.addWeighted(frame, 0.15, self.canvas, 0.85, 0)[mask]
        return out
