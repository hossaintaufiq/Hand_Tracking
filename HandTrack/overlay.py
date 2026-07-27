"""Draw MediaPipe-style dotted hand skeleton on camera frames."""

from __future__ import annotations

import cv2
import numpy as np

from HandTrack.landmarks import CONNECTIONS, LANDMARK_COLOR
from HandTrack.tracker import HandResult


def _dotted_line(
    img: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    gap: int = 6,
    dash: int = 8,
) -> None:
    """Draw a dashed bone segment between two joints."""
    x0, y0 = p0
    x1, y1 = p1
    dist = float(np.hypot(x1 - x0, y1 - y0))
    if dist < 1.0:
        return
    steps = max(1, int(dist // (dash + gap)))
    for i in range(steps + 1):
        t0 = i * (dash + gap) / dist
        t1 = min(1.0, (i * (dash + gap) + dash) / dist)
        if t0 >= 1.0:
            break
        a = (int(x0 + (x1 - x0) * t0), int(y0 + (y1 - y0) * t0))
        b = (int(x0 + (x1 - x0) * t1), int(y0 + (y1 - y0) * t1))
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def draw_hands(frame: np.ndarray, hands: list[HandResult]) -> np.ndarray:
    """Render colored landmark dots + dotted bones (reference style)."""
    h, w = frame.shape[:2]
    out = frame

    for hand in hands:
        pts: list[tuple[int, int]] = []
        for lm in hand.landmarks:
            x = int(np.clip(lm[0], 0.0, 1.0) * (w - 1))
            y = int(np.clip(lm[1], 0.0, 1.0) * (h - 1))
            pts.append((x, y))

        # Soft shadow under bones for readability
        for a, b, color in CONNECTIONS:
            _dotted_line(out, pts[a], pts[b], (25, 25, 25), thickness=4, gap=5, dash=9)
            _dotted_line(out, pts[a], pts[b], color, thickness=2, gap=5, dash=9)

        # Joint dots — tips slightly larger
        tips = {4, 8, 12, 16, 20}
        for i, (x, y) in enumerate(pts):
            color = LANDMARK_COLOR.get(i, (200, 200, 200))
            radius = 6 if i in tips or i == 0 else 4
            cv2.circle(out, (x, y), radius + 2, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)

        label = f"{hand.handedness}  {hand.score:.0%}"
        lx = max(8, pts[0][0] - 24)
        ly = min(h - 8, pts[0][1] + 36)
        cv2.putText(
            out, label, (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA,
        )

    return out


def draw_hud(frame: np.ndarray, *, hands: int, fps: float, message: str = "") -> np.ndarray:
    """Compact status strip."""
    h, w = frame.shape[:2]
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 56), (18, 18, 18), -1)
    frame = cv2.addWeighted(bar, 0.50, frame, 0.50, 0)

    cv2.putText(frame, "HandTrack", (16, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 220, 200), 2, cv2.LINE_AA)
    status = f"{hands} hand" + ("s" if hands != 1 else "")
    cv2.putText(frame, status, (170, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 110, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    if message:
        cv2.putText(frame, message[:72], (16, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    return frame


def draw_help(frame: np.ndarray) -> np.ndarray:
    lines = [
        "Show your hand to the camera",
        "S  =  save snapshot",
        "H  =  toggle help",
        "Q / Esc  =  quit",
    ]
    y = frame.shape[0] - 20 * len(lines) - 12
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (14, y + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA,
        )
    return frame
