"""Draw MediaPipe-style hand skeleton on camera frames."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from HandTrack.landmarks import CONNECTIONS, LANDMARK_COLOR, COLOR_INDEX
from HandTrack.tracker import HandResult


def draw_hands(
    frame: np.ndarray,
    hands: list[HandResult],
    *,
    pen_active: bool = False,
    drawing: bool = False,
    index_tip: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    """Render landmark overlay like the reference photo."""
    h, w = frame.shape[:2]
    out = frame

    for hand in hands:
        pts = []
        for i, lm in enumerate(hand.landmarks):
            x = int(lm[0] * w)
            y = int(lm[1] * h)
            pts.append((x, y))

        # Bones first
        for a, b, color in CONNECTIONS:
            cv2.line(out, pts[a], pts[b], color, 3, cv2.LINE_AA)

        # Joints
        for i, (x, y) in enumerate(pts):
            color = LANDMARK_COLOR.get(i, (200, 200, 200))
            radius = 7 if i in (0, 4, 8, 12, 16, 20) else 5
            cv2.circle(out, (x, y), radius + 2, (30, 30, 30), -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)

        # Handedness tag
        tag = f"{hand.handedness}"
        cv2.putText(
            out, tag, (pts[0][0] - 20, pts[0][1] + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA,
        )

    # Pen cursor
    if index_tip is not None and pen_active:
        cx = int(index_tip[0] * w)
        cy = int(index_tip[1] * h)
        color = (0, 220, 255) if drawing else COLOR_INDEX
        cv2.circle(out, (cx, cy), 14, color, 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 4, color, -1, cv2.LINE_AA)
        if drawing:
            cv2.putText(out, "PEN", (cx + 16, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return out


def draw_hud(
    frame: np.ndarray,
    *,
    message: str,
    pen_active: bool,
    color_index: int,
    fps: float,
) -> np.ndarray:
    """Status strip at the top of the camera view."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 78), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    pen = "PEN ON" if pen_active else "PEN OFF"
    pen_color = (0, 220, 255) if pen_active else (160, 160, 160)
    cv2.putText(frame, "HandTrack", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, pen, (180, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, pen_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, message[:70], (16, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    # Color swatch
    from HandTrack.canvas import PEN_COLORS_BGR
    sw = PEN_COLORS_BGR[color_index % len(PEN_COLORS_BGR)]
    cv2.rectangle(frame, (w - 48, 42), (w - 16, 70), sw, -1)
    cv2.rectangle(frame, (w - 48, 42), (w - 16, 70), (255, 255, 255), 1)
    return frame


def draw_help(frame: np.ndarray) -> np.ndarray:
    lines = [
        "Double-pinch  =  START / STOP pen",
        "Point index   =  write (pen on)",
        "Pinch hold    =  lift stroke (pen stays on)",
        "Open palm     =  clear",
        "Victory       =  next color",
        "Fist          =  pause",
        "Thumbs up     =  save PNG",
        "Q / Esc       =  quit",
    ]
    y = frame.shape[0] - 20 * len(lines) - 12
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (14, y + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA,
        )
    return frame
