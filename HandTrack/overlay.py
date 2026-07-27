"""Draw hand skeleton and game HUD overlays."""

from __future__ import annotations

from typing import Optional

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


def draw_hands(frame: np.ndarray, hands: list[HandResult], *, light: bool = False) -> np.ndarray:
    """Render colored landmark dots + dotted bones."""
    h, w = frame.shape[:2]
    out = frame
    bone_t = 1 if light else 2
    tip_r = 5 if light else 6

    for hand in hands:
        pts: list[tuple[int, int]] = []
        for lm in hand.landmarks:
            x = int(np.clip(lm[0], 0.0, 1.0) * (w - 1))
            y = int(np.clip(lm[1], 0.0, 1.0) * (h - 1))
            pts.append((x, y))

        for a, b, color in CONNECTIONS:
            if not light:
                _dotted_line(out, pts[a], pts[b], (25, 25, 25), thickness=4, gap=5, dash=9)
            _dotted_line(out, pts[a], pts[b], color, thickness=bone_t, gap=5, dash=9)

        tips = {4, 8, 12, 16, 20}
        for i, (x, y) in enumerate(pts):
            color = LANDMARK_COLOR.get(i, (200, 200, 200))
            radius = tip_r if i in tips or i == 0 else 3
            cv2.circle(out, (x, y), radius + 2, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)

    return out


def draw_cursor(
    frame: np.ndarray,
    x: float,
    y: float,
    *,
    pinching: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = int(x * (w - 1)), int(y * (h - 1))
    color = (0, 200, 255) if pinching else (255, 220, 80)
    cv2.circle(frame, (cx, cy), 16, color, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)
    if pinching:
        cv2.putText(frame, "GRAB", (cx + 18, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return frame


def draw_selection(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    locked: bool = False,
) -> np.ndarray:
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    color = (80, 220, 120) if locked else (0, 210, 255)
    overlay = frame.copy()
    cv2.rectangle(overlay, (xa, ya), (xb, yb), color, -1)
    frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
    cv2.rectangle(frame, (xa, ya), (xb, yb), color, 2)
    # Corner handles
    for px, py in ((xa, ya), (xb, ya), (xa, yb), (xb, yb)):
        cv2.circle(frame, (px, py), 6, color, -1, cv2.LINE_AA)
    label = "Selection locked — SPACE to make puzzle" if locked else "Drag with pinch to select"
    cv2.putText(frame, label, (xa, max(24, ya - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return frame


def draw_hud(
    frame: np.ndarray,
    *,
    title: str,
    message: str,
    fps: float,
    extra: str = "",
) -> np.ndarray:
    h, w = frame.shape[:2]
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 64), (16, 16, 16), -1)
    frame = cv2.addWeighted(bar, 0.55, frame, 0.45, 0)
    cv2.putText(frame, title, (16, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 220, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 110, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, message[:78], (16, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)
    if extra:
        cv2.putText(frame, extra[:40], (w - 220, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 255, 180), 1, cv2.LINE_AA)
    return frame


def draw_help(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    y = frame.shape[0] - 20 * len(lines) - 12
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (14, y + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1, cv2.LINE_AA,
        )
    return frame


def draw_win(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (w // 2 - 220, h // 2 - 60), (w // 2 + 220, h // 2 + 60), (20, 40, 20), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
    cv2.putText(frame, "PUZZLE COMPLETE!", (w // 2 - 180, h // 2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (80, 255, 160), 2, cv2.LINE_AA)
    cv2.putText(frame, "N = new puzzle   R = reshuffle", (w // 2 - 170, h // 2 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return frame
