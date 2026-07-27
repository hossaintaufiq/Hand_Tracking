"""Premium visual overlays — hands, dual selection, HUD, win."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from HandTrack.landmarks import CONNECTIONS, LANDMARK_COLOR
from HandTrack.tracker import HandResult


# Brand palette (BGR)
ACCENT = (180, 200, 90)       # soft teal-gold
ACCENT_HOT = (120, 210, 255)  # warm highlight
OK = (110, 210, 150)
LOCKED = (90, 200, 120)
DIM = (18, 18, 22)


def _dotted_line(
    img: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    gap: int = 5,
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
    h, w = frame.shape[:2]
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
                _dotted_line(frame, pts[a], pts[b], (20, 20, 20), thickness=4, gap=5, dash=9)
            _dotted_line(frame, pts[a], pts[b], color, thickness=bone_t, gap=5, dash=9)

        tips = {4, 8, 12, 16, 20}
        for i, (x, y) in enumerate(pts):
            color = LANDMARK_COLOR.get(i, (200, 200, 200))
            radius = tip_r if i in tips or i == 0 else 3
            cv2.circle(frame, (x, y), radius + 2, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)

        tag = hand.handedness[0]  # L / R
        cv2.putText(
            frame, tag, (pts[0][0] - 8, pts[0][1] + 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA,
        )
    return frame


def draw_hand_cursors(frame: np.ndarray, pointers) -> np.ndarray:
    """Draw per-hand index cursors with pinch rings."""
    h, w = frame.shape[:2]
    for p in pointers:
        cx, cy = int(p.x * (w - 1)), int(p.y * (h - 1))
        color = ACCENT_HOT if p.pinching else ACCENT
        cv2.circle(frame, (cx, cy), 18, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 18, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)
        if p.pinching:
            cv2.circle(frame, (cx, cy), 26, color, 1, cv2.LINE_AA)
    return frame


def draw_dual_selection(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    locked: bool = False,
    active: bool = False,
    grid: int = 3,
    label: str = "",
) -> np.ndarray:
    """Premium crop marquee with dimmed exterior + L-corners + grid preview."""
    h, w = frame.shape[:2]
    xa, xb = sorted((int(x0), int(x1)))
    ya, yb = sorted((int(y0), int(y1)))
    xa, ya = max(0, xa), max(0, ya)
    xb, yb = min(w - 1, xb), min(h - 1, yb)
    if xb - xa < 2 or yb - ya < 2:
        return frame

    color = LOCKED if locked else (ACCENT_HOT if active else ACCENT)

    # Dim outside selection
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[ya:yb, xa:xb] = 255
    dimmed = frame.copy()
    dimmed = cv2.addWeighted(dimmed, 0.35, np.full_like(dimmed, DIM), 0.65, 0)
    frame = np.where(mask[..., None] == 255, frame, dimmed)

    # Soft fill inside
    tint = frame.copy()
    cv2.rectangle(tint, (xa, ya), (xb, yb), color, -1)
    frame = cv2.addWeighted(tint, 0.08, frame, 0.92, 0)

    # Outer + inner frame
    cv2.rectangle(frame, (xa, ya), (xb, yb), color, 2, cv2.LINE_AA)
    if xb - xa > 16 and yb - ya > 16:
        cv2.rectangle(frame, (xa + 3, ya + 3), (xb - 3, yb - 3), (255, 255, 255), 1, cv2.LINE_AA)

    # L-shaped corner brackets
    arm = max(14, min(36, (xb - xa) // 8, (yb - ya) // 8))
    for (cx, cy, dx, dy) in (
        (xa, ya, 1, 1), (xb, ya, -1, 1), (xa, yb, 1, -1), (xb, yb, -1, -1),
    ):
        cv2.line(frame, (cx, cy), (cx + dx * arm, cy), color, 3, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * arm), color, 3, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 5, color, -1, cv2.LINE_AA)

    # Grid preview inside selection
    if grid >= 2 and (xb - xa) > 60 and (yb - ya) > 60:
        tw, th = (xb - xa) / grid, (yb - ya) / grid
        for i in range(1, grid):
            x = int(xa + i * tw)
            y = int(ya + i * th)
            cv2.line(frame, (x, ya), (x, yb), color, 1, cv2.LINE_AA)
            cv2.line(frame, (xa, y), (xb, y), color, 1, cv2.LINE_AA)

    # Size badge
    bw, bh = xb - xa, yb - ya
    badge = f"{bw}×{bh}  ·  {grid}×{grid}"
    cv2.putText(frame, badge, (xa + 10, ya + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    if label:
        cv2.putText(frame, label, (xa + 10, max(28, ya - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
    return frame


def draw_hud(
    frame: np.ndarray,
    *,
    title: str,
    message: str,
    fps: float,
    extra: str = "",
    progress: Optional[float] = None,
) -> np.ndarray:
    h, w = frame.shape[:2]
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 72), (10, 12, 14), -1)
    frame = cv2.addWeighted(bar, 0.62, frame, 0.38, 0)
    # Accent rule
    cv2.line(frame, (0, 72), (w, 72), ACCENT, 1, cv2.LINE_AA)

    cv2.putText(frame, title, (22, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, ACCENT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 108, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 175, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, message[:82], (22, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 225, 230), 1, cv2.LINE_AA)
    if extra:
        cv2.putText(frame, extra[:36], (w - 200, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, OK, 1, cv2.LINE_AA)

    if progress is not None:
        pw = int((w - 44) * np.clip(progress, 0, 1))
        cv2.rectangle(frame, (22, 66), (w - 22, 70), (40, 44, 48), -1)
        if pw > 0:
            cv2.rectangle(frame, (22, 66), (22 + pw, 70), ACCENT, -1)
    return frame


def draw_help(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_h = 22 * len(lines) + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, h - panel_h - 12), (420, h - 12), (12, 14, 16), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    cv2.rectangle(frame, (12, h - panel_h - 12), (420, h - 12), (60, 70, 75), 1, cv2.LINE_AA)
    y0 = h - panel_h + 6
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (26, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 215, 220), 1, cv2.LINE_AA,
        )
    return frame


def draw_win(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (8, 16, 12), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
    cx, cy = w // 2, h // 2
    cv2.rectangle(frame, (cx - 260, cy - 70), (cx + 260, cy + 70), (16, 28, 22), -1)
    cv2.rectangle(frame, (cx - 260, cy - 70), (cx + 260, cy + 70), OK, 2, cv2.LINE_AA)
    cv2.putText(frame, "PUZZLE COMPLETE", (cx - 195, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 1.05, OK, 2, cv2.LINE_AA)
    cv2.putText(frame, "N  new capture     R  reshuffle", (cx - 175, cy + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 225, 230), 1, cv2.LINE_AA)
    return frame


def draw_status_chip(frame: np.ndarray, text: str, x: int, y: int, color=ACCENT) -> None:
    tw = 11 * len(text) + 24
    cv2.rectangle(frame, (x, y), (x + tw, y + 28), (16, 18, 20), -1)
    cv2.rectangle(frame, (x, y), (x + tw, y + 28), color, 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x + 12, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
