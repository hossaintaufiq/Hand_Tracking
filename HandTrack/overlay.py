"""Premium visual overlays — hands, dual selection, HUD, win."""

from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

from HandTrack.landmarks import CONNECTIONS, LANDMARK_COLOR
from HandTrack.tracker import HandResult
from HandTrack import ui


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
    bone_t = 1 if light else 2
    tip_r = 4 if light else 6
    h, w = frame.shape[:2]

    for hand in hands:
        pts: list[tuple[int, int]] = []
        for lm in hand.landmarks:
            x = int(np.clip(lm[0], 0.0, 1.0) * (w - 1))
            y = int(np.clip(lm[1], 0.0, 1.0) * (h - 1))
            pts.append((x, y))

        for a, b, color in CONNECTIONS:
            if not light:
                _dotted_line(frame, pts[a], pts[b], (12, 12, 14), thickness=3, gap=5, dash=9)
            col = tuple(int(c * (0.75 if light else 1.0)) for c in color)
            _dotted_line(frame, pts[a], pts[b], col, thickness=bone_t, gap=5, dash=9)

        tips = {4, 8, 12, 16, 20}
        for i, (x, y) in enumerate(pts):
            color = LANDMARK_COLOR.get(i, (200, 200, 200))
            radius = tip_r if i in tips or i == 0 else 3
            cv2.circle(frame, (x, y), radius + 1, (10, 10, 12), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)

        tag = "L" if hand.handedness.startswith("L") else "R"
        ui.put_text(frame, tag, (pts[0][0] - 6, pts[0][1] + 32), scale=0.5, color=ui.TEXT_MUTED)
    return frame


def draw_hand_cursors(frame: np.ndarray, pointers, *, t: Optional[float] = None) -> np.ndarray:
    h, w = frame.shape[:2]
    now = t if t is not None else time.perf_counter()
    pulse = 0.5 + 0.5 * math.sin(now * 6.0)

    for p in pointers:
        cx, cy = int(p.x * (w - 1)), int(p.y * (h - 1))
        color = ui.ACCENT_HOT if p.pinching else ui.ACCENT
        # Soft glow disc
        glow = frame.copy()
        rad = 28 + int(6 * pulse) if p.pinching else 22
        cv2.circle(glow, (cx, cy), rad, color, -1, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(glow, 0.10 if p.pinching else 0.06, frame, 0.90 if p.pinching else 0.94, 0)

        cv2.circle(frame, (cx, cy), 17, (8, 8, 10), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 16, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)
        if p.pinching:
            cv2.circle(frame, (cx, cy), 24 + int(3 * pulse), color, 1, cv2.LINE_AA)
    return frame


def draw_dual_selection(
    frame: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    locked: bool = False,
    active: bool = False,
    grid: int = 3,
    label: str = "",
) -> np.ndarray:
    h, w = frame.shape[:2]
    xa, xb = sorted((int(x0), int(x1)))
    ya, yb = sorted((int(y0), int(y1)))
    xa, ya = max(0, xa), max(0, ya)
    xb, yb = min(w - 1, xb), min(h - 1, yb)
    if xb - xa < 2 or yb - ya < 2:
        return frame

    color = ui.SUCCESS if locked else (ui.ACCENT_HOT if active else ui.ACCENT)

    # Soft exterior dim (feathered via blur mask)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[ya:yb, xa:xb] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=8)
    dim = (frame.astype(np.float32) * 0.38 + np.array(ui.BG, dtype=np.float32) * 0.62)
    frame_f = frame.astype(np.float32)
    out = frame_f * mask[..., None] + dim * (1.0 - mask[..., None])
    frame[:] = np.clip(out, 0, 255).astype(np.uint8)

    # Inner tint
    tint = frame.copy()
    cv2.rectangle(tint, (xa, ya), (xb, yb), color, -1)
    frame[:] = cv2.addWeighted(tint, 0.06, frame, 0.94, 0)

    # Dual frame lines
    cv2.rectangle(frame, (xa, ya), (xb, yb), color, 2, cv2.LINE_AA)
    if xb - xa > 20 and yb - ya > 20:
        cv2.rectangle(frame, (xa + 4, ya + 4), (xb - 4, yb - 4), (255, 255, 255), 1, cv2.LINE_AA)

    # Corner brackets
    arm = max(16, min(40, (xb - xa) // 7, (yb - ya) // 7))
    for (cx, cy, dx, dy) in (
        (xa, ya, 1, 1), (xb, ya, -1, 1), (xa, yb, 1, -1), (xb, yb, -1, -1),
    ):
        cv2.line(frame, (cx, cy), (cx + dx * arm, cy), color, 3, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * arm), color, 3, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1, cv2.LINE_AA)

    # Soft grid preview
    if grid >= 2 and (xb - xa) > 80 and (yb - ya) > 80:
        tw, th = (xb - xa) / grid, (yb - ya) / grid
        grid_layer = frame.copy()
        for i in range(1, grid):
            x = int(xa + i * tw)
            y = int(ya + i * th)
            cv2.line(grid_layer, (x, ya), (x, yb), color, 1, cv2.LINE_AA)
            cv2.line(grid_layer, (xa, y), (xb, y), color, 1, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(grid_layer, 0.35, frame, 0.65, 0)

    # Floating label chip above selection
    bw, bh = xb - xa, yb - ya
    badge = f"{bw} × {bh}   {grid}×{grid}"
    if label:
        ui.chip(frame, label, xa, max(12, ya - 40), color=color)
    ui.chip(frame, badge, xa, ya + 10, color=ui.TEXT)

    return frame


def draw_hud(
    frame: np.ndarray,
    *,
    title: str,
    message: str,
    fps: float,
    extra: str = "",
    progress: Optional[float] = None,
    progress_display: Optional[float] = None,
    step: int = 1,
    timer: str = "",
) -> tuple[np.ndarray, Optional[float]]:
    h, w = frame.shape[:2]
    top = 88 if progress is None else 96
    frame = ui.glass_panel(frame, (0, 0), (w, top), alpha=0.74, radius=0, accent_top=True)

    cv2.circle(frame, (28, 30), 7, ui.ACCENT, -1, cv2.LINE_AA)
    cv2.circle(frame, (28, 30), 7, (255, 255, 255), 1, cv2.LINE_AA)
    ui.put_text(frame, title, (46, 36), scale=0.72, color=ui.ACCENT, weight=2)

    steps = ("SELECT", "ASSEMBLE", "COMPLETE")
    sx = max(220, w // 2 - 150)
    for i, name in enumerate(steps):
        active = (i + 1) == step
        done = (i + 1) < step
        col = ui.SUCCESS if done else (ui.ACCENT if active else ui.STROKE)
        ui.chip(frame, name, sx + i * 108, 14, color=col, filled=active or done)
        if i < 2:
            cv2.line(
                frame,
                (sx + i * 108 + 92, 28),
                (sx + (i + 1) * 108 - 4, 28),
                ui.STROKE_SOFT, 1, cv2.LINE_AA,
            )

    ui.put_text(frame, f"{fps:.0f} fps", (w - 100, 32), scale=0.48, color=ui.TEXT_MUTED)
    ui.put_text(frame, message[:86], (46, 64), scale=0.48, color=ui.TEXT)
    right = f"{timer}   {extra}".strip() if timer else extra
    if right:
        ui.put_text(frame, right[:40], (w - 220, 64), scale=0.48, color=ui.SUCCESS)

    disp = progress_display
    if progress is not None:
        disp = ui.progress_bar(frame, 46, 78, w - 92, 7, progress, display=disp)
    return frame, disp


def draw_framing_link(frame: np.ndarray, dual, w: int, h: int) -> np.ndarray:
    """Gold link between both pinched index tips while framing."""
    if not (dual.left and dual.right and dual.both_pinching):
        return frame
    a = (int(dual.left.x * (w - 1)), int(dual.left.y * (h - 1)))
    b = (int(dual.right.x * (w - 1)), int(dual.right.y * (h - 1)))
    overlay = frame.copy()
    cv2.line(overlay, a, b, ui.ACCENT, 2, cv2.LINE_AA)
    mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
    cv2.circle(overlay, mid, 5, ui.ACCENT, -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)


def draw_help(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_h = 24 * len(lines) + 28
    x1, y1 = 16, h - panel_h - 16
    x2 = min(w - 16, 460)
    y2 = h - 16
    frame = ui.glass_panel(frame, (x1, y1), (x2, y2), alpha=0.68, radius=16, accent_top=True)
    ui.put_text(frame, "CONTROLS", (x1 + 18, y1 + 22), scale=0.42, color=ui.ACCENT, weight=1)
    for i, line in enumerate(lines):
        ui.put_text(
            frame, line, (x1 + 18, y1 + 46 + i * 24),
            scale=0.46, color=ui.TEXT,
        )
    return frame


def draw_win(frame: np.ndarray, *, t: Optional[float] = None) -> np.ndarray:
    h, w = frame.shape[:2]
    now = t if t is not None else time.perf_counter()
    pulse = 0.5 + 0.5 * math.sin(now * 2.5)

    veil = frame.copy()
    cv2.rectangle(veil, (0, 0), (w, h), (10, 14, 12), -1)
    frame = cv2.addWeighted(veil, 0.42, frame, 0.58, 0)

    cx, cy = w // 2, h // 2
    pw, ph = 520, 150
    frame = ui.glass_panel(
        frame, (cx - pw // 2, cy - ph // 2), (cx + pw // 2, cy + ph // 2),
        alpha=0.82, radius=20, accent_top=True,
    )
    # Soft ring
    cv2.circle(frame, (cx, cy - 10), 70 + int(4 * pulse), ui.SUCCESS, 1, cv2.LINE_AA)
    ui.put_text(frame, "PUZZLE COMPLETE", (cx - 168, cy - 8), scale=1.0, color=ui.SUCCESS, weight=2)
    ui.put_text(
        frame, "N  new capture      R  reshuffle",
        (cx - 168, cy + 36), scale=0.52, color=ui.TEXT_MUTED,
    )
    return frame


def draw_status_chip(frame: np.ndarray, text: str, x: int, y: int, color=ui.ACCENT) -> None:
    ui.chip(frame, text, x, y, color=color)
