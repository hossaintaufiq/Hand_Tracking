"""Premium jigsaw board with dual-hand piece control."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class Piece:
    row: int
    col: int
    image: np.ndarray
    x: float
    y: float
    target_x: float
    target_y: float
    w: int
    h: int
    placed: bool = False
    grab_dx: float = 0.0
    grab_dy: float = 0.0
    holder: Optional[str] = None  # hand key holding this piece


@dataclass
class JigsawPuzzle:
    source: np.ndarray
    rows: int
    cols: int
    board_x: int
    board_y: int
    board_w: int
    board_h: int
    pieces: list[Piece] = field(default_factory=list)
    held: dict[str, int] = field(default_factory=dict)  # hand_key -> piece index
    snap_px: float = 28.0
    completed: bool = False
    last_snap: bool = False

    @classmethod
    def from_image(
        cls,
        image: np.ndarray,
        rows: int,
        cols: int,
        board_x: int,
        board_y: int,
        board_w: int,
        board_h: int,
    ) -> "JigsawPuzzle":
        # High-quality downscale when shrinking; cubic when enlarging
        ih, iw = image.shape[:2]
        interp = cv2.INTER_AREA if (board_w < iw or board_h < ih) else cv2.INTER_CUBIC
        src = cv2.resize(image, (board_w, board_h), interpolation=interp)
        tw = board_w // cols
        th = board_h // rows
        usable_w, usable_h = tw * cols, th * rows
        src = src[:usable_h, :usable_w]
        board_w, board_h = usable_w, usable_h

        pieces: list[Piece] = []
        for r in range(rows):
            for c in range(cols):
                tile = src[r * th:(r + 1) * th, c * tw:(c + 1) * tw].copy()
                # Subtle inner bevel
                cv2.rectangle(tile, (1, 1), (tw - 2, th - 2), (255, 255, 255), 1)
                cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), (40, 40, 40), 1)
                pieces.append(
                    Piece(
                        row=r,
                        col=c,
                        image=tile,
                        x=0.0,
                        y=0.0,
                        target_x=float(board_x + c * tw),
                        target_y=float(board_y + r * th),
                        w=tw,
                        h=th,
                    )
                )

        puzzle = cls(
            source=src,
            rows=rows,
            cols=cols,
            board_x=board_x,
            board_y=board_y,
            board_w=board_w,
            board_h=board_h,
            pieces=pieces,
            snap_px=max(24.0, min(tw, th) * 0.32),
        )
        puzzle.shuffle()
        return puzzle

    def shuffle(self) -> None:
        self.completed = False
        self.held.clear()
        self.last_snap = False
        margin = 20
        span_x = max(margin, self.board_x + self.board_w - margin)
        span_y = max(margin, self.board_y + self.board_h - margin)
        for p in self.pieces:
            p.placed = False
            p.holder = None
            for _ in range(8):
                p.x = float(random.randint(margin, max(margin, span_x - p.w)))
                p.y = float(random.randint(margin, max(margin, span_y - p.h)))
                if abs(p.x - p.target_x) > self.snap_px or abs(p.y - p.target_y) > self.snap_px:
                    break

    @property
    def placed_count(self) -> int:
        return sum(1 for p in self.pieces if p.placed)

    @property
    def total(self) -> int:
        return len(self.pieces)

    def pick(self, hand_key: str, px: float, py: float) -> bool:
        if hand_key in self.held:
            return True
        for i in range(len(self.pieces) - 1, -1, -1):
            p = self.pieces[i]
            if p.placed or p.holder is not None:
                continue
            if p.x <= px <= p.x + p.w and p.y <= py <= p.y + p.h:
                p.grab_dx = px - p.x
                p.grab_dy = py - p.y
                p.holder = hand_key
                self.pieces.append(self.pieces.pop(i))
                self.held[hand_key] = len(self.pieces) - 1
                return True
        return False

    def drag(self, hand_key: str, px: float, py: float, frame_w: int, frame_h: int) -> None:
        idx = self.held.get(hand_key)
        if idx is None:
            return
        # Refresh index if list was reordered by another hand
        idx = next((i for i, p in enumerate(self.pieces) if p.holder == hand_key), None)
        if idx is None:
            self.held.pop(hand_key, None)
            return
        self.held[hand_key] = idx
        p = self.pieces[idx]
        p.x = float(np.clip(px - p.grab_dx, 0, max(0, frame_w - p.w)))
        p.y = float(np.clip(py - p.grab_dy, 0, max(0, frame_h - p.h)))

    def drop(self, hand_key: str) -> bool:
        self.last_snap = False
        idx = next((i for i, p in enumerate(self.pieces) if p.holder == hand_key), None)
        self.held.pop(hand_key, None)
        if idx is None:
            return False
        p = self.pieces[idx]
        p.holder = None
        if abs(p.x - p.target_x) <= self.snap_px and abs(p.y - p.target_y) <= self.snap_px:
            p.x = p.target_x
            p.y = p.target_y
            p.placed = True
            self.last_snap = True
            self.completed = all(q.placed for q in self.pieces)
            return True
        return False

    def near_slot(self, hand_key: str) -> bool:
        idx = next((i for i, p in enumerate(self.pieces) if p.holder == hand_key), None)
        if idx is None:
            return False
        p = self.pieces[idx]
        return abs(p.x - p.target_x) <= self.snap_px and abs(p.y - p.target_y) <= self.snap_px

    def draw(self, frame: np.ndarray, *, show_guides: bool = True) -> np.ndarray:
        out = frame
        bx, by, bw, bh = self.board_x, self.board_y, self.board_w, self.board_h

        # Board well with soft inset
        well = out.copy()
        cv2.rectangle(well, (bx - 6, by - 6), (bx + bw + 6, by + bh + 6), (12, 14, 18), -1)
        cv2.rectangle(well, (bx, by), (bx + bw, by + bh), (28, 30, 36), -1)
        out = cv2.addWeighted(well, 0.55, out, 0.45, 0)
        cv2.rectangle(out, (bx - 1, by - 1), (bx + bw, by + bh), (120, 190, 170), 2)
        cv2.rectangle(out, (bx - 4, by - 4), (bx + bw + 3, by + bh + 3), (60, 80, 75), 1)

        # Ghost reference at low opacity
        ghost = out.copy()
        ghost[by:by + bh, bx:bx + bw] = self.source
        out = cv2.addWeighted(ghost, 0.12, out, 0.88, 0)

        if show_guides:
            tw, th = bw // self.cols, bh // self.rows
            for r in range(1, self.rows):
                y = by + r * th
                cv2.line(out, (bx, y), (bx + bw, y), (55, 60, 65), 1, cv2.LINE_AA)
            for c in range(1, self.cols):
                x = bx + c * tw
                cv2.line(out, (x, by), (x, by + bh), (55, 60, 65), 1, cv2.LINE_AA)

        # Drop shadow pass for free pieces
        shadow = out.copy()
        for p in self.pieces:
            if p.placed:
                continue
            sx, sy = int(p.x) + 4, int(p.y) + 5
            if 0 <= sx < out.shape[1] - p.w and 0 <= sy < out.shape[0] - p.h:
                cv2.rectangle(shadow, (sx, sy), (sx + p.w, sy + p.h), (0, 0, 0), -1)
        out = cv2.addWeighted(shadow, 0.22, out, 0.78, 0)

        for p in self.pieces:
            x, y = int(p.x), int(p.y)
            if x < 0 or y < 0 or x + p.w > out.shape[1] or y + p.h > out.shape[0]:
                continue
            out[y:y + p.h, x:x + p.w] = p.image
            if p.placed:
                color, thick = (90, 210, 140), 1
            elif p.holder:
                color, thick = (0, 200, 255), 3
            else:
                color, thick = (200, 205, 210), 1
            cv2.rectangle(out, (x, y), (x + p.w - 1, y + p.h - 1), color, thick, cv2.LINE_AA)

        return out
