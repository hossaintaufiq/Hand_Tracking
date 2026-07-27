"""Jigsaw puzzle board: slice a selected image into movable tiles."""

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
    image: np.ndarray          # BGR tile
    x: float                   # top-left on board (pixels)
    y: float
    target_x: float
    target_y: float
    w: int
    h: int
    placed: bool = False
    grab_dx: float = 0.0
    grab_dy: float = 0.0

    @property
    def cx(self) -> float:
        return self.x + self.w * 0.5

    @property
    def cy(self) -> float:
        return self.y + self.h * 0.5

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


@dataclass
class JigsawPuzzle:
    """Scrambled rectangular tiles from a source crop."""

    source: np.ndarray
    rows: int
    cols: int
    board_x: int
    board_y: int
    board_w: int
    board_h: int
    pieces: list[Piece] = field(default_factory=list)
    held: Optional[int] = None
    snap_px: float = 28.0
    completed: bool = False

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
        src = cv2.resize(image, (board_w, board_h), interpolation=cv2.INTER_AREA)
        # Make dimensions divisible
        tw = board_w // cols
        th = board_h // rows
        usable_w = tw * cols
        usable_h = th * rows
        src = src[:usable_h, :usable_w]
        board_w, board_h = usable_w, usable_h

        pieces: list[Piece] = []
        for r in range(rows):
            for c in range(cols):
                tile = src[r * th:(r + 1) * th, c * tw:(c + 1) * tw].copy()
                # Soft edge so pieces read as tiles
                cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), (240, 240, 240), 1)
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
            snap_px=max(22.0, min(tw, th) * 0.28),
        )
        puzzle.shuffle()
        return puzzle

    def shuffle(self) -> None:
        self.completed = False
        self.held = None
        margin = 16
        for p in self.pieces:
            p.placed = False
            p.x = float(random.randint(margin, max(margin, self.board_x + self.board_w - p.w - margin)))
            p.y = float(random.randint(margin, max(margin, self.board_y + self.board_h - p.h - margin)))
            # Nudge away from exact target so it isn't already solved
            if abs(p.x - p.target_x) < self.snap_px and abs(p.y - p.target_y) < self.snap_px:
                p.x = float(random.randint(margin, max(margin, self.board_w - p.w)))
                p.y = float(random.randint(margin, max(margin, self.board_h - p.h)))

    @property
    def placed_count(self) -> int:
        return sum(1 for p in self.pieces if p.placed)

    @property
    def total(self) -> int:
        return len(self.pieces)

    def pick(self, px: float, py: float) -> bool:
        """Grab topmost unplaced piece under the pointer."""
        if self.held is not None:
            return True
        # Reverse order so last-drawn (top) is preferred
        for i in range(len(self.pieces) - 1, -1, -1):
            p = self.pieces[i]
            if p.placed:
                continue
            if p.contains(px, py):
                p.grab_dx = px - p.x
                p.grab_dy = py - p.y
                self.held = i
                # Bring to front
                self.pieces.append(self.pieces.pop(i))
                self.held = len(self.pieces) - 1
                return True
        return False

    def drag(self, px: float, py: float, frame_w: int, frame_h: int) -> None:
        if self.held is None:
            return
        p = self.pieces[self.held]
        p.x = float(np.clip(px - p.grab_dx, 0, frame_w - p.w))
        p.y = float(np.clip(py - p.grab_dy, 0, frame_h - p.h))

    def drop(self) -> bool:
        """Release held piece; snap if close to target. Returns True if newly placed."""
        if self.held is None:
            return False
        p = self.pieces[self.held]
        self.held = None
        if abs(p.x - p.target_x) <= self.snap_px and abs(p.y - p.target_y) <= self.snap_px:
            p.x = p.target_x
            p.y = p.target_y
            p.placed = True
            self.completed = all(q.placed for q in self.pieces)
            return True
        return False

    def draw(self, frame: np.ndarray, *, show_guides: bool = True) -> np.ndarray:
        out = frame
        bx, by, bw, bh = self.board_x, self.board_y, self.board_w, self.board_h

        # Dim board well + ghost outline
        overlay = out.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (30, 30, 30), -1)
        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (90, 200, 180), 2)

        if show_guides:
            tw = bw // self.cols
            th = bh // self.rows
            for r in range(1, self.rows):
                y = by + r * th
                cv2.line(out, (bx, y), (bx + bw, y), (70, 70, 70), 1)
            for c in range(1, self.cols):
                x = bx + c * tw
                cv2.line(out, (x, by), (x, by + bh), (70, 70, 70), 1)

        # Draw placed first, then free, held last (already ordered with held at end)
        for i, p in enumerate(self.pieces):
            x, y = int(p.x), int(p.y)
            h, w = p.image.shape[:2]
            if y + h > out.shape[0] or x + w > out.shape[1] or x < 0 or y < 0:
                continue
            roi = out[y:y + h, x:x + w]
            out[y:y + h, x:x + w] = p.image
            color = (80, 220, 120) if p.placed else ((0, 210, 255) if i == self.held else (220, 220, 220))
            cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), color, 2 if i == self.held else 1)

        return out
