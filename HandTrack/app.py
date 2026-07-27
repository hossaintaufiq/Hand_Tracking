"""HandTrack Studio — premium two-hand jigsaw from a live camera crop."""

from __future__ import annotations

import time
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from HandTrack.jigsaw import JigsawPuzzle
from HandTrack.overlay import (
    draw_dual_selection,
    draw_hand_cursors,
    draw_hands,
    draw_help,
    draw_hud,
    draw_status_chip,
    draw_win,
)
from HandTrack.pointer import DualPointerEngine, DualPointerState
from HandTrack.tracker import HandTracker


class Mode(Enum):
    SELECT = auto()
    PLAY = auto()
    WIN = auto()


class HandTrackApp:
    """Two-hand framing + dual-hand jigsaw assembly."""

    MIN_SEL = 140

    def __init__(self, camera_index: int = 0) -> None:
        root = Path(__file__).resolve().parent
        model = root / "models" / "hand_landmarker.task"
        self.tracker = HandTracker(model, max_hands=2)
        self.pointer = DualPointerEngine()
        self.camera_index = camera_index
        self.show_help = True
        self._fps = 0.0
        self._frames = 0
        self._fps_t = time.perf_counter()

        self.mode = Mode.SELECT
        self.grid = 3
        self._corner_a: Optional[tuple[int, int]] = None
        self._corner_b: Optional[tuple[int, int]] = None
        self._sel_locked = False
        self._framing = False
        self._frozen: Optional[np.ndarray] = None
        self.puzzle: Optional[JigsawPuzzle] = None
        self._snap_flash = 0

    def run(self) -> int:
        cap = self._open_camera()
        if cap is None:
            print("ERROR: Could not open webcam.")
            return 1

        # Prefer HD when the device supports it
        for size in ((1920, 1080), (1280, 720), (960, 540)):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            ok, probe = cap.read()
            if ok and probe is not None and probe.shape[1] >= size[0] * 0.8:
                break
        cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        win = "HandTrack Studio — Dual-Hand Jigsaw"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        ok, first = cap.read()
        if ok and first is not None:
            cv2.resizeWindow(win, first.shape[1], first.shape[0])
        else:
            cv2.resizeWindow(win, 1280, 720)

        print("HandTrack Studio — both hands pinch to frame · SPACE to craft puzzle · Q quit")

        try:
            while True:
                ok, live = cap.read()
                if not ok:
                    break
                live = cv2.flip(live, 1)
                h, w = live.shape[:2]

                hands = self.tracker.process(live, mirrored=True)
                dual = self.pointer.update(hands)

                if self.mode == Mode.SELECT:
                    frame = self._update_select(live, dual, w, h)
                else:
                    frame = self._update_play(live, dual, w, h)

                if self._snap_flash > 0:
                    flash = frame.copy()
                    cv2.rectangle(flash, (0, 0), (w, h), (180, 255, 200), -1)
                    frame = cv2.addWeighted(flash, 0.12, frame, 0.88, 0)
                    self._snap_flash -= 1

                self._tick_fps()
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    break
        finally:
            self.tracker.close()
            cap.release()
            cv2.destroyAllWindows()
        return 0

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(self.camera_index, backend)
            if cap.isOpened():
                return cap
            cap.release()
        cap = cv2.VideoCapture(self.camera_index)
        return cap if cap.isOpened() else None

    # ---- SELECT: two-hand pinch framing -----------------------------------

    def _update_select(self, live: np.ndarray, dual: DualPointerState, w: int, h: int) -> np.ndarray:
        # Live view while framing; frozen after lock
        if self._sel_locked and self._frozen is not None:
            frame = self._frozen.copy()
        else:
            frame = live.copy()

        # Dual-pinch = stretch selection between both index tips
        if dual.left and dual.right and not self._sel_locked:
            if dual.both_pinching:
                if dual.both_rising or not self._framing:
                    self._framing = True
                ax = int(dual.left.x * (w - 1))
                ay = int(dual.left.y * (h - 1))
                bx = int(dual.right.x * (w - 1))
                by = int(dual.right.y * (h - 1))
                self._corner_a = (ax, ay)
                self._corner_b = (bx, by)
                # Keep a rolling freeze candidate so lock captures sharp frame
                self._frozen = live.copy()
            elif dual.both_falling and self._framing:
                self._framing = False
                if self._selection_valid():
                    self._sel_locked = True
                    if self._frozen is None:
                        self._frozen = live.copy()
                else:
                    self._clear_selection()

        # Draw selection marquee
        if self._corner_a and self._corner_b:
            if self._sel_locked:
                label = "Locked — press SPACE to create puzzle"
            elif self._framing:
                label = "Framing — keep both hands pinched"
            else:
                label = "Release to lock · SPACE to create"
            frame = draw_dual_selection(
                frame,
                self._corner_a[0], self._corner_a[1],
                self._corner_b[0], self._corner_b[1],
                locked=self._sel_locked,
                active=self._framing,
                grid=self.grid,
                label=label,
            )

        # Always show both hand skeletons + cursors
        hand_results = [p.hand for p in dual.hands]
        frame = draw_hands(frame, hand_results, light=False)
        draw_hand_cursors(frame, dual.hands)

        # Presence chips
        draw_status_chip(frame, f"Hands {dual.count}/2", w - 150, 86)
        if dual.both_pinching:
            draw_status_chip(frame, "DUAL PINCH", w - 150, 120, color=(120, 210, 255))

        msg = self._select_message(dual)
        frame = draw_hud(
            frame,
            title="STUDIO  ·  Select",
            message=msg,
            fps=self._fps,
            extra=f"{self.grid}×{self.grid}",
        )
        if self.show_help:
            frame = draw_help(frame, [
                "Both hands pinch  =  frame the area",
                "Stretch corners with L + R index tips",
                "Release both      =  lock selection",
                "SPACE             =  create jigsaw",
                "3 / 4 / 5  grid · C clear · H help · Q quit",
            ])
        return frame

    def _select_message(self, dual: DualPointerState) -> str:
        if self._sel_locked:
            return "Selection locked — SPACE crafts your jigsaw"
        if dual.count < 2:
            return "Show both hands — pinch together to frame a region"
        if dual.both_pinching:
            return "Stretch the frame between both pinched hands…"
        if self._corner_a and self._corner_b and self._selection_valid():
            return "Good frame — dual-pinch again to adjust, or SPACE to create"
        return "Pinch with BOTH hands to select opposite corners"

    def _selection_valid(self) -> bool:
        if not self._corner_a or not self._corner_b:
            return False
        x0, x1 = sorted((self._corner_a[0], self._corner_b[0]))
        y0, y1 = sorted((self._corner_a[1], self._corner_b[1]))
        return (x1 - x0) >= self.MIN_SEL and (y1 - y0) >= self.MIN_SEL

    def _clear_selection(self) -> None:
        self._corner_a = None
        self._corner_b = None
        self._sel_locked = False
        self._framing = False
        self._frozen = None

    def _start_puzzle(self) -> None:
        if not self._selection_valid():
            return
        src = self._frozen
        if src is None:
            return
        x0, x1 = sorted((self._corner_a[0], self._corner_b[0]))  # type: ignore
        y0, y1 = sorted((self._corner_a[1], self._corner_b[1]))  # type: ignore
        # Clamp to image
        y0, y1 = max(0, y0), min(src.shape[0], y1)
        x0, x1 = max(0, x0), min(src.shape[1], x1)
        crop = src[y0:y1, x0:x1].copy()
        if crop.size == 0:
            return

        fh, fw = src.shape[:2]
        # Large premium board, keep crop aspect
        max_w, max_h = int(fw * 0.72), int(fh * 0.72)
        aspect = crop.shape[1] / max(1, crop.shape[0])
        board_w = max_w
        board_h = int(board_w / aspect)
        if board_h > max_h:
            board_h = max_h
            board_w = int(board_h * aspect)
        board_w = max(self.grid * 48, board_w)
        board_h = max(self.grid * 48, board_h)
        board_x = (fw - board_w) // 2
        board_y = (fh - board_h) // 2 + 16

        self.puzzle = JigsawPuzzle.from_image(
            crop, self.grid, self.grid, board_x, board_y, board_w, board_h,
        )
        self.mode = Mode.PLAY

    # ---- PLAY: either / both hands move pieces ----------------------------

    def _update_play(self, live: np.ndarray, dual: DualPointerState, w: int, h: int) -> np.ndarray:
        assert self.puzzle is not None
        frame = live.copy()
        veil = np.full_like(frame, (14, 16, 20))
        frame = cv2.addWeighted(veil, 0.50, frame, 0.50, 0)

        if self.mode == Mode.PLAY:
            for hp in dual.hands:
                key = hp.handedness
                px, py = hp.x * (w - 1), hp.y * (h - 1)
                if hp.pinch_rising:
                    self.puzzle.pick(key, px, py)
                if hp.pinching:
                    self.puzzle.drag(key, px, py, w, h)
                    if self.puzzle.near_slot(key):
                        draw_status_chip(frame, "SNAP ZONE", int(px) + 20, int(py) - 30,
                                         color=(110, 210, 150))
                if hp.pinch_falling:
                    if self.puzzle.drop(key):
                        self._snap_flash = 6
                    if self.puzzle.completed:
                        self.mode = Mode.WIN

        frame = self.puzzle.draw(frame, show_guides=True)
        frame = draw_hands(frame, [p.hand for p in dual.hands], light=True)
        draw_hand_cursors(frame, dual.hands)

        progress = self.puzzle.placed_count / max(1, self.puzzle.total)
        if self.mode == Mode.WIN:
            frame = draw_win(frame)
            msg = "Masterfully assembled"
        else:
            msg = "Pinch pieces with either hand · both hands can hold at once"
        frame = draw_hud(
            frame,
            title="STUDIO  ·  Assemble",
            message=msg,
            fps=self._fps,
            extra=f"{self.puzzle.placed_count}/{self.puzzle.total}",
            progress=progress,
        )
        if self.show_help and self.mode == Mode.PLAY:
            frame = draw_help(frame, [
                "Pinch a tile with either hand to lift it",
                "Both hands can move two pieces at once",
                "Release in the glow zone to snap",
                "R reshuffle · N new capture · Q quit",
            ])
        return frame

    # ---- keys -------------------------------------------------------------

    def _handle_key(self, key: int) -> bool:
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in (ord("h"), ord("H")):
            self.show_help = not self.show_help
        if key in (ord("3"), ord("4"), ord("5")) and self.mode == Mode.SELECT:
            self.grid = int(chr(key))
        if key in (ord("c"), ord("C")) and self.mode == Mode.SELECT:
            self._clear_selection()
        if key in (ord(" "), 13) and self.mode == Mode.SELECT:
            if self._selection_valid():
                self._sel_locked = True
                self._start_puzzle()
        if key in (ord("r"), ord("R")) and self.puzzle is not None:
            self.puzzle.shuffle()
            self.mode = Mode.PLAY
        if key in (ord("n"), ord("N")):
            self.puzzle = None
            self._clear_selection()
            self.mode = Mode.SELECT
        return True

    def _tick_fps(self) -> None:
        self._frames += 1
        now = time.perf_counter()
        if now - self._fps_t >= 1.0:
            self._fps = self._frames / (now - self._fps_t)
            self._frames = 0
            self._fps_t = now
