"""HandTrack Jigsaw — select a camera area and assemble it with your hand."""

from __future__ import annotations

import time
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from HandTrack.jigsaw import JigsawPuzzle
from HandTrack.overlay import (
    draw_cursor,
    draw_hands,
    draw_help,
    draw_hud,
    draw_selection,
    draw_win,
)
from HandTrack.pointer import PointerEngine
from HandTrack.tracker import HandTracker


class Mode(Enum):
    SELECT = auto()
    PLAY = auto()
    WIN = auto()


class HandTrackApp:
    """Camera jigsaw game controlled by hand pinch + index tip."""

    MIN_SEL = 120  # minimum selection side in pixels

    def __init__(self, camera_index: int = 0) -> None:
        root = Path(__file__).resolve().parent
        model = root / "models" / "hand_landmarker.task"
        self.tracker = HandTracker(model, max_hands=1)
        self.pointer = PointerEngine()
        self.camera_index = camera_index
        self.show_help = True
        self._fps = 0.0
        self._frames = 0
        self._fps_t = time.perf_counter()

        self.mode = Mode.SELECT
        self.grid = 3  # 3x3 default; keys 3/4/5 change size
        self._sel_start: Optional[tuple[int, int]] = None
        self._sel_end: Optional[tuple[int, int]] = None
        self._sel_dragging = False
        self._sel_locked = False
        self._frozen: Optional[np.ndarray] = None
        self.puzzle: Optional[JigsawPuzzle] = None

    def run(self) -> int:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("ERROR: Could not open webcam.")
            return 1

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        win = "HandTrack Jigsaw"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        print("HandTrack Jigsaw — pinch-drag to select, SPACE to start. Q to quit.")

        try:
            while True:
                ok, live = cap.read()
                if not ok:
                    break
                live = cv2.flip(live, 1)
                h, w = live.shape[:2]

                # Track on live frame always (better continuity)
                hands = self.tracker.process(live, mirrored=True)
                ptr = self.pointer.update(hands)

                if self.mode == Mode.SELECT:
                    frame = self._update_select(live, ptr, w, h)
                else:
                    frame = self._update_play(live, ptr, w, h)

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

    # ---- SELECT -----------------------------------------------------------

    def _update_select(
        self,
        live: np.ndarray,
        ptr,
        w: int,
        h: int,
    ) -> np.ndarray:
        # Background: frozen crop preview or live camera
        frame = self._frozen.copy() if self._frozen is not None else live.copy()

        if ptr is not None:
            px, py = int(ptr.x * (w - 1)), int(ptr.y * (h - 1))

            if ptr.pinch_rising and not self._sel_locked:
                self._sel_start = (px, py)
                self._sel_end = (px, py)
                self._sel_dragging = True
                self._frozen = live.copy()  # freeze the moment selection starts
                frame = self._frozen.copy()

            if self._sel_dragging and ptr.pinching and self._sel_start is not None:
                self._sel_end = (px, py)

            if ptr.pinch_falling and self._sel_dragging:
                self._sel_dragging = False
                if self._selection_valid():
                    self._sel_locked = True
                else:
                    self._clear_selection()

            frame = draw_hands(frame, [ptr.hand] if ptr.hand else [], light=True)
            draw_cursor(frame, ptr.x, ptr.y, pinching=ptr.pinching)

        if self._sel_start and self._sel_end:
            frame = draw_selection(
                frame,
                self._sel_start[0], self._sel_start[1],
                self._sel_end[0], self._sel_end[1],
                locked=self._sel_locked,
            )

        msg = self._select_message()
        frame = draw_hud(
            frame,
            title="Jigsaw — Select",
            message=msg,
            fps=self._fps,
            extra=f"{self.grid}x{self.grid}",
        )
        if self.show_help:
            frame = draw_help(frame, [
                "Pinch-drag  =  select area",
                "SPACE / open confirm  =  make puzzle",
                "3 / 4 / 5  =  grid size",
                "C  =  clear selection",
                "H  help · Q quit",
            ])
        return frame

    def _select_message(self) -> str:
        if self._sel_locked:
            return "Selection ready — press SPACE to create jigsaw"
        if self._sel_dragging:
            return "Drag to cover the area you want as the puzzle…"
        return "Pinch and drag to select an area from the camera"

    def _selection_valid(self) -> bool:
        if not self._sel_start or not self._sel_end:
            return False
        x0, x1 = sorted((self._sel_start[0], self._sel_end[0]))
        y0, y1 = sorted((self._sel_start[1], self._sel_end[1]))
        return (x1 - x0) >= self.MIN_SEL and (y1 - y0) >= self.MIN_SEL

    def _clear_selection(self) -> None:
        self._sel_start = None
        self._sel_end = None
        self._sel_dragging = False
        self._sel_locked = False
        self._frozen = None

    def _start_puzzle(self) -> None:
        if not self._selection_valid() or self._frozen is None:
            return
        x0, x1 = sorted((self._sel_start[0], self._sel_end[0]))  # type: ignore
        y0, y1 = sorted((self._sel_start[1], self._sel_end[1]))  # type: ignore
        crop = self._frozen[y0:y1, x0:x1].copy()
        if crop.size == 0:
            return

        fh, fw = self._frozen.shape[:2]
        # Center a board sized from the selection (capped)
        board_w = min(fw - 80, max(360, x1 - x0))
        board_h = min(fh - 100, max(280, y1 - y0))
        # Keep aspect of crop
        aspect = crop.shape[1] / max(1, crop.shape[0])
        if board_w / max(1, board_h) > aspect:
            board_w = int(board_h * aspect)
        else:
            board_h = int(board_w / aspect)
        board_w = max(self.grid * 40, board_w)
        board_h = max(self.grid * 40, board_h)
        board_x = (fw - board_w) // 2
        board_y = (fh - board_h) // 2 + 10

        self.puzzle = JigsawPuzzle.from_image(
            crop, self.grid, self.grid, board_x, board_y, board_w, board_h,
        )
        self.mode = Mode.PLAY

    # ---- PLAY -------------------------------------------------------------

    def _update_play(self, live: np.ndarray, ptr, w: int, h: int) -> np.ndarray:
        assert self.puzzle is not None
        # Soft live backdrop under the puzzle
        frame = live.copy()
        dim = frame.copy()
        cv2.rectangle(dim, (0, 0), (w, h), (20, 20, 20), -1)
        frame = cv2.addWeighted(dim, 0.45, frame, 0.55, 0)

        if ptr is not None and self.mode == Mode.PLAY:
            px, py = ptr.x * (w - 1), ptr.y * (h - 1)
            if ptr.pinch_rising:
                self.puzzle.pick(px, py)
            if ptr.pinching:
                self.puzzle.drag(px, py, w, h)
            if ptr.pinch_falling:
                self.puzzle.drop()
                if self.puzzle.completed:
                    self.mode = Mode.WIN

        frame = self.puzzle.draw(frame, show_guides=True)

        if ptr is not None and ptr.hand is not None:
            frame = draw_hands(frame, [ptr.hand], light=True)
            draw_cursor(frame, ptr.x, ptr.y, pinching=ptr.pinching)

        if self.mode == Mode.WIN:
            frame = draw_win(frame)
            msg = "You solved it!"
        else:
            msg = "Pinch a piece to grab · release near its slot to snap"
        frame = draw_hud(
            frame,
            title="Jigsaw — Play",
            message=msg,
            fps=self._fps,
            extra=f"{self.puzzle.placed_count}/{self.puzzle.total}",
        )
        if self.show_help and self.mode == Mode.PLAY:
            frame = draw_help(frame, [
                "Pinch piece  =  grab / move",
                "Release near slot  =  snap",
                "R  =  reshuffle   N  =  new select",
                "H  help · Q quit",
            ])
        return frame

    # ---- keys / utils -----------------------------------------------------

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
            if self._sel_locked or self._selection_valid():
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
