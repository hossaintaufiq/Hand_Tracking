"""HandTrack Studio — premium two-hand jigsaw from a live camera crop."""

from __future__ import annotations

import time
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from HandTrack import ui
from HandTrack.effects import Effects
from HandTrack.jigsaw import JigsawPuzzle
from HandTrack.overlay import (
    draw_dual_selection,
    draw_framing_link,
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
        self.fx = Effects()
        self.camera_index = camera_index
        self.show_help = True
        self._fps = 0.0
        self._frames = 0
        self._fps_t = time.perf_counter()

        self.mode = Mode.SELECT
        self.grid = 3
        self._corner_a: Optional[tuple[float, float]] = None
        self._corner_b: Optional[tuple[float, float]] = None
        self._smooth_a: Optional[tuple[float, float]] = None
        self._smooth_b: Optional[tuple[float, float]] = None
        self._sel_locked = False
        self._framing = False
        self._frozen: Optional[np.ndarray] = None
        self.puzzle: Optional[JigsawPuzzle] = None
        self._progress_disp: Optional[float] = None
        self._play_started: Optional[float] = None
        self._win_celebrated = False

    def run(self) -> int:
        cap = self._open_camera()
        if cap is None:
            print("ERROR: Could not open webcam.")
            return 1

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

        win = "HandTrack Studio"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        ok, first = cap.read()
        if ok and first is not None:
            cv2.resizeWindow(win, first.shape[1], first.shape[0])
        else:
            cv2.resizeWindow(win, 1280, 720)

        self.fx.flash_fade(0.65)
        print("HandTrack Studio — dual-hand frame · SPACE to craft · Q quit")

        try:
            while True:
                ok, live = cap.read()
                if not ok:
                    break
                live = cv2.flip(live, 1)
                h, w = live.shape[:2]
                now = time.perf_counter()

                hands = self.tracker.process(live, mirrored=True)
                dual = self.pointer.update(hands)

                if self.mode == Mode.SELECT:
                    frame = self._update_select(live, dual, w, h, now)
                else:
                    frame = self._update_play(live, dual, w, h, now)

                frame = ui.vignette(frame, strength=0.26)
                self.fx.update()
                frame = self.fx.draw(frame)

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

    def _timer_str(self) -> str:
        if self._play_started is None:
            return ""
        sec = int(max(0.0, time.perf_counter() - self._play_started))
        return f"{sec // 60:02d}:{sec % 60:02d}"

    # ---- SELECT -----------------------------------------------------------

    def _update_select(
        self, live: np.ndarray, dual: DualPointerState, w: int, h: int, now: float,
    ) -> np.ndarray:
        if self._sel_locked and self._frozen is not None:
            frame = self._frozen.copy()
        else:
            frame = live.copy()

        if dual.left and dual.right and not self._sel_locked:
            if dual.both_pinching:
                self._framing = True
                self._corner_a = (dual.left.x * (w - 1), dual.left.y * (h - 1))
                self._corner_b = (dual.right.x * (w - 1), dual.right.y * (h - 1))
                self._frozen = live.copy()
            elif dual.both_falling and self._framing:
                self._framing = False
                if self._selection_valid():
                    self._sel_locked = True
                    self.fx.burst(
                        (self._corner_a[0] + self._corner_b[0]) * 0.5,
                        (self._corner_a[1] + self._corner_b[1]) * 0.5,
                        color=ui.ACCENT, n=12,
                    )
                    if self._frozen is None:
                        self._frozen = live.copy()
                else:
                    self._clear_selection()

        if self._corner_a and self._corner_b:
            self._smooth_a = ui.lerp_point(self._smooth_a, self._corner_a, 0.42)
            self._smooth_b = ui.lerp_point(self._smooth_b, self._corner_b, 0.42)
            if self._sel_locked:
                label = "Locked — SPACE to create"
            elif self._framing:
                label = "Framing"
            else:
                label = "Ready — SPACE or adjust"
            frame = draw_dual_selection(
                frame,
                self._smooth_a[0], self._smooth_a[1],
                self._smooth_b[0], self._smooth_b[1],
                locked=self._sel_locked,
                active=self._framing,
                grid=self.grid,
                label=label,
            )

        frame = draw_framing_link(frame, dual, w, h)
        frame = draw_hands(frame, [p.hand for p in dual.hands], light=False)
        draw_hand_cursors(frame, dual.hands, t=now)

        draw_status_chip(frame, f"{dual.count}/2 hands", w - 150, 108)
        if dual.both_pinching:
            draw_status_chip(frame, "DUAL PINCH", w - 150, 146, color=ui.ACCENT_HOT)

        frame, _ = draw_hud(
            frame,
            title="STUDIO",
            message=self._select_message(dual),
            fps=self._fps,
            extra=f"{self.grid}×{self.grid}",
            step=1,
        )
        if self.show_help:
            frame = draw_help(frame, [
                "Pinch with both hands to frame corners",
                "Stretch L + R index tips to resize",
                "Release to lock the selection",
                "SPACE creates the jigsaw",
                "3 / 4 / 5 grid · C clear · H · Q",
            ])
        return frame

    def _select_message(self, dual: DualPointerState) -> str:
        if self._sel_locked:
            return "Selection locked — press SPACE to craft your puzzle"
        if dual.count < 2:
            return "Show both hands to begin framing"
        if dual.both_pinching:
            return "Stretch the frame between your pinched hands…"
        if self._corner_a and self._corner_b and self._selection_valid():
            return "Looking good — SPACE to create, or dual-pinch to adjust"
        return "Pinch with both hands — each tip is a corner"

    def _selection_valid(self) -> bool:
        if not self._corner_a or not self._corner_b:
            return False
        x0, x1 = sorted((self._corner_a[0], self._corner_b[0]))
        y0, y1 = sorted((self._corner_a[1], self._corner_b[1]))
        return (x1 - x0) >= self.MIN_SEL and (y1 - y0) >= self.MIN_SEL

    def _clear_selection(self) -> None:
        self._corner_a = None
        self._corner_b = None
        self._smooth_a = None
        self._smooth_b = None
        self._sel_locked = False
        self._framing = False
        self._frozen = None

    def _start_puzzle(self) -> None:
        if not self._selection_valid() or self._frozen is None:
            return
        src = self._frozen
        x0, x1 = sorted((int(self._corner_a[0]), int(self._corner_b[0])))  # type: ignore
        y0, y1 = sorted((int(self._corner_a[1]), int(self._corner_b[1])))  # type: ignore
        y0, y1 = max(0, y0), min(src.shape[0], y1)
        x0, x1 = max(0, x0), min(src.shape[1], x1)
        crop = src[y0:y1, x0:x1].copy()
        if crop.size == 0:
            return

        fh, fw = src.shape[:2]
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
        board_y = (fh - board_h) // 2 + 20

        self.puzzle = JigsawPuzzle.from_image(
            crop, self.grid, self.grid, board_x, board_y, board_w, board_h,
        )
        self._progress_disp = 0.0
        self._play_started = time.perf_counter()
        self._win_celebrated = False
        self.fx.flash_fade(0.5)
        self.mode = Mode.PLAY

    # ---- PLAY -------------------------------------------------------------

    def _update_play(
        self, live: np.ndarray, dual: DualPointerState, w: int, h: int, now: float,
    ) -> np.ndarray:
        assert self.puzzle is not None
        frame = live.copy()
        veil = np.full_like(frame, ui.BG)
        frame = cv2.addWeighted(veil, 0.58, frame, 0.42, 0)

        if self.mode == Mode.PLAY:
            for hp in dual.hands:
                key = hp.handedness
                px, py = hp.x * (w - 1), hp.y * (h - 1)
                if hp.pinch_rising:
                    self.puzzle.pick(key, px, py)
                if hp.pinching:
                    self.puzzle.drag(key, px, py, w, h)
                    if self.puzzle.near_slot(key):
                        draw_status_chip(
                            frame, "SNAP", int(px) + 18, int(py) - 36, color=ui.SUCCESS,
                        )
                if hp.pinch_falling:
                    snapped, center = self.puzzle.drop(key)
                    if snapped and center is not None:
                        self.fx.burst(center[0], center[1], color=ui.SUCCESS, n=16)
                    if self.puzzle.completed:
                        self.mode = Mode.WIN

        frame = self.puzzle.draw(frame, show_guides=True)
        if self.mode == Mode.PLAY:
            frame = self.puzzle.draw_reference(frame)

        frame = draw_hands(frame, [p.hand for p in dual.hands], light=True)
        draw_hand_cursors(frame, dual.hands, t=now)

        progress = self.puzzle.placed_count / max(1, self.puzzle.total)
        step = 3 if self.mode == Mode.WIN else 2
        if self.mode == Mode.WIN:
            if not self._win_celebrated:
                self.fx.confetti(w, h, n=55)
                self.fx.flash_fade(0.4)
                self._win_celebrated = True
            frame = draw_win(frame, t=now)
            msg = "Beautifully assembled"
        else:
            msg = "Magnetic snap assists when you’re close · both hands welcome"

        frame, self._progress_disp = draw_hud(
            frame,
            title="STUDIO",
            message=msg,
            fps=self._fps,
            extra=f"{self.puzzle.placed_count} / {self.puzzle.total}",
            progress=progress,
            progress_display=self._progress_disp,
            step=step,
            timer=self._timer_str(),
        )
        if self.show_help and self.mode == Mode.PLAY:
            frame = draw_help(frame, [
                "Pinch a tile to lift — either hand",
                "Hold two pieces at once if you like",
                "Release near the slot — magnetism helps",
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
            self._progress_disp = 0.0
            self._play_started = time.perf_counter()
            self._win_celebrated = False
            self.fx.flash_fade(0.35)
            self.mode = Mode.PLAY
        if key in (ord("n"), ord("N")):
            self.puzzle = None
            self._clear_selection()
            self._progress_disp = None
            self._play_started = None
            self._win_celebrated = False
            self.fx.flash_fade(0.4)
            self.mode = Mode.SELECT
        return True

    def _tick_fps(self) -> None:
        self._frames += 1
        now = time.perf_counter()
        if now - self._fps_t >= 1.0:
            self._fps = self._frames / (now - self._fps_t)
            self._frames = 0
            self._fps_t = now
