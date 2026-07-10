"""HandTrack application — camera, landmarks, pen drawing, gestures."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from HandTrack.canvas import PenCanvas
from HandTrack.gestures import GestureEngine
from HandTrack.overlay import draw_hands, draw_help, draw_hud
from HandTrack.tracker import HandTracker


class HandTrackApp:
    """Full camera hand tracking + gesture pen."""

    def __init__(self, camera_index: int = 0) -> None:
        root = Path(__file__).resolve().parent
        model = root / "models" / "hand_landmarker.task"
        self.tracker = HandTracker(model, max_hands=2)
        self.gestures = GestureEngine()
        self.canvas: Optional[PenCanvas] = None
        self.camera_index = camera_index
        self.show_help = True
        self._fps = 0.0
        self._frames = 0
        self._fps_t = time.perf_counter()
        self.snapshots_dir = root / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)

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

        win = "HandTrack — Double-pinch START / STOP pen"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

        print("HandTrack running. Double-pinch to toggle pen. Q to quit.")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                if self.canvas is None:
                    self.canvas = PenCanvas(w, h)
                else:
                    self.canvas.resize(w, h)

                hands = self.tracker.process(frame)
                primary = hands[0] if hands else None
                lms = primary.landmarks if primary else None
                state = self.gestures.update(lms, time.perf_counter())

                if self.gestures.consume_clear():
                    self.canvas.clear()

                if state.drawing and state.index_tip is not None:
                    self.canvas.draw_to(state.index_tip, self.gestures.pen_color_index)
                elif state.pen_active:
                    self.canvas.lift()

                frame = self.canvas.composite(frame)
                frame = draw_hands(
                    frame,
                    hands,
                    pen_active=state.pen_active,
                    drawing=state.drawing,
                    index_tip=state.index_tip,
                )

                self._tick_fps()
                msg = state.message
                if self.gestures.consume_save():
                    path = self._save(frame)
                    msg = f"Saved {path.name}"

                frame = draw_hud(
                    frame,
                    message=msg,
                    pen_active=state.pen_active,
                    color_index=self.gestures.pen_color_index,
                    fps=self._fps,
                )
                if self.show_help:
                    frame = draw_help(frame)

                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("h"), ord("H")):
                    self.show_help = not self.show_help
                if key in (ord("c"), ord("C")):
                    self.canvas.clear()
                if key in (ord("s"), ord("S")):
                    self._save(frame)
        finally:
            self.tracker.close()
            cap.release()
            cv2.destroyAllWindows()
        return 0

    def _tick_fps(self) -> None:
        self._frames += 1
        now = time.perf_counter()
        if now - self._fps_t >= 1.0:
            self._fps = self._frames / (now - self._fps_t)
            self._frames = 0
            self._fps_t = now

    def _save(self, frame: np.ndarray) -> Path:
        name = datetime.now().strftime("handtrack_%Y%m%d_%H%M%S.png")
        path = self.snapshots_dir / name
        cv2.imwrite(str(path), frame)
        print(f"Saved {path}")
        return path
