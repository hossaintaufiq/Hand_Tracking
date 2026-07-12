"""Gesture recognition: double-pinch start/stop pen, draw, clear, colors."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, Optional

import numpy as np


class Gesture(Enum):
    NONE = auto()
    DOUBLE_PINCH = auto()    # start or stop pen
    PINCH = auto()           # lift stroke (pen stays on)
    DRAW = auto()            # index writing
    OPEN_PALM = auto()       # clear canvas
    FIST = auto()            # pause drawing
    VICTORY = auto()         # next pen color
    THUMBS_UP = auto()       # save snapshot flag
    POINT = auto()


@dataclass
class GestureState:
    gesture: Gesture = Gesture.NONE
    pen_active: bool = False
    drawing: bool = False
    index_tip: Optional[tuple[float, float]] = None  # normalized
    pinch_dist: float = 1.0
    message: str = ""


class GestureEngine:
    """Double-pinch starts the pen; double-pinch again stops it."""

    # Two quick pinches with a release between them
    DOUBLE_PINCH_MAX_GAP = 0.85   # seconds between 1st and 2nd pinch
    DOUBLE_PINCH_MIN_GAP = 0.12   # ignore bounce / same pinch
    PINCH_ON = 0.05
    PINCH_OFF = 0.08              # hysteresis so release is clear
    THUMBS_UP_MIN_DISTANCE = 0.22
    THUMBS_UP_MIN_STABLE_FRAMES = 8

    def __init__(self) -> None:
        self.pen_active: bool = False
        self._pinch_times: Deque[float] = deque(maxlen=4)
        self._pinching = False
        self._released_since_pinch = True
        self._stable: dict[str, int] = {}
        self._cooldown = 0
        self.pen_color_index: int = 0
        self.want_clear: bool = False
        self.want_save: bool = False
        self.paused: bool = False

    def update(self, landmarks: Optional[np.ndarray], now: float) -> GestureState:
        if self._cooldown > 0:
            self._cooldown -= 1

        state = GestureState(pen_active=self.pen_active)
        if landmarks is None:
            state.message = "No hand"
            return state

        tips = {
            "thumb": landmarks[4],
            "index": landmarks[8],
            "middle": landmarks[12],
            "ring": landmarks[16],
            "pinky": landmarks[20],
        }
        pips = {
            "index": landmarks[6],
            "middle": landmarks[10],
            "ring": landmarks[14],
            "pinky": landmarks[18],
        }
        wrist = landmarks[0]

        def up(tip: np.ndarray, pip: np.ndarray) -> bool:
            return float(tip[1]) < float(pip[1]) - 0.02

        fingers = {
            "index": up(tips["index"], pips["index"]),
            "middle": up(tips["middle"], pips["middle"]),
            "ring": up(tips["ring"], pips["ring"]),
            "pinky": up(tips["pinky"], pips["pinky"]),
        }
        thumb_up = (
            float(np.linalg.norm(tips["thumb"][:2] - wrist[:2])) > self.THUMBS_UP_MIN_DISTANCE
            and float(tips["thumb"][1]) < float(wrist[1]) - 0.04
        )

        pinch_dist = float(np.linalg.norm(tips["thumb"][:2] - tips["index"][:2]))
        # Hysteresis: must open fingers between pinches for a real double-pinch
        if self._pinching:
            pinching = pinch_dist < self.PINCH_OFF
        else:
            pinching = pinch_dist < self.PINCH_ON

        if not pinching:
            self._released_since_pinch = True

        # Rising edge after a release = one pinch tap
        if pinching and not self._pinching and self._released_since_pinch:
            self._released_since_pinch = False
            self._pinch_times.append(now)
            self._prune_pinches(now)
            if self._is_double_pinch() and self._cooldown == 0:
                self.pen_active = not self.pen_active
                self._pinch_times.clear()
                self._cooldown = 22
                state.gesture = Gesture.DOUBLE_PINCH
                if self.pen_active:
                    state.message = "PEN STARTED — point to write"
                else:
                    state.message = "PEN STOPPED — double-pinch to start"
                    state.drawing = False
                state.pen_active = self.pen_active
                state.index_tip = (float(tips["index"][0]), float(tips["index"][1]))
                state.pinch_dist = pinch_dist
                self._pinching = pinching
                return state

        self._pinching = pinching

        state.pen_active = self.pen_active
        state.index_tip = (float(tips["index"][0]), float(tips["index"][1]))
        state.pinch_dist = pinch_dist

        # Discrete gestures (stable frames) — skip while pinching so taps stay clean
        label = "none"
        if not pinching:
            if all(not v for v in fingers.values()) and not thumb_up:
                label = "fist"
            elif all(fingers.values()) and thumb_up:
                label = "palm"
            elif fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
                label = "victory"
            elif thumb_up and not any(fingers.values()):
                label = "thumbs"
            elif fingers["index"] and not fingers["middle"] and not fingers["ring"]:
                label = "point"

        self._stable[label] = self._stable.get(label, 0) + 1
        for k in list(self._stable.keys()):
            if k != label:
                self._stable[k] = 0

        if self._stable.get(label, 0) >= self.THUMBS_UP_MIN_STABLE_FRAMES and self._cooldown == 0:
            if label == "palm":
                self.want_clear = True
                state.gesture = Gesture.OPEN_PALM
                state.message = "Clear canvas"
                self._cooldown = 25
            elif label == "fist":
                self.paused = not self.paused
                state.gesture = Gesture.FIST
                state.message = "Paused" if self.paused else "Resume"
                self._cooldown = 25
            elif label == "victory":
                self.pen_color_index = (self.pen_color_index + 1) % 5
                state.gesture = Gesture.VICTORY
                state.message = f"Color #{self.pen_color_index + 1}"
                self._cooldown = 25
            elif label == "thumbs":
                self.want_save = True
                state.gesture = Gesture.THUMBS_UP
                state.message = "Snapshot"
                self._cooldown = 25

        # Drawing only while pen is started
        if self.pen_active and not self.paused and fingers["index"] and not pinching:
            state.drawing = True
            state.gesture = Gesture.DRAW
            if not state.message:
                state.message = "Writing…  (double-pinch to stop)"
        elif self.pen_active and pinching:
            state.gesture = Gesture.PINCH
            if not state.message:
                state.message = "Stroke lift — double-pinch again to STOP pen"
        elif self.pen_active and not state.message:
            state.message = "Pen ON — point to write · double-pinch to STOP"
        elif not state.message:
            state.message = "Double-pinch to START pen"

        return state

    def _is_double_pinch(self) -> bool:
        if len(self._pinch_times) < 2:
            return False
        gap = self._pinch_times[-1] - self._pinch_times[-2]
        return self.DOUBLE_PINCH_MIN_GAP <= gap <= self.DOUBLE_PINCH_MAX_GAP

    def _prune_pinches(self, now: float) -> None:
        while self._pinch_times and now - self._pinch_times[0] > self.DOUBLE_PINCH_MAX_GAP:
            self._pinch_times.popleft()

    def consume_clear(self) -> bool:
        flag = self.want_clear
        self.want_clear = False
        return flag

    def consume_save(self) -> bool:
        flag = self.want_save
        self.want_save = False
        return flag
