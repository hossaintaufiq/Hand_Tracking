"""Simple hand gesture helpers for the jigsaw game."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from HandTrack.tracker import HandResult


@dataclass
class PointerState:
    """Index-fingertip cursor + pinch grab state."""

    x: float  # normalized 0..1
    y: float
    pinching: bool
    pinch_rising: bool
    pinch_falling: bool
    hand: Optional[HandResult] = None


class PointerEngine:
    """Tracks index tip and pinch (thumb–index) with hysteresis."""

    PINCH_ON = 0.048
    PINCH_OFF = 0.078

    def __init__(self) -> None:
        self._pinching = False
        self._was_pinching = False

    def update(self, hands: list[HandResult]) -> Optional[PointerState]:
        if not hands:
            self._was_pinching = self._pinching
            self._pinching = False
            return None

        hand = hands[0]
        tips = hand.landmarks
        index = tips[8]
        thumb = tips[4]
        dist = float(np.linalg.norm(index[:2] - thumb[:2]))

        if self._pinching:
            pinching = dist < self.PINCH_OFF
        else:
            pinching = dist < self.PINCH_ON

        rising = pinching and not self._pinching
        falling = (not pinching) and self._pinching
        self._was_pinching = self._pinching
        self._pinching = pinching

        return PointerState(
            x=float(np.clip(index[0], 0.0, 1.0)),
            y=float(np.clip(index[1], 0.0, 1.0)),
            pinching=pinching,
            pinch_rising=rising,
            pinch_falling=falling,
            hand=hand,
        )
