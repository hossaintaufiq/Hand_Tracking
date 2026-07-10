"""MediaPipe Hand Landmarker wrapper (Tasks API)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class HandResult:
    """One detected hand."""

    landmarks: np.ndarray  # (21, 3) normalized x,y,z
    handedness: str        # "Left" | "Right"
    score: float


class HandTracker:
    """Tracks hands from BGR frames using MediaPipe HandLandmarker."""

    def __init__(self, model_path: Path, max_hands: int = 2) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not model_path.is_file():
            raise FileNotFoundError(f"Missing hand model: {model_path}")

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray) -> list[HandResult]:
        from mediapipe import Image as MpImage, ImageFormat
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        hands: list[HandResult] = []
        if not result.hand_landmarks:
            return hands

        for i, lms in enumerate(result.hand_landmarks):
            pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            handedness = "Right"
            score = 0.0
            if result.handedness and i < len(result.handedness):
                cat = result.handedness[i][0]
                handedness = cat.category_name
                score = float(cat.score)
            hands.append(HandResult(landmarks=pts, handedness=handedness, score=score))
        return hands

    def close(self) -> None:
        self._landmarker.close()
