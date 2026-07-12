import numpy as np

from HandTrack.gestures import GestureEngine


def make_thumbs_up_landmarks() -> np.ndarray:
    landmarks = np.array([[0.5, 0.5]] * 21, dtype=float)

    # Thumb is raised; all fingers are relaxed and pointing down.
    landmarks[4] = [0.30, 0.12]
    landmarks[8] = [0.50, 0.58]
    landmarks[12] = [0.50, 0.64]
    landmarks[16] = [0.50, 0.70]
    landmarks[20] = [0.50, 0.76]

    landmarks[6] = [0.50, 0.44]
    landmarks[10] = [0.50, 0.50]
    landmarks[14] = [0.50, 0.56]
    landmarks[18] = [0.50, 0.62]

    return landmarks


def test_save_gesture_requires_a_longer_hold_before_triggering() -> None:
    engine = GestureEngine()
    landmarks = make_thumbs_up_landmarks()

    for index in range(5):
        engine.update(landmarks, now=index * 0.01)
        assert engine.consume_save() is False


def test_save_gesture_triggers_once_after_a_stable_hold() -> None:
    engine = GestureEngine()
    landmarks = make_thumbs_up_landmarks()

    for index in range(8):
        engine.update(landmarks, now=index * 0.01)

    assert engine.consume_save() is True
    assert engine.consume_save() is False
