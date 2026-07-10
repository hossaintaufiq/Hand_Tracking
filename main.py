#!/usr/bin/env python3
"""HandTrack — webcam hand landmarks + gesture pen.

Run
---
    pip install -r requirements.txt
    python main.py

Controls
--------
Double-pinch (pinch twice quickly)  START pen / STOP pen
Point with index (pen on)           Write on screen
Pinch while pen on                  Lift stroke (pen stays on)
Open palm                           Clear drawing
Victory (V)                         Next pen color
Fist                                Pause / resume
Thumbs up                           Save snapshot
H                                   Toggle help
C                                   Clear
S                                   Save
Q / Esc                             Quit
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HandTrack.app import HandTrackApp


def main() -> int:
    return HandTrackApp(camera_index=0).run()


if __name__ == "__main__":
    raise SystemExit(main())
