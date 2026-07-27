#!/usr/bin/env python3
"""HandTrack Jigsaw — select a camera area and solve it by hand.

Run
---
    pip install -r requirements.txt
    python main.py

Select mode
-----------
Pinch-drag                          Select area
SPACE / Enter                       Create jigsaw from selection
3 / 4 / 5                           Grid size (3x3, 4x4, 5x5)
C                                   Clear selection

Play mode
---------
Pinch on a piece                    Grab / move
Release near correct slot           Snap in place
R                                   Reshuffle
N                                   New selection

H                                   Toggle help
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
