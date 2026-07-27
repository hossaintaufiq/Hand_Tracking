#!/usr/bin/env python3
"""HandTrack — accurate webcam hand landmark tracking.

Run
---
    pip install -r requirements.txt
    python main.py

Controls
--------
S                                   Save snapshot
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
