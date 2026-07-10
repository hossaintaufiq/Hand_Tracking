# HandTrack — webcam hand tracking with MediaPipe landmarks and a gesture pen.

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Features

- Live camera view with colored hand skeleton (like MediaPipe reference)
- **Double-pinch** to START the pen; **double-pinch again** to STOP
- Write on the display with your **index fingertip**
- Gestures: clear (open palm), color change (victory), pause (fist), save (thumbs up)

## Controls

| Gesture / Key | Action |
|---------------|--------|
| Double-pinch | Start pen / stop pen |
| Index point (pen on) | Draw |
| Pinch hold | Lift stroke (pen stays on) |
| Open palm | Clear canvas |
| Victory | Next color |
| Fist | Pause |
| Thumbs up / S | Save PNG |
| H | Help overlay |
| Q | Quit |
