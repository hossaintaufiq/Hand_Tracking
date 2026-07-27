# HandTrack — accurate webcam hand landmarks

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Features

- Live camera hand tracking with MediaPipe Hand Landmarker
- Colored joint dots + dotted finger/palm bones (reference style)
- Temporal smoothing for stable landmarks
- Correct Left/Right labels on mirrored camera view

## Controls

| Key | Action |
|-----|--------|
| S | Save PNG snapshot |
| H | Toggle help |
| Q / Esc | Quit |
