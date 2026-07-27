# HandTrack Studio — Premium Dual-Hand Jigsaw Puzzle

A real-time hand gesture recognition and tracking application that enables users to create and play jigsaw puzzles using dual-hand gestures. This project leverages MediaPipe for hand tracking and OpenCV for computer vision processing.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Selection Mode](#selection-mode)
  - [Play Mode](#play-mode)
  - [Keyboard Controls](#keyboard-controls)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

---

## Overview

HandTrack Studio is an interactive computer vision application that transforms your webcam into a gesture-controlled interface for creating and playing jigsaw puzzles. Using dual-hand tracking, you can:

1. **Frame a region** of your camera view using two-handed pinch gestures
2. **Create a jigsaw puzzle** from that framed area
3. **Solve the puzzle** by manipulating pieces with intuitive hand gestures

The application uses **MediaPipe Hand Landmarker** for accurate hand skeleton detection and **temporal smoothing** to ensure stable, responsive interactions.

---

## Features

✨ **Dual-Hand Tracking**
- Real-time detection and tracking of both hands simultaneously
- Supports up to 2 hands with independent skeleton detection
- Temporal smoothing for smooth, jitter-free movement

🎮 **Gesture Recognition**
- Pinch detection for grabbing puzzle pieces
- Dual-hand framing for region selection
- Smooth interpolation for natural hand movement

🧩 **Interactive Jigsaw Puzzle**
- Configurable grid sizes (3×3, 4×4, 5×5)
- Two-handed piece manipulation
- Automatic snap-to-grid placement
- Reshuffle and reset options

🎨 **Visual Feedback**
- Real-time hand skeleton overlay
- Gesture cursor indicators
- Selection frame preview
- Win state animation
- Adjustable UI with help overlay

---

## Requirements

### System Requirements
- **Python 3.8+** (tested with 3.9+)
- **Webcam** for camera input
- **CPU:** Intel Core i5 or equivalent (GPU optional but recommended)
- **RAM:** 4GB minimum (8GB recommended)

### Dependencies
- `opencv-python>=4.9.0` — Computer vision and image processing
- `mediapipe>=0.10.9` — Hand landmark detection
- `numpy>=1.26.0` — Numerical computing

---

## Installation

### 1. Clone or Download the Project

```bash
cd "d:\Project 2026\3D_Learning_Model"
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import cv2, mediapipe; print('✓ Dependencies installed successfully')"
```

---

## Quick Start

### Run the Application

```bash
python main.py
```

Or directly:

```bash
python -m HandTrack.app
```

The webcam feed will open with hand tracking enabled. You should see:
- Real-time camera video
- Hand skeleton overlays (when hands are visible)
- HUD information in the top-left corner

---

## Usage Guide

### Selection Mode
This is the default mode when you start the application. Use it to frame the region for your puzzle.

#### Steps:

1. **Show Both Hands**
   - Position both hands in front of the camera
   - Verify that both hand skeletons appear on screen

2. **Initiate Selection**
   - With both hands visible, bring your index fingers and thumbs close together
   - **Pinch** simultaneously with both hands
   - Your index finger tips become the **top-left and bottom-right corners** of the selection frame

3. **Adjust Frame**
   - Move your hands to resize and reposition the selection frame
   - The frame preview will update in real-time
   - Minimum frame size: 140×140 pixels

4. **Confirm Selection**
   - **Release** both pinches to lock the selection
   - A green confirmation frame appears

5. **Create Puzzle**
   - Press **SPACE** or **ENTER** to generate the jigsaw puzzle
   - The selected region is captured and divided into grid pieces
   - Automatically transitions to **Play Mode**

### Play Mode
Manipulate puzzle pieces to solve the jigsaw.

#### Controls:

| Gesture | Action |
|---------|--------|
| **Single pinch** (either hand) | Grab a puzzle piece |
| **Move pinched hand** | Drag the piece across the board |
| **Dual pinch** (both hands) | Grab and move two pieces simultaneously |
| **Release near slot** | Piece snaps to closest grid position |
| **Release in open area** | Piece returns to random shuffle position |

#### Tips:
- Pieces automatically snap to grid positions when released within proximity
- Move pieces independently with one hand or coordinate both hands for complex moves
- Watch for visual feedback—pieces highlight when selected

### Keyboard Controls

| Key | Mode | Action |
|-----|------|--------|
| **SPACE** / **ENTER** | Selection | Create jigsaw from selection |
| **3** / **4** / **5** | Play | Set grid size (3×3, 4×4, 5×5) |
| **C** | Selection | Clear/cancel selection |
| **R** | Play | Reshuffle all pieces randomly |
| **N** | Both | Capture new frame (return to selection) |
| **H** | Both | Toggle help overlay |
| **Q** / **ESC** | Both | Quit application |

---

## Project Structure

```
3D_Learning_Model/
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
└── HandTrack/                   # Main package
    ├── __init__.py              # Package initialization
    ├── app.py                   # Main application class (HandTrackApp)
    ├── tracker.py               # Hand tracking and landmark detection
    ├── pointer.py               # Dual-hand gesture recognition
    ├── landmarks.py             # Hand landmark utilities
    ├── jigsaw.py                # Jigsaw puzzle logic
    ├── effects.py               # Visual effects and animations
    ├── overlay.py               # Drawing and rendering functions
    ├── ui.py                    # UI utilities and constants
    │
    ├── models/                  # Pre-trained ML models
    │   ├── hand_landmarker.task # MediaPipe hand detection model
    │   └── gesture_recognizer.task # Gesture recognition model
    │
    └── snapshots/               # Cached puzzle snapshots
```

---

## Architecture

### Application Flow

```
HandTrackApp (Main Loop)
├── HandTracker (MediaPipe Hand Detection)
│   ├── Input: Camera frame (960px wide)
│   ├── Process: Landmark detection + smoothing
│   └── Output: Hand skeletons for 0-2 hands
│
├── DualPointerEngine (Gesture Recognition)
│   ├── Input: Hand landmarks
│   ├── Process: Pinch detection, hand state tracking
│   └── Output: Pointer positions and gestures
│
├── JigsawPuzzle (Game Logic)
│   ├── Input: Selected frame region, grid size
│   ├── Process: Puzzle generation, piece movement
│   └── Output: Piece positions, win state
│
└── Overlay Rendering (UI Display)
    ├── Input: Hands, pointer state, puzzle state
    ├── Process: Draw skeletons, cursors, UI elements
    └── Output: Annotated frame to display
```

### Core Components

#### 1. **HandTracker** (`tracker.py`)
- Wraps MediaPipe Hand Landmarker for real-time detection
- Detects up to 2 hands per frame
- Applies temporal smoothing (EMA) to reduce jitter
- Returns `HandResult` objects with landmarks, handedness, and confidence

#### 2. **DualPointerEngine** (`pointer.py`)
- Processes hand landmarks to detect gestures
- Tracks pinch state for both hands independently
- Maintains `DualPointerState` with cursor positions and grab status
- Handles threshold-based pinch detection

#### 3. **JigsawPuzzle** (`jigsaw.py`)
- Manages puzzle grid and piece positions
- Handles collision detection for snapping pieces
- Implements win condition checking
- Supports grid size changes (3×3, 4×4, 5×5)

#### 4. **Effects** (`effects.py`)
- Provides visual feedback animations
- Manages puzzle state transitions
- Renders winning effects

#### 5. **Overlay** (`overlay.py`)
- Renders hand skeletons, cursors, and UI elements
- Draws selection frame during framing
- Displays HUD and help information
- Annotates puzzle pieces with visual feedback

---

## Technical Details

### Hand Tracking Resolution
- **Inference Width:** 960 pixels (internal processing)
- **Display Resolution:** Native camera resolution
- **Benefit:** Balances accuracy and performance

### Temporal Smoothing
- **Algorithm:** Exponential Moving Average (EMA)
- **Default Alpha:** 0.42
- **Purpose:** Reduces jitter from hand tremor while maintaining responsiveness

### Gesture Detection
- **Pinch Detection:** Distance between thumb tip and index tip < threshold
- **State Machine:** Tracks pinch enter/exit for reliable gesture recognition
- **Dual-Hand Coordination:** Independent hand state with synchronized actions

### Performance Optimization
- Frame skipping for reduced latency
- Cached hand detection results
- Efficient NumPy operations for landmark processing
- GPU acceleration available via MediaPipe/OpenCV

---

## Troubleshooting

### Common Issues

#### **Hands Not Detected**
- Ensure good lighting (natural light preferred)
- Keep hands fully visible in frame
- Remove occlusions (jewelry, sleeves)
- Try adjusting camera angle and distance
- Check camera permissions in OS settings

#### **Jittery Hand Movement**
- Increase smoothing by adjusting `LandmarkSmoother.alpha` in `tracker.py`
- Ensure stable lighting
- Move more slowly and deliberately
- Reduce background clutter

#### **Puzzle Not Creating**
- Verify both hands are visible before pinching
- Ensure selection frame is larger than 140×140 pixels
- Check for "Press SPACE to create" prompt in HUD
- Try pressing SPACE again if selection frame is green

#### **Pieces Not Snapping**
- Release pieces closer to the target grid position
- Ensure puzzle grid is fully visible
- Check that pieces are not overlapping other pieces
- Try reshuffle (R key) to reset piece positions

#### **Camera Not Opening**
- Verify webcam is connected and not in use by another application
- Try specifying a different camera index in `main.py`:
  ```python
  HandTrackApp(camera_index=1).run()  # Try camera 1 instead of 0
  ```
- Check webcam permissions in Windows settings

#### **High CPU/GPU Usage**
- Reduce camera resolution if possible
- Lower inference width in `app.py`
- Close background applications
- Check for multiple HandTrack instances running

#### **MediaPipe Model Not Found**
- Ensure models are present in `HandTrack/models/`
- Verify file names match exactly:
  - `hand_landmarker.task`
  - `gesture_recognizer.task`
- Re-download models if corrupted

### Debug Mode
For troubleshooting, enable debug output:

```python
# In HandTrack/app.py
self.tracker = HandTracker(model, max_hands=2, debug=True)
```

### Performance Profiling
Monitor FPS and hand detection latency in the HUD (top-left corner):
```
FPS: 30.5
Hands: 2
Latency: 12ms
```

---

## Controls Quick Reference

| Category | Command |
|----------|---------|
| **Start** | Show both hands, pinch, stretch |
| **Create Puzzle** | SPACE / ENTER |
| **Adjust Grid** | 3 / 4 / 5 |
| **Reset** | R (reshuffle), N (new capture), C (clear) |
| **Help & Exit** | H (help), Q / ESC (quit) |

---

## Future Enhancements

Potential improvements for future versions:
- 🎨 Custom image uploads for puzzles
- 🎵 Sound effects and background music
- 🏆 Leaderboard and timing system
- 🎯 Difficulty levels with piece rotation
- 🖼️ Multiple puzzle themes and styles
- 💾 Save/load game progress
- 🌐 Multiplayer hand tracking
- 📊 Performance analytics and statistics

---

## License

This project is part of the 3D Learning Model initiative.

---

## Support

For issues, questions, or feature requests:
1. Check the **Troubleshooting** section above
2. Review the code comments in relevant modules
3. Ensure all dependencies are correctly installed
4. Test with the example gestures in the **Usage Guide**

---

**Happy puzzling! 🧩**
