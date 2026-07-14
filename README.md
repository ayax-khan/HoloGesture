# HoloGesture

**Vision-Based Hand Gesture Control for Real-Time 3D Object Manipulation**

HoloGesture is a desktop application that enables touchless interaction with virtual 3D objects through natural hand gestures captured via a standard webcam. It combines computer vision, machine learning, and GPU-accelerated rendering to deliver a real-time holographic manipulation experience.

The system uses MediaPipe HandLandmarker for 21-point hand tracking, OpenCV for camera acquisition, and OpenGL for 3D rendering with custom holographic shaders. The PyQt5-based interface presents a futuristic heads-up display with real-time feedback on gestures, tracking status, and performance metrics.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8?logo=opencv)]()
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-0097A7?logo=mediapipe)]()

---

## Table of Contents

- [Features](#features)
- [Gesture Controls](#gesture-controls)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [State Machine](#state-machine)
- [Testing](#testing)
- [Requirements](#requirements)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features

- **Hand Tracking** — Real-time 21-point hand landmark detection using MediaPipe HandLandmarker with configurable confidence thresholds and automatic tracking recovery
- **Gesture Recognition** — Seven distinct static gestures recognized through geometric analysis of landmark positions, angles, and distances
- **3D Object Manipulation** — Continuous rotation, scaling, and translation of virtual objects mapped to hand movements with configurable sensitivity
- **Multiple Object Types** — Solid Cube, Solid Sphere, Wireframe Cube, and Holographic Sphere, each rendered with specialized shaders
- **Holographic Rendering** — Custom GLSL shaders featuring Phong lighting, Fresnel glow, scan-line effects, and real-time pulsing animations
- **Gesture History Buffer** — Temporal majority voting over the last five frames to filter spurious detections and improve recognition stability
- **State Machine** — Formal application lifecycle with defined states, transitions, entry/exit handlers, and error recovery paths
- **Settings Management** — Configurable camera selection, smoothing factor, gesture sensitivity, and debug visualization options
- **Error Handling** — Graceful degradation on camera disconnect, performance degradation warnings, and global exception handling with logging
- **Orbit Camera** — Three-dimensional camera system supporting orbital rotation and zoom around the target object
- **Noise Reduction** — Low-pass filtering applied to landmark positions and exponential smoothing on transformation parameters

---

## Gesture Controls

| Gesture | Action | Description |
|---------|--------|-------------|
| Open Palm | Rotation | Hand movement along X/Y axes maps to object rotation about Y/X axes respectively |
| Pinch | Scaling | Distance between thumb and index tip modulates object scale; forward/backward motion controls zoom direction |
| Pointing | Translation | Index finger extended; hand position delta maps to object position offset |
| Swipe Left/Right | Translation | Rapid lateral wrist motion triggers discrete translation steps along X axis |
| Thumbs Up | Reset | Returns object to default position, scale, and orientation |
| Victory | (Status) | Recognized; displayed in HUD with confidence metric |
| Fist | (Status) | Recognized; displayed in HUD with confidence metric |

Each gesture requires confidence above a configurable threshold (`GestureConfig.confidence_threshold`, default 0.6) and is subject to a cooldown period to prevent rapid toggling.

---

## Architecture

The system follows a layered, modular architecture with well-defined interfaces between components:

```
┌──────────────────────────────────────────────────────────┐
│                    Application Layer                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │
│  │ Pipeline  │  │ State    │  │ Config   │  │ Errors  │  │
│  │           │  │ Machine  │  │ Manager  │  │         │  │
│  └─────┬─────┘  └──────────┘  └──────────┘  └─────────┘  │
├────────┼──────────────────────────────────────────────────┤
│        │              Processing Pipeline                  │
│  ┌─────▼─────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │
│  │  Camera    │─▶│  Hand    │─▶│ Gesture  │─▶│ Object  │  │
│  │  Capture   │  │ Tracker  │  │ Engine   │  │ Model   │  │
│  └───────────┘  └──────────┘  └──────────┘  └────┬────┘  │
│                                                   │       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐        │       │
│  │   HUD     │  │ Renderer │◀─│  Scene   │◀───────┘       │
│  │  Overlay  │  │ (OpenGL) │  │  Graph   │                │
│  └──────────┘  └──────────┘  └──────────┘                 │
└──────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. Camera thread captures BGR frames via OpenCV
2. Frames are converted to RGB and passed to MediaPipe HandLandmarker
3. Extracted 21×3 landmark coordinates are processed by the Gesture Engine
4. Classified gestures produce transformation deltas applied to the Object Model
5. The Object Model's 4×4 model matrix is consumed by the OpenGL Renderer each frame
6. The HUD overlay is composited on top of the rendered 3D scene

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python 3.10+ | Rapid prototyping, extensive library ecosystem |
| Hand Tracking | MediaPipe HandLandmarker | State-of-the-art on-device 21-point landmark model with optimized inference |
| Camera | OpenCV 5.0 | Cross-platform video capture, efficient frame preprocessing |
| 3D Rendering | PyOpenGL (OpenGL 3.3 Core) | Low-level GPU access with custom programmable shader pipeline |
| GUI | PyQt5 5.15 | Cross-platform widget toolkit with OpenGL canvas integration |
| Mathematics | NumPy | Vectorized array operations for landmark processing and matrix transformations |
| Testing | Pytest | Parametrized test cases for gesture classification and transform logic |

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- A webcam (640×480 minimum; 1280×720 recommended)
- A GPU supporting OpenGL 3.3 Core Profile

### Installation

```bash
git clone https://github.com/yourusername/HoloGesture.git
cd HoloGesture
pip install -r requirements.txt
```

The MediaPipe HandLandmarker model (`hand_landmarker.task`, ~14 MB) is downloaded automatically on first execution. Alternatively, you may place it manually in the project root directory.

### Running

```bash
python main.py
```

The application window will open and begin initializing the camera and rendering context. Once a hand is detected in the camera frame, gesture recognition activates automatically.

---

## Project Structure

```
HoloGesture/
├── main.py                         # Application entry point
├── config.py                       # Centralized configuration dataclasses
├── requirements.txt                # Python package dependencies
├── hand_landmarker.task            # MediaPipe inference model
│
├── app/
│   ├── camera.py                   # OpenCV video capture wrapper with reconnection
│   ├── hand_tracker.py             # MediaPipe HandLandmarker interface
│   ├── gesture_engine.py           # Geometric gesture classification engine
│   ├── object_model.py             # 3D object transform state with limits
│   ├── renderer.py                 # OpenGL rendering pipeline and shader management
│   ├── shaders.py                  # GLSL vertex and fragment shader source code
│   ├── mesh.py                     # Procedural mesh generators (cube, sphere, wireframe)
│   ├── hud.py                      # HUD data model and state management
│   ├── state_machine.py            # Finite state machine for application lifecycle
│   ├── pipeline.py                 # Processing pipeline orchestrator
│   ├── main_window.py              # PyQt5 main window and UI components
│   ├── math_utils.py               # Vector, matrix, and interpolation utilities
│   └── errors.py                   # Exception hierarchy and logging
│
├── assets/
│   └── shaders/                    # External GLSL shader files (reference copies)
│
└── tests/
    ├── test_gesture_engine.py      # Gesture classification unit tests
    ├── test_math_utils.py          # Mathematical utility unit tests
    └── test_object_model.py        # Object model transform unit tests
```

---

## State Machine

The application operates under a formal finite state machine governing initialization, operation, error recovery, and shutdown:

```
                        ┌──────────────┐
                        │    START      │
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                  ┌─────│ INITIALIZING  │─────┐
                  │     └──────┬───────┘     │
                  │            │              │
                  │     ┌──────▼───────┐     │
                  │     │  SEARCHING   │     │
                  │     │  FOR HAND    │     │
                  │     └──────┬───────┘     │
                  │            │              │
                  │     ┌──────▼───────┐     │
                  │     │    READY      │     │
                  │     └──────┬───────┘     │
                  │            │              │
                  │     ┌──────▼───────┐     │
                  │     │ INTERACTING  │     │
                  │     └──────┬───────┘     │
                  │            │              │
                  │     ┌──────▼───────┐     │
                  └────▶│    ERROR      │◀────┘
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │  SHUTTING    │
                        │   DOWN       │
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │   SHUTDOWN    │
                        └──────────────┘
```

| State | Description |
|-------|-------------|
| START | Initial state before any initialization |
| INITIALIZING | Loading models, initializing camera and OpenGL context |
| SEARCHING_FOR_HAND | Camera active, awaiting hand detection |
| READY | Hand detected, gesture recognition active |
| INTERACTING | Gesture classified, object transformation in progress |
| ERROR | Camera failure or unrecoverable error; recovery attempted |
| SHUTTING_DOWN | Resource cleanup in progress |
| SHUTDOWN | Application terminated |

---

## Testing

The project includes 29 unit tests across three test modules:

```bash
pytest tests/ -v
```

| Module | Tests | Scope |
|--------|-------|-------|
| `test_gesture_engine.py` | 7 | Gesture classification correctness, cooldown logic, history buffer, reset behavior |
| `test_math_utils.py` | 11 | Euclidean distance, angle calculation, matrix construction, interpolation, filtering |
| `test_object_model.py` | 11 | State initialization, rotation/translation/scale operations, clamping, model matrix properties |

---

## Requirements

### Hardware

- x86-64 processor with at least 4 logical cores
- 4 GB RAM (8 GB recommended)
- Webcam supporting 640×480 @ 30 FPS
- GPU with OpenGL 3.3 support (integrated graphics sufficient)

### Software

- Windows 10+, macOS 11+, or Linux (X11/Wayland)
- Python 3.10–3.13
- OpenGL 3.3 compatible graphics driver

### Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ≥5.0 | Camera capture and image processing |
| `mediapipe` | ≥0.10 | Hand landmark detection model |
| `PyOpenGL` | ≥3.1 | OpenGL bindings for 3D rendering |
| `PyQt5` | ≥5.15 | GUI framework and HUD rendering |
| `numpy` | ≥1.24 | Array operations and matrix math |
| `Pillow` | ≥9.0 | Image format support |

---

## Roadmap

### Phase 2 (Planned)

- Dual-hand tracking and interaction
- Custom gesture training via ML classifier
- Speech command integration
- Augmented reality mode (webcam background compositing)

### Phase 3 (Future)

- Physics simulation (inertia, collisions)
- Multi-object scene graph with selection
- External 3D model import (.glb, .obj, .fbx)
- Cloud configuration sync

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">Powered By <strong>Ayax-Khan</strong> Co-Founder Of <a href="https://devssdo.com">devssdo.com</a></p>
