# Mood Mirror

Mood Mirror is a Python desktop application that uses a live webcam feed to estimate facial expression signals and transform the entire interface to match the detected mood in real time.

The project is designed as a demo-friendly, visually reactive experience rather than a clinical emotion analysis tool. It combines webcam capture, face detection, optional facial landmarks, lightweight expression heuristics, and a theme-driven PySide6 interface to create a responsive "emotion mirror" for demos, experiments, and creative tooling.

## What It Does

Mood Mirror watches the active webcam stream, finds a face, estimates expression cues such as mouth openness, smile curvature, eyebrow movement, and eye openness, then maps those cues into a mood category.

The current mood drives both the visual theme and the data shown in the side panel.

- `happy`: warm gradients, bright accents, floating light-orb styling
- `sad`: cool palette with blue rain particle effects
- `angry`: darker red pulse-ring visuals and ember effects
- `surprise`: bright scene with confetti-style animation
- `neutral`: calm slate-toned background with subtle drift

## App Modes

### Mirror Mode

Mirror Mode is the default reactive experience.

- Shows the live webcam feed
- Draws a face box and landmarks when available
- Tracks recent mood history
- Updates the scene theme continuously based on the top detected emotion

### Interview Mode

Interview Mode focuses on session-style behavioral signals rather than pure theme changes.

- `Calmness`: derived from the ongoing emotion mix and confidence pattern
- `Smile Rate`: estimated from facial smile signals over time
- `Surprise Moments`: counted from surprise-related face signals with hysteresis to avoid duplicate counts every frame
- Session timer and rolling mood timeline for quick review

This mode is meant for mock interviews, presentation practice, and self-review demos.

### Streamer Mode

Streamer Mode turns expression spikes into presentation-style overlays.

- Reaction spike counting
- Short callouts such as `chat loved that reaction`
- Live mood confidence bars
- Theme changes that feel more like stream overlays than analytics panels

## Core Features

- Real-time webcam preview inside a dedicated viewport card
- Face detection using OpenCV Haar cascades
- Optional 68-point facial landmark tracking using OpenCV Facemark LBF
- Fallback feature tracking when landmarks are not available
- Emotion score estimation from facial geometry and simple heuristics
- Mood-driven animated UI themes built in PySide6
- Camera source switching on macOS with AVFoundation-backed enumeration
- Auto-download of the open-source LBF landmark model on first run

## How Emotion Detection Works

This project does not use a large pretrained deep-learning emotion model. Instead, it uses a lightweight geometry-based pipeline that is easier to run locally and easier to demo in a pure Python desktop app.

The processing flow is:

1. Capture a webcam frame with OpenCV.
2. Detect the largest visible face.
3. Try to fit facial landmarks.
4. If landmarks are available, compute expression metrics such as:
   - mouth width
   - mouth openness
   - smile curvature
   - eye openness
   - brow raise and brow furrow
5. Convert those metrics into heuristic emotion scores.
6. Select the dominant emotion and update the interface.

This makes the app lightweight and easy to run locally, but it also means emotion estimates are approximate and best suited for interactive demos rather than formal analysis.

## Project Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── session.py      # Session metrics, interview counters, streamer overlays
│   ├── ui.py           # PySide6 UI, animated themes, camera widget, mode panels
│   └── vision.py       # Face detection, landmarks, expression metrics, emotion scoring
├── main.py             # Application entry point
├── requirements.txt    # Python dependencies
└── README.md
```

## Requirements

- Python 3.10+ recommended
- Webcam access enabled for your terminal or editor
- macOS, Linux, or Windows with compatible wheels for the listed dependencies

The project has been tested in the current workspace with Python 3.14, but package compatibility still depends on available wheels for your platform.

## Installation

Create and activate a virtual environment if you want an isolated setup, then install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If you already know which Python interpreter you want to use, you can also install directly with that interpreter.

```bash
/usr/local/bin/python3 -m pip install -r requirements.txt
```

## Running the App

```bash
python main.py
```

Or, using the interpreter path that has been used in this workspace:

```bash
/usr/local/bin/python3 main.py
```

## First-Run Behavior

On the first run, Mood Mirror attempts to download the open-source OpenCV LBF landmark model into the local `models/` directory.

- If the download succeeds, the app can draw facial landmarks and use the higher-fidelity geometry path.
- If the download fails, the app still runs using cascade-based fallback tracking.

## Platform Notes

### macOS

- Camera enumeration uses AVFoundation when available.
- You may need to grant camera access to Terminal or VS Code in System Settings.
- If you have multiple camera sources, the app can switch between them from the UI.

### Virtual Environments

The repository intentionally ignores `.venv/`, `__pycache__/`, and generated model files via `.gitignore`.

## Troubleshooting

### The camera opens but the wrong source is selected

- Use the `Switch Camera` button in the app.
- Close virtual camera tools if they are taking priority.

### The webcam is visible but face tracking does not trigger

- Make sure your face is centered and well lit.
- Avoid extreme backlighting.
- Confirm the app reports either `Landmark tracking live` or `Using fallback feature tracking`.

### Interview mode counters look too low

Interview metrics now use direct facial signals, not only the dominant mood label. Smile rate and surprise moments should update when expression thresholds are crossed, but they still depend on lighting, camera angle, and how clearly the face is visible.

## Dependencies

Main libraries used in this project:

- `PySide6` for the desktop UI
- `opencv-contrib-python` for webcam capture, face detection, and facemark support
- `numpy` for geometry and numerical operations
- `requests` for downloading the landmark model
- `pyobjc-framework-AVFoundation` for native camera enumeration on macOS

## Use Cases

- Demo project for emotion-reactive UI concepts
- Webcam-based visual interaction experiments
- Mock interview feedback prototype
- Streamer overlay concept demo
- Desktop computer vision portfolio project

## Limitations

- Emotion inference is heuristic, not clinically validated
- Performance and camera compatibility depend on your local machine
- Landmark model availability depends on the initial model download
- The current implementation is a desktop app, not a hosted web application

## Future Improvements

- Replace heuristic emotion scoring with a pretrained expression model
- Add session export for interview metrics
- Add adjustable camera rendering presets such as raw or enhanced
- Package native builds for easier desktop distribution

## License / Usage

No explicit license file has been added yet. If you plan to publish the repository publicly, add a license before distributing it broadly.