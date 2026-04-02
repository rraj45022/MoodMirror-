# Mood Mirror

Mood Mirror is a Python desktop app that turns live webcam expressions into a full-screen reactive interface.

## Features

- Live webcam preview with a face box and facial landmark overlay when the landmark model is available.
- Emotion-driven themes:
  - happy: bright warm gradients and floating light orbs
  - sad: blue rain particles
  - angry: red pulse rings and ember effects
  - surprise: confetti bursts
  - neutral: calm slate drift
- Interview mode with calmness, smile-rate, and surprise tracking.
- Streamer mode with confidence-spike callouts such as "chat loved that reaction".
- Fallback feature tracking if the landmark model is unavailable.

## Install

```bash
/usr/local/bin/python3 -m pip install -r requirements.txt
```

## Run

```bash
/usr/local/bin/python3 main.py
```

## Notes

- On first run, the app tries to download OpenCV's open-source LBF landmark model into the local `models/` directory.
- If the landmark download fails, the app still runs using OpenCV cascades and a lower-fidelity expression fallback.
- The current environment is Python 3.14, so package availability depends on wheels published for that version.