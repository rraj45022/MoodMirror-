# Mood Mirror

Mood Mirror is being migrated from a single-process Python desktop app into a web architecture built around FastAPI, React, and Supabase.

The current repository now has two application surfaces:

- the original PySide6 desktop prototype, which still contains the live camera, expression analysis, and voice interview flow
- a new FastAPI backend plus React frontend scaffold that introduces login, user-scoped session persistence, and aggregated dashboard reporting

This is the architecture needed for hosted deployment and per-user data management. The desktop implementation remains valuable as the source of the current interview and expression logic while the web stack replaces the delivery layer.

## New Architecture

```text
.
├── app/                   # Existing desktop prototype modules
├── backend/
│   └── app/
│       ├── analytics.py   # Session aggregation logic adapted from SessionTracker
│       ├── db.py          # SQLite-backed persistence for users, tokens, sessions, samples, messages
│       ├── main.py        # FastAPI app and route definitions
│       ├── schemas.py     # Pydantic API contracts
│       └── security.py    # Password hashing and auth token helpers
├── frontend/
│   ├── package.json
│   ├── index.html
│   └── src/
│       ├── App.jsx        # Login, dashboard, and session workspace
│       ├── api.js         # API client
│       └── styles.css     # UI styling
├── main.py                # Existing desktop entrypoint
└── requirements.txt       # Python dependencies for desktop and API work
```

## What Has Changed

- `FastAPI backend`: supports register/login, bearer-token auth, per-user sessions, session messages, emotion samples, and dashboard aggregation
- `User-scoped persistence`: stored in Supabase Postgres through the backend service layer
- `React frontend`: adds a login screen, dashboard metrics, session creation, transcript capture, and emotion sample persistence
- `Session analytics`: backend summaries reuse the same event logic that previously lived only inside the in-memory desktop tracker

## Supabase Setup

Before the backend can start, create a Supabase project and provide these values in your local `.env`:

- `SUPABASE_URL`: your project URL
- `SUPABASE_SERVICE_ROLE_KEY`: the service-role key from Project Settings → API
- `FRONTEND_ORIGIN`: optional, defaults to `http://localhost:5173`

Then open the Supabase SQL Editor and run [backend/supabase/schema.sql](backend/supabase/schema.sql).

The current backend uses the service-role key only on the server side. Do not expose that key to the React app.

## API Surface

The new backend exposes these main routes:

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/me`
- `GET /api/dashboard/summary`
- `GET /api/sessions`
- `POST /api/sessions`
- `GET /api/sessions/{session_id}`
- `POST /api/sessions/{session_id}/messages`
- `POST /api/sessions/{session_id}/samples`
- `POST /api/sessions/{session_id}/complete`

## Local Development

### Backend

Create or activate your virtual environment, then install the Python requirements.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

The API will start on `http://127.0.0.1:8000` by default once Supabase is configured and the schema has been applied.

### Render Deployment

The repository now includes [render.yaml](render.yaml) plus [backend/requirements-render.txt](backend/requirements-render.txt) for deploying only the FastAPI backend on Render.

Use these settings on Render:

- Root Directory: repository root
- Build Command: `pip install -r backend/requirements-render.txt`
- Start Command: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`

Set these environment variables in Render before the first deploy:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `FRONTEND_ORIGIN` for one or more allowed production origins, separated by commas if needed
- `FRONTEND_ORIGIN_REGEX` if you want to allow preview domains such as Vercel preview URLs
- `REDIS_URL` for hosted Redis, not your local `localhost` instance

If you use the interview endpoints, also set:

- `GROQ_API_KEY`
- `GROQ_MODEL` or `GROQ_REVIEW_MODEL` for the heavier review pass
- `GROQ_INTERVIEW_MODEL` if you want to override the low-latency live interview model
- `GROQ_TRANSCRIPTION_MODEL`

If you want Google and GitHub login in the React app, also configure Supabase Auth providers and add your frontend URL as an allowed redirect URL. The frontend expects:

- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_PUBLISHABLE_KEY`

Then enable the Google and GitHub providers in Supabase Auth and point their redirect back to your frontend origin.

The hosted backend now uses browser-provided media capture from the React app. Render handles API, persistence, Groq requests, and uploaded frame analysis. `PySide6` remains a desktop-only dependency and is not required by the deployed FastAPI interview routes.

For a typical Vercel plus Render setup:

- set `FRONTEND_ORIGIN` to your main Vercel production URL
- optionally set `FRONTEND_ORIGIN_REGEX` to a regex such as `https://.*\\.vercel\\.app` if you want preview deploys to call the API too

### Frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

The React app defaults to `http://127.0.0.1:8000` for API calls. To point it somewhere else, set `VITE_API_URL` before running the dev server.

## Migration Direction

The current web scaffold intentionally solves the architectural foundation first:

- identity and login
- durable user session storage
- aggregate reporting per user
- a frontend shell for session workflows

The next migration step is tightening the browser-first delivery layer. That includes:

- webcam and microphone capture in React
- browser-side face analysis or streamed frame inference
- posting transcript turns and expression samples into the FastAPI session endpoints
- tightening auth and token lifecycle around production deployment requirements

## Existing Desktop Prototype

The original desktop app is still available and still runs with:

```bash
python3 main.py
```

That code is the behavioral reference for the remaining migration work, especially for:

- emotion scoring
- interview orchestration
- review generation
- live expression metrics

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
- `Start` and `End` controls for the live interview loop
- Automatic microphone listening after each interviewer prompt
- Groq-backed speech transcription and follow-up generation from your spoken answer
- Automatic transcript and macOS voice playback for interviewer prompts via the built-in `say` command

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
- Voice-driven AI interviewer panel with Groq chat completions and speech transcription support
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

Before starting the AI interview panel, add your Groq key to `.env` in the repository root.

```bash
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TRANSCRIPTION_MODEL=whisper-large-v3-turbo
```

If `.env` is missing or the key is empty, the interview panel can still show fallback prompts, but the automatic microphone-to-transcript loop will stay disabled.

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
- You may also need to grant microphone access for the live interview listener.
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

### The AI interviewer is not hearing my answer

- Confirm microphone permission for VS Code or Terminal in macOS Settings.
- Make sure `.env` contains a valid Groq key.
- The app waits for a short pause before sending the captured answer for transcription, so pause briefly after each answer.

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
- Voice-based mock interview practice with a live interviewer transcript and spoken prompts
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
