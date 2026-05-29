import React, { useEffect, useRef, useState } from "react";

import {
  addSessionMessages,
  addSessionSamples,
  analyzeSessionFrame,
  completeSession,
  createSession,
  fetchDashboard,
  fetchSession,
  fetchSessions,
  login,
  requestInterviewReview,
  register,
  respondToInterview,
  startInterview,
} from "./api";


const defaultReview = {
  overall_score: 78,
  answer_score: 76,
  expression_score: 80,
  summary: "Strong baseline session. Answers were structured, but a few responses could use more specific technical depth.",
  strengths: ["Clear pacing", "Strong composure", "Good use of examples"],
  brush_up_topics: ["System design tradeoffs", "Behavioral story precision"],
  answer_feedback: ["Add one metric-backed impact example", "Tighten long answers into three-step structures"],
  expression_feedback: ["Keep eye line stable between answers", "Use brief pauses instead of filler transitions"],
};

const emotionTemplates = {
  happy: { smile_curve: 0.16, mouth_width: 0.39, mouth_open: 0.06, eye_open: 0.32, brow_raise: 0.06 },
  neutral: { smile_curve: 0.02, mouth_width: 0.31, mouth_open: 0.03, eye_open: 0.29, brow_raise: 0.05 },
  surprise: { smile_curve: 0.01, mouth_width: 0.34, mouth_open: 0.12, eye_open: 0.36, brow_raise: 0.09 },
  sad: { smile_curve: -0.02, mouth_width: 0.28, mouth_open: 0.02, eye_open: 0.24, brow_raise: 0.03 },
  angry: { smile_curve: 0.0, mouth_width: 0.29, mouth_open: 0.02, eye_open: 0.25, brow_raise: 0.02, brow_furrow: 0.11 },
};

function speakInterviewerMessage(message) {
  if (!("speechSynthesis" in window) || !message) {
    return false;
  }

  const utterance = new SpeechSynthesisUtterance(message);
  utterance.rate = 1;
  utterance.pitch = 1;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
  return true;
}


function InterviewMediaPanel({ active, session, token, onSessionUpdate, onStatus, onError }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const animationFrameRef = useRef(null);
  const frameIntervalRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const [mediaState, setMediaState] = useState("idle");
  const [mediaError, setMediaError] = useState("");
  const [micLevel, setMicLevel] = useState(0);
  const [analysis, setAnalysis] = useState(null);
  const [interviewStarted, setInterviewStarted] = useState(false);
  const [recording, setRecording] = useState(false);
  const [processing, setProcessing] = useState(false);
  const sessionClosed = !session || session.status !== "active";

  useEffect(() => {
    if (!session || session.mode !== "interview") {
      setInterviewStarted(false);
      return;
    }

    const hasAssistantTurn = session.messages.some((message) => message.role === "assistant");
    setInterviewStarted(hasAssistantTurn && session.status === "active");
  }, [session]);

  useEffect(() => {
    if (!active) {
      cleanupMedia();
      setMediaState("idle");
      setMediaError("");
      setMicLevel(0);
      setAnalysis(null);
      setInterviewStarted(false);
      setRecording(false);
      setProcessing(false);
      return undefined;
    }

    let cancelled = false;

    async function startMedia() {
      setMediaState("loading");
      setMediaError("");

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => undefined);
        }
        setupMicMeter(stream);
        setMediaState("ready");
      } catch (error) {
        if (cancelled) {
          return;
        }
        setMediaState("error");
        const nextError = error instanceof Error ? error.message : "Could not access camera and microphone.";
        setMediaError(nextError);
        onError(nextError);
      }
    }

    startMedia();

    return () => {
      cancelled = true;
      cleanupMedia();
    };
  }, [active]);

  function setupMicMeter(stream) {
    cleanupAudioOnly();

    const audioContext = new AudioContext();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    const buffer = new Uint8Array(analyser.frequencyBinCount);

    audioContextRef.current = { audioContext, analyser, buffer, source };

    const tick = () => {
      analyser.getByteTimeDomainData(buffer);
      let sum = 0;
      for (let index = 0; index < buffer.length; index += 1) {
        const normalized = (buffer[index] - 128) / 128;
        sum += normalized * normalized;
      }
      const level = Math.min(100, Math.round(Math.sqrt(sum / buffer.length) * 180));
      setMicLevel(level);
      animationFrameRef.current = requestAnimationFrame(tick);
    };

    tick();
  }

  function cleanupAudioOnly() {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.source.disconnect();
      audioContextRef.current.analyser.disconnect();
      audioContextRef.current.audioContext.close().catch(() => undefined);
      audioContextRef.current = null;
    }
  }

  function cleanupMedia() {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    cleanupAudioOnly();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }

  function speakTestPrompt() {
    if (!("speechSynthesis" in window)) {
      setMediaError("Speech synthesis is not available in this browser.");
      return;
    }
    speakInterviewerMessage("Interviewer audio test. Camera preview and microphone monitor are active.");
  }

  useEffect(() => {
    if (!active || mediaState !== "ready" || !session || session.mode !== "interview") {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }
      return undefined;
    }

    let busy = false;

    async function pushFrame() {
      if (busy || !videoRef.current || !canvasRef.current || videoRef.current.readyState < 2) {
        return;
      }

      busy = true;
      const canvas = canvasRef.current;
      const video = videoRef.current;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 360;
      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.82));

      if (!blob) {
        busy = false;
        return;
      }

      try {
        const result = await analyzeSessionFrame(token, session.id, blob);
        setAnalysis(result.analysis);
        onSessionUpdate(result.session);
      } catch (error) {
        onError(error instanceof Error ? error.message : "Could not analyze the camera frame.");
      } finally {
        busy = false;
      }
    }

    pushFrame();
    frameIntervalRef.current = setInterval(pushFrame, 3000);
    return () => {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }
    };
  }, [active, mediaState, onError, onSessionUpdate, session, token]);

  async function handleStartInterview() {
    if (!session || sessionClosed) {
      return;
    }
    setProcessing(true);
    try {
      const result = await startInterview(token, session.id);
      onSessionUpdate(result.session);
      setInterviewStarted(true);
      onStatus(result.assistant_message);
      speakInterviewerMessage(result.assistant_message);
    } catch (error) {
      onError(error instanceof Error ? error.message : "Could not start the interview.");
    } finally {
      setProcessing(false);
    }
  }

  function handleRecordAnswer() {
    if (sessionClosed) {
      return;
    }

    if (!streamRef.current) {
      onError("Camera and microphone are not ready yet.");
      return;
    }

    if (recording) {
      onStatus("Processing your answer...");
      setProcessing(true);
      mediaRecorderRef.current?.stop();
      setRecording(false);
      return;
    }

    const audioTracks = streamRef.current.getAudioTracks();
    if (!audioTracks.length) {
      onError("No microphone track is available.");
      return;
    }

    const mimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find((type) => MediaRecorder.isTypeSupported(type)) || "";
    const recorder = new MediaRecorder(new MediaStream(audioTracks), mimeType ? { mimeType } : undefined);
    audioChunksRef.current = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data);
      }
    };
    recorder.onstop = async () => {
      const blob = new Blob(audioChunksRef.current, { type: recorder.mimeType || "audio/webm" });
      if (!blob.size) {
        onError("The recording was empty. Try speaking again.");
        return;
      }

      try {
        const result = await respondToInterview(token, session.id, blob);
        onSessionUpdate(result.session);
        onStatus(`You said: ${result.transcript}`);
        speakInterviewerMessage(result.assistant_message);
      } catch (error) {
        onError(error instanceof Error ? error.message : "Could not process the recorded answer.");
      } finally {
        setProcessing(false);
      }
    };

    recorder.start(250);
    mediaRecorderRef.current = recorder;
    setInterviewStarted(true);
    setRecording(true);
    onStatus("Recording your answer...");
  }

  return (
    <section className="media-panel">
      <canvas hidden ref={canvasRef} />
      <div className="media-frame">
        <video autoPlay className="media-video" muted playsInline ref={videoRef} />
        {mediaState !== "ready" ? (
          <div className="media-overlay">
            <strong>{mediaState === "loading" ? "Connecting camera and microphone..." : "Camera preview offline"}</strong>
            <p>
              {mediaState === "error"
                ? mediaError || "Browser permissions are blocking camera or microphone access."
                : "Choose an interview session and allow browser permissions to show the live preview."}
            </p>
          </div>
        ) : null}
      </div>

      <div className="media-toolbar">
        <div className="mic-card">
          <p className="panel-label">Microphone input</p>
          <div aria-hidden="true" className="mic-meter-track">
            <div className="mic-meter-fill" style={{ width: `${micLevel}%` }} />
          </div>
          <strong>{micLevel}% live level</strong>
          <p className="media-status-text">
            {analysis
              ? `${analysis.emotion} at ${Math.round(analysis.confidence * 100)}% confidence`
              : mediaState === "ready"
                ? "Waiting for face analysis..."
                : "Waiting for media permissions..."}
          </p>
        </div>

        <div className="media-actions">
          <button className="primary" disabled={processing || interviewStarted || sessionClosed} onClick={handleStartInterview} type="button">
            {processing && !interviewStarted ? "Starting..." : "Start interview"}
          </button>
          <button className="ghost" disabled={processing || mediaState !== "ready" || !interviewStarted || sessionClosed} onClick={handleRecordAnswer} type="button">
            {processing ? "Processing..." : recording ? "Stop recording" : "Record answer"}
          </button>
          <button className="ghost" onClick={speakTestPrompt} type="button">Test interviewer voice</button>
          <p className="media-note">Preview uses browser camera and microphone access on `localhost`.</p>
        </div>
      </div>
    </section>
  );
}


function App() {
  const [token, setToken] = useState(() => localStorage.getItem("mood-mirror-token") || "");
  const [user, setUser] = useState(null);
  const [dashboard, setDashboard] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [authMode, setAuthMode] = useState("login");
  const [authForm, setAuthForm] = useState({ email: "", password: "", display_name: "" });
  const [sessionForm, setSessionForm] = useState({ title: "Frontend practice round", mode: "interview" });
  const [messageDraft, setMessageDraft] = useState("I recently led a migration that reduced API latency by 28%.");
  const [messageRole, setMessageRole] = useState("user");
  const [statusMessage, setStatusMessage] = useState("Connect the web client to start storing sessions per user.");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setInterviewStateFromSession(selectedSession);
  }, [selectedSession?.id]);

  function setInterviewStateFromSession(session) {
    if (!session || session.mode !== "interview") {
      return;
    }
    const latestAssistant = [...session.messages].reverse().find((message) => message.role === "assistant");
    if (latestAssistant) {
      setStatusMessage(latestAssistant.content);
    }
  }

  function syncSessionDetail(detail) {
    setSelectedSession(detail);
    setSessions((currentSessions) => {
      const summary = {
        id: detail.id,
        user_id: detail.user_id,
        mode: detail.mode,
        title: detail.title,
        status: detail.status,
        started_at: detail.started_at,
        completed_at: detail.completed_at,
        duration_seconds: detail.duration_seconds,
        calmness_percent: detail.calmness_percent,
        smile_events: detail.smile_events,
        surprise_events: detail.surprise_events,
        reaction_spikes: detail.reaction_spikes,
        smiles_per_minute: detail.smiles_per_minute,
        dominant_emotion: detail.dominant_emotion,
        mood_mix: detail.mood_mix,
        total_samples: detail.total_samples,
        transcript_turns: detail.transcript_turns,
        latest_expression: detail.latest_expression,
        overall_score: detail.overall_score,
      };
      const existingIndex = currentSessions.findIndex((session) => session.id === detail.id);
      if (existingIndex === -1) {
        return [summary, ...currentSessions];
      }
      const nextSessions = [...currentSessions];
      nextSessions[existingIndex] = summary;
      return nextSessions;
    });
  }

  useEffect(() => {
    if (!token) {
      setDashboard(null);
      setSessions([]);
      setSelectedSession(null);
      setUser(null);
      return;
    }

    let active = true;
    setLoading(true);
    Promise.all([fetchDashboard(token), fetchSessions(token)])
      .then(([dashboardResponse, sessionsResponse]) => {
        if (!active) {
          return;
        }
        setDashboard(dashboardResponse);
        setSessions(sessionsResponse);
        setUser(dashboardResponse.user);
        if (sessionsResponse.length && !selectedSession) {
          void selectSession(sessionsResponse[0].id, token);
        }
      })
      .catch((requestError) => {
        if (!active) {
          return;
        }
        setError(requestError.message);
        clearAuth();
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [token]);

  async function handleAuthSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = authMode === "login" ? await login(authForm) : await register(authForm);
      localStorage.setItem("mood-mirror-token", response.token);
      setToken(response.token);
      setUser(response.user);
      setStatusMessage("Authentication complete. User-scoped dashboard loaded.");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  function clearAuth() {
    localStorage.removeItem("mood-mirror-token");
    setToken("");
  }

  async function refreshData(activeToken = token) {
    const [dashboardResponse, sessionsResponse] = await Promise.all([
      fetchDashboard(activeToken),
      fetchSessions(activeToken),
    ]);
    setDashboard(dashboardResponse);
    setSessions(sessionsResponse);
    setUser(dashboardResponse.user);
  }

  async function selectSession(sessionId, activeToken = token) {
    const detail = await fetchSession(activeToken, sessionId);
    setSelectedSession(detail);
    return detail;
  }

  async function handleCreateSession(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const created = await createSession(token, sessionForm);
      if (created.mode === "interview") {
        const started = await startInterview(token, created.id);
        syncSessionDetail(started.session);
        await refreshData();
        setStatusMessage(started.assistant_message);
        speakInterviewerMessage(started.assistant_message);
      } else {
        const detail = await selectSession(created.id);
        syncSessionDetail(detail);
        await refreshData();
        setStatusMessage(`Created ${created.mode} session for ${created.title}.`);
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleAddMessage(event) {
    event.preventDefault();
    if (!selectedSession) {
      return;
    }
    setLoading(true);
    setError("");
    try {
      const detail = await addSessionMessages(token, selectedSession.id, [{ role: messageRole, content: messageDraft }]);
      syncSessionDetail(detail);
      await refreshData();
      setMessageDraft("");
      setStatusMessage("Transcript turn stored against the active user session.");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleAddEmotion(emotion) {
    if (!selectedSession) {
      return;
    }
    setLoading(true);
    setError("");
    try {
      const metrics = emotionTemplates[emotion];
      const scores = Object.fromEntries(Object.keys(emotionTemplates).map((item) => [item, item === emotion ? 0.82 : 0.08]));
      const detail = await addSessionSamples(token, selectedSession.id, [{ emotion, confidence: 0.82, metrics, scores }]);
      syncSessionDetail(detail);
      setStatusMessage(`Captured a ${emotion} sample and refreshed aggregate metrics.`);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleCompleteSession() {
    if (!selectedSession) {
      return;
    }
    setLoading(true);
    setError("");
    try {
      const detail = selectedSession.mode === "interview"
        ? (await requestInterviewReview(token, selectedSession.id)).session
        : await completeSession(token, selectedSession.id, { review: defaultReview });
      syncSessionDetail(detail);
      await refreshData();
      setStatusMessage(selectedSession.mode === "interview" ? "Interview ended and review snapshot stored." : "Session completed and review snapshot stored.");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  if (!token) {
    return (
      <main className="shell shell-auth">
        <section className="auth-card">
          <p className="eyebrow">Mood Mirror Web</p>
          <h1>Login-first interview analytics.</h1>
          <p className="lede">FastAPI stores user sessions and aggregate metrics. React becomes the delivery surface for webcam, transcript, and review flows.</p>
          <div className="toggle-row">
            <button className={authMode === "login" ? "active" : ""} onClick={() => setAuthMode("login")} type="button">Login</button>
            <button className={authMode === "register" ? "active" : ""} onClick={() => setAuthMode("register")} type="button">Register</button>
          </div>
          <form className="stack" onSubmit={handleAuthSubmit}>
            {authMode === "register" ? (
              <label>
                Display name
                <input value={authForm.display_name} onChange={(event) => setAuthForm({ ...authForm, display_name: event.target.value })} required />
              </label>
            ) : null}
            <label>
              Email
              <input type="email" value={authForm.email} onChange={(event) => setAuthForm({ ...authForm, email: event.target.value })} required />
            </label>
            <label>
              Password
              <input type="password" value={authForm.password} onChange={(event) => setAuthForm({ ...authForm, password: event.target.value })} required minLength={8} />
            </label>
            <button className="primary" disabled={loading} type="submit">{loading ? "Working..." : authMode === "login" ? "Login" : "Create account"}</button>
          </form>
          {error ? <p className="error-banner">{error}</p> : null}
        </section>
      </main>
    );
  }

  return (
    <main className="shell">
      <header className="hero-card">
        <div>
          <p className="eyebrow">Session architecture refresh</p>
          <h1>{user ? `${user.display_name}'s dashboard` : "Dashboard"}</h1>
          <p className="lede">User-scoped storage is now the center of the app. Browser capture and AI interview flow can post samples and transcript turns into the same backend session record.</p>
        </div>
        <div className="hero-actions">
          <p>{statusMessage}</p>
          <button className="ghost" onClick={clearAuth} type="button">Logout</button>
        </div>
      </header>

      {error ? <p className="error-banner">{error}</p> : null}

      <section className="grid metrics-grid">
        <MetricCard label="Total sessions" value={dashboard?.total_sessions ?? 0} />
        <MetricCard label="Completed" value={dashboard?.completed_sessions ?? 0} />
        <MetricCard label="Average calmness" value={`${dashboard?.average_calmness ?? 0}%`} />
        <MetricCard label="Average score" value={dashboard?.average_score ? `${dashboard.average_score}/100` : "Pending"} />
      </section>

      <section className="grid content-grid">
        <aside className="panel stack gap-lg">
          <div>
            <p className="panel-label">Create session</p>
            <form className="stack" onSubmit={handleCreateSession}>
              <label>
                Session title
                <input value={sessionForm.title} onChange={(event) => setSessionForm({ ...sessionForm, title: event.target.value })} required />
              </label>
              <label>
                Mode
                <select value={sessionForm.mode} onChange={(event) => setSessionForm({ ...sessionForm, mode: event.target.value })}>
                  <option value="interview">Interview</option>
                  <option value="mirror">Mirror</option>
                  <option value="streamer">Streamer</option>
                </select>
              </label>
              <button className="primary" disabled={loading} type="submit">Start tracked session</button>
            </form>
          </div>

          <div>
            <p className="panel-label">Recent sessions</p>
            <div className="session-list">
              {sessions.map((session) => (
                <button
                  className={`session-item ${selectedSession?.id === session.id ? "selected" : ""}`}
                  key={session.id}
                  onClick={() => void selectSession(session.id)}
                  type="button"
                >
                  <strong>{session.title}</strong>
                  <span>{session.mode} · {session.status}</span>
                  <span>{session.latest_expression}</span>
                </button>
              ))}
            </div>
          </div>
        </aside>

        <section className="panel stack gap-lg wide-panel">
          <div className="workspace-head">
            <div>
              <p className="panel-label">Session workspace</p>
              <h2>{selectedSession?.title || "Create or pick a session"}</h2>
            </div>
            {selectedSession ? (
              <button className="primary" disabled={loading || selectedSession.status === "completed"} onClick={() => void handleCompleteSession()} type="button">
                {selectedSession.status === "completed" ? "Session completed" : selectedSession.mode === "interview" ? "End interview" : "Complete session"}
              </button>
            ) : null}
          </div>

          {selectedSession ? (
            <>
              {selectedSession.mode === "interview" ? (
                <InterviewMediaPanel
                  active
                  onError={(message) => setError(message)}
                  onSessionUpdate={(detail) => {
                    syncSessionDetail(detail);
                  }}
                  onStatus={setStatusMessage}
                  session={selectedSession}
                  token={token}
                />
              ) : null}

              <div className="grid detail-grid">
                <MetricCard label="Calmness" value={`${selectedSession.calmness_percent}%`} />
                <MetricCard label="Smiles" value={selectedSession.smile_events} />
                <MetricCard label="Surprises" value={selectedSession.surprise_events} />
                <MetricCard label="Turns" value={selectedSession.transcript_turns} />
              </div>

              <div className="stack gap-md">
                <p className="panel-label">Quick emotion capture</p>
                <div className="emotion-row">
                  {Object.keys(emotionTemplates).map((emotion) => (
                    <button disabled={loading || selectedSession.status === "completed"} key={emotion} onClick={() => void handleAddEmotion(emotion)} type="button">
                      {emotion}
                    </button>
                  ))}
                </div>
              </div>

              <form className="stack gap-md" onSubmit={handleAddMessage}>
                <p className="panel-label">Transcript staging</p>
                <label>
                  Speaker
                  <select value={messageRole} onChange={(event) => setMessageRole(event.target.value)}>
                    <option value="user">User</option>
                    <option value="assistant">Interviewer</option>
                    <option value="system">System</option>
                  </select>
                </label>
                <label>
                  Transcript content
                  <textarea rows="4" value={messageDraft} onChange={(event) => setMessageDraft(event.target.value)} />
                </label>
                <button className="ghost" disabled={loading || selectedSession.status === "completed"} type="submit">Store transcript turn</button>
              </form>

              <div className="stack gap-md">
                <p className="panel-label">Stored transcript</p>
                <div className="transcript-list">
                  {selectedSession.messages.map((message, index) => (
                    <article className="transcript-item" key={`${message.created_at}-${index}`}>
                      <strong>{message.role}</strong>
                      <p>{message.content}</p>
                    </article>
                  ))}
                </div>
              </div>

              {selectedSession.review ? (
                <div className="review-card">
                  <p className="panel-label">Session review</p>
                  <h3>{selectedSession.review.overall_score}/100 overall</h3>
                  <p>{selectedSession.review.summary}</p>
                </div>
              ) : null}
            </>
          ) : (
            <div className="empty-state">
              <p>Select a session to start storing transcript and expression data.</p>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}


function MetricCard({ label, value }) {
  return (
    <article className="metric-card">
      <p>{label}</p>
      <strong>{value}</strong>
    </article>
  );
}


export default App;