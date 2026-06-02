import React, { useEffect, useRef, useState } from "react";

import {
  addSessionMessages,
  addSessionSamples,
  analyzeSessionFrameBatch,
  analyzeSessionFrame,
  createSession,
  deleteSession,
  fetchDashboard,
  fetchSession,
  fetchSessions,
  login,
  oauthLogin,
  requestInterviewReview,
  register,
  respondToInterview,
  startInterview,
} from "./api";
import { supabase } from "./supabase";


const emotionTemplates = {
  happy: { smile_curve: 0.16, mouth_width: 0.39, mouth_open: 0.06, eye_open: 0.32, brow_raise: 0.06 },
  neutral: { smile_curve: 0.02, mouth_width: 0.31, mouth_open: 0.03, eye_open: 0.29, brow_raise: 0.05 },
  surprise: { smile_curve: 0.01, mouth_width: 0.34, mouth_open: 0.12, eye_open: 0.36, brow_raise: 0.09 },
  sad: { smile_curve: -0.02, mouth_width: 0.28, mouth_open: 0.02, eye_open: 0.24, brow_raise: 0.03 },
  angry: { smile_curve: 0.0, mouth_width: 0.29, mouth_open: 0.02, eye_open: 0.25, brow_raise: 0.02, brow_furrow: 0.11 },
};

function formatDuration(totalSeconds) {
  if (!totalSeconds) {
    return "Just started";
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (!minutes) {
    return `${seconds}s`;
  }
  if (!seconds) {
    return `${minutes}m`;
  }
  return `${minutes}m ${seconds}s`;
}

function average(values) {
  if (!values.length) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function formatTimelineDate(timestamp) {
  if (!timestamp) {
    return "Now";
  }
  return new Date(timestamp * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function buildProgressView(sessions) {
  const completedSessions = [...sessions]
    .filter((session) => session.status === "completed")
    .sort((left, right) => (left.completed_at || left.started_at) - (right.completed_at || right.started_at));

  const scoreSessions = completedSessions.filter((session) => typeof session.overall_score === "number");
  const baselineWindow = scoreSessions.slice(0, Math.min(3, scoreSessions.length));
  const recentWindow = scoreSessions.slice(-Math.min(3, scoreSessions.length));
  const calmnessBaseline = completedSessions.slice(0, Math.min(3, completedSessions.length));
  const calmnessRecent = completedSessions.slice(-Math.min(3, completedSessions.length));
  const emotionCounts = completedSessions.reduce((counts, session) => {
    const emotion = session.dominant_emotion || "neutral";
    return { ...counts, [emotion]: (counts[emotion] || 0) + 1 };
  }, {});

  return {
    completedSessions,
    scoreDelta: scoreSessions.length > 1
      ? Math.round(average(recentWindow.map((session) => session.overall_score)) - average(baselineWindow.map((session) => session.overall_score)))
      : null,
    calmnessDelta: completedSessions.length > 1
      ? Math.round(average(calmnessRecent.map((session) => session.calmness_percent)) - average(calmnessBaseline.map((session) => session.calmness_percent)))
      : null,
    totalPracticeMinutes: Math.round(completedSessions.reduce((sum, session) => sum + session.duration_seconds, 0) / 60),
    bestScore: scoreSessions.length ? Math.max(...scoreSessions.map((session) => session.overall_score)) : null,
    emotionBreakdown: Object.entries(emotionCounts)
      .sort((left, right) => right[1] - left[1])
      .slice(0, 5)
      .map(([emotion, count]) => ({
        emotion,
        count,
        share: completedSessions.length ? Math.round((count / completedSessions.length) * 100) : 0,
      })),
  };
}

function getWrapUp(session) {
  if (!session?.review) {
    return null;
  }

  return {
    score: session.review.overall_score,
    summary: session.review.summary,
    strengths: session.review.strengths.slice(0, 3),
    brushUp: session.review.brush_up_topics.slice(0, 3),
    answerFeedback: session.review.answer_feedback.slice(0, 2),
    expressionFeedback: session.review.expression_feedback.slice(0, 2),
  };
}

function getDashboardStatus({ dashboard, selectedSession, wrapUp }) {
  if (wrapUp) {
    return `Latest wrap-up ready: ${wrapUp.score}/100 overall with fresh coaching points.`;
  }

  if (selectedSession?.status === "active") {
    return `Active interview ready: ${selectedSession.title}. Enter the interview room when you want to start live capture.`;
  }

  if ((dashboard?.total_sessions ?? 0) > 0) {
    return `${dashboard?.completed_sessions ?? 0} completed interviews, ${dashboard?.llm_calls_remaining_today ?? 0} LLM calls left today.`;
  }

  return "Create your first interview session to start building a performance baseline.";
}

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


function InterviewMediaPanel({ active, endingSession, session, token, onEndInterview, onSessionUpdate, onStatus, onError }) {
  const FRAME_CAPTURE_INTERVAL_MS = 3000;
  const MAX_BUFFERED_FRAMES = 12;
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const animationFrameRef = useRef(null);
  const frameIntervalRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const bufferedFramesRef = useRef([]);
  const flushFramesPromiseRef = useRef(null);
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
      bufferedFramesRef.current = [];
      flushFramesPromiseRef.current = null;
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
    bufferedFramesRef.current = [];
    flushFramesPromiseRef.current = null;
  }

  async function captureFrameBlob() {
    if (!videoRef.current || !canvasRef.current || videoRef.current.readyState < 2) {
      return null;
    }

    const canvas = canvasRef.current;
    const video = videoRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 360;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.82));
    if (!blob) {
      return null;
    }

    return { blob, recordedAt: Date.now() / 1000 };
  }

  async function flushBufferedFrames(sessionId) {
    if (!bufferedFramesRef.current.length) {
      return;
    }
    if (flushFramesPromiseRef.current) {
      await flushFramesPromiseRef.current;
      return;
    }

    const frames = bufferedFramesRef.current;
    bufferedFramesRef.current = [];
    const flushPromise = analyzeSessionFrameBatch(token, sessionId, frames)
      .then((result) => {
        setAnalysis(result.analysis);
        onSessionUpdate(result.session);
      })
      .catch((error) => {
        onError(error instanceof Error ? error.message : "Could not sync buffered face analysis.");
        bufferedFramesRef.current = [...frames, ...bufferedFramesRef.current].slice(-MAX_BUFFERED_FRAMES);
      })
      .finally(() => {
        flushFramesPromiseRef.current = null;
      });

    flushFramesPromiseRef.current = flushPromise;
    await flushPromise;
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
      if (busy || flushFramesPromiseRef.current) {
        return;
      }

      busy = true;
      const frame = await captureFrameBlob();
      if (!frame) {
        busy = false;
        return;
      }

      try {
        if (recording) {
          bufferedFramesRef.current = [...bufferedFramesRef.current, frame].slice(-MAX_BUFFERED_FRAMES);
          return;
        }
        if (processing) {
          return;
        }
        const result = await analyzeSessionFrame(token, session.id, frame.blob);
        setAnalysis(result.analysis);
        onSessionUpdate(result.session);
      } catch (error) {
        onError(error instanceof Error ? error.message : "Could not analyze the camera frame.");
      } finally {
        busy = false;
      }
    }

    pushFrame();
    frameIntervalRef.current = setInterval(pushFrame, FRAME_CAPTURE_INTERVAL_MS);
    return () => {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }
    };
  }, [active, mediaState, onError, onSessionUpdate, processing, recording, session, token]);

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
        setProcessing(false);
        return;
      }

      try {
        const result = await respondToInterview(token, session.id, blob);
        onSessionUpdate(result.session);
        onStatus(`You said: ${result.transcript}`);
        speakInterviewerMessage(result.assistant_message);
        void flushBufferedFrames(session.id);
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
          <div className="toggle-row interview-toggle" role="group" aria-label="Interview controls">
            <button
              className={interviewStarted ? "" : "active"}
              disabled={processing || endingSession || interviewStarted || sessionClosed}
              onClick={handleStartInterview}
              type="button"
            >
              {processing && !interviewStarted ? "Starting..." : "Start interview"}
            </button>
            <button
              className={interviewStarted && !sessionClosed ? "active" : ""}
              disabled={processing || endingSession || !interviewStarted || sessionClosed}
              onClick={() => void onEndInterview()}
              type="button"
            >
              {endingSession ? "Ending..." : "End interview"}
            </button>
          </div>
          <button className="ghost" disabled={processing || mediaState !== "ready" || !interviewStarted || sessionClosed} onClick={handleRecordAnswer} type="button">
            {processing ? "Processing..." : recording ? "Stop recording" : "Record answer"}
          </button>
          <button className="ghost" onClick={speakTestPrompt} type="button">Test interviewer voice</button>
          <p className="media-note">Preview uses browser camera and microphone access in any supported secure browser session.</p>
        </div>
      </div>
    </section>
  );
}


function App() {
  const oauthSyncRef = useRef(false);
  const [token, setToken] = useState(() => localStorage.getItem("mood-mirror-token") || "");
  const [user, setUser] = useState(null);
  const [dashboard, setDashboard] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [activeView, setActiveView] = useState("dashboard");
  const [endingSession, setEndingSession] = useState(false);
  const [recentWrapUp, setRecentWrapUp] = useState(null);
  const [authMode, setAuthMode] = useState("login");
  const [authForm, setAuthForm] = useState({ email: "", password: "", display_name: "" });
  const [sessionForm, setSessionForm] = useState({ title: "Frontend practice round", mode: "interview" });
  const [messageDraft, setMessageDraft] = useState("I recently led a migration that reduced API latency by 28%.");
  const [messageRole, setMessageRole] = useState("user");
  const [statusMessage, setStatusMessage] = useState("Connect the web client to start storing sessions per user.");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const socialAuthEnabled = Boolean(supabase);

  useEffect(() => {
    if (activeView !== "interview") {
      return;
    }
    setInterviewStateFromSession(selectedSession);
  }, [activeView, selectedSession]);

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
    if (detail.review) {
      setRecentWrapUp(detail);
    }
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
      setActiveView("dashboard");
      setRecentWrapUp(null);
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
        const interviewSessions = sessionsResponse.filter((session) => session.mode === "interview");
        setDashboard(dashboardResponse);
        setSessions(interviewSessions);
        setUser(dashboardResponse.user);
        if (interviewSessions.length && !selectedSession) {
          void selectSession(interviewSessions[0].id, token);
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

  useEffect(() => {
    if (!supabase) {
      return undefined;
    }

    async function restoreOAuthSession() {
      if (token || oauthSyncRef.current) {
        return;
      }
      const { data, error: sessionError } = await supabase.auth.getSession();
      if (sessionError || !data.session) {
        return;
      }
      await exchangeOAuthSession(data.session, true);
    }

    void restoreOAuthSession();

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, session) => {
      if (!session || token || oauthSyncRef.current) {
        return;
      }
      if (event === "SIGNED_IN" || event === "INITIAL_SESSION" || event === "TOKEN_REFRESHED") {
        void exchangeOAuthSession(session, true);
      }
    });

    return () => {
      subscription.unsubscribe();
    };
  }, [token]);

  async function exchangeOAuthSession(session, silent = false) {
    if (!session?.access_token || oauthSyncRef.current) {
      return;
    }
    oauthSyncRef.current = true;
    if (!silent) {
      setLoading(true);
      setError("");
    }
    try {
      const response = await oauthLogin({
        access_token: session.access_token,
        provider: session.user?.app_metadata?.provider || null,
      });
      localStorage.setItem("mood-mirror-token", response.token);
      setToken(response.token);
      setUser(response.user);
      setStatusMessage("Authentication complete. User-scoped dashboard loaded.");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      oauthSyncRef.current = false;
      if (!silent) {
        setLoading(false);
      }
    }
  }

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

  async function handleSocialAuth(provider) {
    if (!supabase) {
      setError("Supabase OAuth is not configured for this frontend.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const { error: authError } = await supabase.auth.signInWithOAuth({
        provider,
        options: {
          redirectTo: window.location.origin,
        },
      });
      if (authError) {
        throw authError;
      }
    } catch (requestError) {
      setError(requestError.message);
      setLoading(false);
    }
  }

  function clearAuth() {
    if (supabase) {
      void supabase.auth.signOut();
    }
    localStorage.removeItem("mood-mirror-token");
    setToken("");
    setUser(null);
  }

  async function refreshData(activeToken = token) {
    const [dashboardResponse, sessionsResponse] = await Promise.all([
      fetchDashboard(activeToken),
      fetchSessions(activeToken),
    ]);
    const interviewSessions = sessionsResponse.filter((session) => session.mode === "interview");
    setDashboard(dashboardResponse);
    setSessions(interviewSessions);
    setUser(dashboardResponse.user);
    return { dashboard: dashboardResponse, sessions: interviewSessions };
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
      const created = await createSession(token, { ...sessionForm, mode: "interview" });
      const detail = await selectSession(created.id);
      syncSessionDetail(detail);
      await refreshData();
      setActiveView("dashboard");
      setStatusMessage(`Interview session ready for ${created.title}. Open the interview room when you want to start.`);
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
    setEndingSession(true);
    setError("");
    try {
      const detail = (await requestInterviewReview(token, selectedSession.id)).session;
      syncSessionDetail(detail);
      await refreshData();
      setRecentWrapUp(detail);
      setActiveView("dashboard");
      setStatusMessage("Interview ended and your session wrap-up is ready.");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setEndingSession(false);
    }
  }

  async function handleDeleteSession() {
    if (!selectedSession) {
      return;
    }

    const sessionToDelete = selectedSession;
    const confirmed = window.confirm(`Delete \"${sessionToDelete.title}\" and all of its stored transcript and expression data?`);
    if (!confirmed) {
      return;
    }

    setLoading(true);
    setError("");
    try {
      await deleteSession(token, sessionToDelete.id);
      const { sessions: remainingSessions } = await refreshData();
      const nextSession = remainingSessions.find((session) => session.id !== sessionToDelete.id) || null;

      setActiveView("dashboard");

      if (nextSession) {
        await selectSession(nextSession.id);
      } else {
        setSelectedSession(null);
        if (recentWrapUp?.id === sessionToDelete.id) {
          setRecentWrapUp(null);
        }
      }

      if (recentWrapUp?.id === sessionToDelete.id) {
        setRecentWrapUp(null);
      }
      setStatusMessage(`Deleted session data for ${sessionToDelete.title}.`);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  function handleEnterInterview() {
    if (!selectedSession) {
      return;
    }
    setActiveView("interview");
    setStatusMessage(selectedSession.status === "completed" ? "Review is ready below. Start a fresh session to practice again." : "Interview room ready. Camera will connect when this screen opens.");
  }

  const selectedWrapUp = getWrapUp(selectedSession);
  const dashboardWrapUp = getWrapUp(recentWrapUp) || selectedWrapUp;
  const progressView = buildProgressView(sessions);
  const canEnterInterview = selectedSession && selectedSession.status === "active";
  const heroMessage = activeView === "dashboard"
    ? getDashboardStatus({ dashboard, selectedSession, wrapUp: dashboardWrapUp })
    : statusMessage;

  if (!token) {
    return (
      <main className="shell shell-auth">
        <div className="shell-auth-stack">
          <section className="auth-card">
            <p className="eyebrow">Interview Presence Coach Web</p>
            <h1>Login-first interview analytics.</h1>
            <p className="lede">Practice mock interviews, review your session history, and track how your confidence and performance improve over time.</p>
            <div className="social-auth-grid">
              <button className="oauth-button" disabled={loading || !socialAuthEnabled} onClick={() => void handleSocialAuth("google")} type="button">
                Continue with Google
              </button>
              <button className="oauth-button" disabled={loading || !socialAuthEnabled} onClick={() => void handleSocialAuth("github")} type="button">
                Continue with GitHub
              </button>
            </div>
            <p className="auth-helper-text">
              {socialAuthEnabled
                ? "Social sign-in uses Supabase OAuth and then links that identity to your Interview Presence Coach history."
                : "Add VITE_SUPABASE_URL and VITE_SUPABASE_PUBLISHABLE_KEY to enable Google and GitHub sign-in."}
            </p>
            <div className="auth-divider"><span>or continue with email</span></div>
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
          <SiteFooter />
        </div>
      </main>
    );
  }

  return (
    <main className="shell">
      <header className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Interview practice hub</p>
          <h1>{user ? `${user.display_name}'s dashboard` : "Dashboard"}</h1>
          <p className="lede">Start from your session overview, review your aggregate performance, then enter the interview room only when you are ready for camera and microphone capture.</p>
        </div>
        <div className="hero-actions">
          <p className="hero-status">{heroMessage}</p>
          <button className="ghost" onClick={clearAuth} type="button">Logout</button>
        </div>
      </header>

      {error ? <p className="error-banner">{error}</p> : null}

      <section className="grid metrics-grid">
        <MetricCard label="Total sessions" value={dashboard?.total_sessions ?? 0} />
        <MetricCard label="Completed" value={dashboard?.completed_sessions ?? 0} />
        <MetricCard label="Average calmness" value={`${dashboard?.average_calmness ?? 0}%`} />
        <MetricCard label="Calls left today" value={dashboard ? `${dashboard.llm_calls_remaining_today}/${dashboard.daily_llm_call_limit}` : "--"} />
      </section>

      <section className="panel progress-panel">
        <div className="workspace-head">
          <div>
            <p className="panel-label">Progress view</p>
            <h2>How your practice is moving</h2>
          </div>
          <div className="workspace-actions progress-summary-row">
            <TrendBadge label="Score trend" value={progressView.scoreDelta} suffix="pts" />
            <TrendBadge label="Calmness trend" value={progressView.calmnessDelta} suffix="pts" />
            <TrendBadge label="Practice time" value={progressView.totalPracticeMinutes || null} suffix="min" neutral />
            <TrendBadge label="Best score" value={progressView.bestScore} suffix="/100" neutral />
          </div>
        </div>
        <div className="grid progress-grid">
          <ProgressChart sessions={progressView.completedSessions} />
          <EmotionBreakdown items={progressView.emotionBreakdown} />
        </div>
      </section>

      {activeView === "dashboard" ? (
        <section className="grid content-grid">
          <aside className="panel stack gap-lg">
            <div>
              <p className="panel-label">Create session</p>
              <form className="stack" onSubmit={handleCreateSession}>
                <label>
                  Session title
                  <input value={sessionForm.title} onChange={(event) => setSessionForm({ ...sessionForm, title: event.target.value })} required />
                </label>
                <button className="primary" disabled={loading} type="submit">{loading ? "Creating..." : "Create interview session"}</button>
              </form>
            </div>

            <div>
              <p className="panel-label">Interview sessions</p>
              <div className="session-list">
                {sessions.map((session) => (
                  <button
                    className={`session-item ${selectedSession?.id === session.id ? "selected" : ""}`}
                    key={session.id}
                    onClick={() => void selectSession(session.id)}
                    type="button"
                  >
                    <strong>{session.title}</strong>
                    <span>{session.status} · {formatDuration(session.duration_seconds)}</span>
                    <span>{session.overall_score ? `${session.overall_score}/100 overall` : session.latest_expression}</span>
                  </button>
                ))}
              </div>
            </div>
          </aside>

          <section className="panel stack gap-lg wide-panel">
            <div className="workspace-head">
              <div>
                <p className="panel-label">Start page</p>
                <h2>{selectedSession?.title || "Create or pick a session"}</h2>
              </div>
              <div className="workspace-actions">
                {selectedSession ? (
                  <button className="ghost" disabled={loading || endingSession} onClick={() => void handleDeleteSession()} type="button">
                    Delete session
                  </button>
                ) : null}
                {canEnterInterview ? (
                  <button className="primary" onClick={handleEnterInterview} type="button">Go to interview session</button>
                ) : null}
              </div>
            </div>

            {selectedSession ? (
              <>
                <div className="session-summary-card">
                  <div>
                    <p className="panel-label">Session status</p>
                    <h3>{selectedSession.status === "completed" ? "Session completed" : "Ready to practice"}</h3>
                    <p className="lede">
                      {selectedSession.status === "completed"
                        ? "Review the wrap-up below or create a new session for another round."
                        : "Camera and microphone stay off on this page. Open the interview room when you want to begin live capture."}
                    </p>
                  </div>
                  <div className="summary-pills">
                    <span>{selectedSession.transcript_turns} transcript turns</span>
                    <span>{selectedSession.total_samples} expression samples</span>
                    <span>{selectedSession.dominant_emotion} dominant mood</span>
                  </div>
                </div>

                <div className="grid detail-grid">
                  <MetricCard label="Calmness" value={`${selectedSession.calmness_percent}%`} />
                  <MetricCard label="Smiles" value={selectedSession.smile_events} />
                  <MetricCard label="Surprises" value={selectedSession.surprise_events} />
                  <MetricCard label="Duration" value={formatDuration(selectedSession.duration_seconds)} />
                </div>

                {dashboardWrapUp ? (
                  <div className="review-card wrap-up-card">
                    <p className="panel-label">Latest wrap-up</p>
                    <h3>{dashboardWrapUp.score}/100 overall</h3>
                    <p>{dashboardWrapUp.summary}</p>
                    <div className="wrap-up-grid">
                      <WrapUpList label="Strengths" items={dashboardWrapUp.strengths} />
                      <WrapUpList label="Brush up next" items={dashboardWrapUp.brushUp} />
                      <WrapUpList label="Answer coaching" items={dashboardWrapUp.answerFeedback} />
                      <WrapUpList label="Expression coaching" items={dashboardWrapUp.expressionFeedback} />
                    </div>
                  </div>
                ) : (
                  <div className="empty-state compact-empty-state">
                    <p>Complete one interview to get a full wrap-up with score, strengths, and coaching points.</p>
                  </div>
                )}
              </>
            ) : (
              <div className="empty-state">
                <p>Create an interview session to see your performance summary and enter the interview room.</p>
              </div>
            )}
          </section>
        </section>
      ) : (
        <section className="panel stack gap-lg wide-panel">
          <div className="workspace-head">
            <div>
              <p className="panel-label">Interview session</p>
              <h2>{selectedSession?.title || "Interview room"}</h2>
            </div>
            <div className="workspace-actions">
              {selectedSession ? (
                <button className="ghost" disabled={loading || endingSession} onClick={() => void handleDeleteSession()} type="button">Delete session</button>
              ) : null}
              <button className="ghost" onClick={() => setActiveView("dashboard")} type="button">Back to start page</button>
            </div>
          </div>

          {endingSession ? (
            <div className="ending-state">
              <div className="loading-orb" aria-hidden="true" />
              <h3>Wrapping up your interview</h3>
              <p>Scoring answers, summarizing expression signals, and preparing next-step coaching.</p>
            </div>
          ) : null}

          {selectedSession ? (
            <>
              <InterviewMediaPanel
                active={!endingSession}
                endingSession={endingSession}
                onEndInterview={handleCompleteSession}
                onError={(message) => setError(message)}
                onSessionUpdate={(detail) => {
                  syncSessionDetail(detail);
                }}
                onStatus={setStatusMessage}
                session={selectedSession}
                token={token}
              />

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
                    <button disabled={loading || endingSession || selectedSession.status === "completed"} key={emotion} onClick={() => void handleAddEmotion(emotion)} type="button">
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
                <button className="ghost" disabled={loading || endingSession || selectedSession.status === "completed"} type="submit">Store transcript turn</button>
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
            </>
          ) : (
            <div className="empty-state">
              <p>Select a session from the start page before entering the interview room.</p>
            </div>
          )}
        </section>
      )}

      <SiteFooter />
    </main>
  );
}


function WrapUpList({ label, items }) {
  if (!items?.length) {
    return null;
  }

  return (
    <div className="wrap-up-list">
      <p className="panel-label">{label}</p>
      <div className="wrap-up-items">
        {items.map((item) => (
          <p key={item}>{item}</p>
        ))}
      </div>
    </div>
  );
}


function SiteFooter() {
  return (
    <footer className="legal-panel">
      <div className="legal-block">
        <p className="panel-label">Privacy policy</p>
        <p>
          Interview Presence Coach stores interview sessions, transcript turns, expression samples, and review summaries to support your
          practice history. Keep sensitive personal or company information out of mock interview responses unless you are
          comfortable storing it in this application.
        </p>
      </div>
      <div className="legal-block legal-block-compact">
        <p className="panel-label">Copyright</p>
        <p>© 2026 Interview Presence Coach. All rights reserved.</p>
      </div>
    </footer>
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


function TrendBadge({ label, value, suffix, neutral = false }) {
  const hasValue = typeof value === "number";
  const signedValue = hasValue && !neutral && value > 0 ? `+${value}` : value;
  return (
    <article className={`trend-badge ${neutral ? "neutral" : hasValue && value >= 0 ? "up" : "down"}`}>
      <p>{label}</p>
      <strong>{hasValue ? `${signedValue}${suffix}` : "--"}</strong>
    </article>
  );
}


function ProgressChart({ sessions }) {
  if (!sessions.length) {
    return (
      <div className="progress-card empty-state compact-empty-state">
        <p>Complete a few interviews and the score and calmness trend will appear here.</p>
      </div>
    );
  }

  const width = 620;
  const height = 220;
  const padding = 28;
  const innerWidth = width - padding * 2;
  const innerHeight = height - padding * 2;
  const chartSessions = sessions.slice(-8);
  const step = chartSessions.length > 1 ? innerWidth / (chartSessions.length - 1) : 0;
  const buildPoint = (session, index, value) => {
    const x = padding + index * step;
    const y = padding + innerHeight - (Math.max(0, Math.min(100, value)) / 100) * innerHeight;
    return `${x},${y}`;
  };
  const scorePoints = chartSessions
    .filter((session) => typeof session.overall_score === "number")
    .map((session, index) => buildPoint(session, index, session.overall_score));
  const calmnessPoints = chartSessions.map((session, index) => buildPoint(session, index, session.calmness_percent));

  return (
    <article className="progress-card progress-chart-card">
      <div className="progress-card-head">
        <div>
          <p className="panel-label">Performance timeline</p>
          <h3>Score and calmness trend</h3>
        </div>
        <div className="chart-legend">
          <span className="legend-item legend-score">Overall score</span>
          <span className="legend-item legend-calmness">Calmness</span>
        </div>
      </div>
      <svg className="progress-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Overall score and calmness trend">
        {[0, 50, 100].map((tick) => {
          const y = padding + innerHeight - (tick / 100) * innerHeight;
          return <line className="chart-grid-line" key={tick} x1={padding} x2={width - padding} y1={y} y2={y} />;
        })}
        {scorePoints.length > 1 ? <polyline className="chart-line chart-line-score" fill="none" points={scorePoints.join(" ")} /> : null}
        {calmnessPoints.length > 1 ? <polyline className="chart-line chart-line-calmness" fill="none" points={calmnessPoints.join(" ")} /> : null}
        {chartSessions.map((session, index) => {
          const x = padding + index * step;
          const scoreY = typeof session.overall_score === "number"
            ? padding + innerHeight - (session.overall_score / 100) * innerHeight
            : null;
          const calmnessY = padding + innerHeight - (session.calmness_percent / 100) * innerHeight;
          return (
            <g key={session.id}>
              {scoreY !== null ? <circle className="chart-point chart-point-score" cx={x} cy={scoreY} r="4" /> : null}
              <circle className="chart-point chart-point-calmness" cx={x} cy={calmnessY} r="4" />
            </g>
          );
        })}
      </svg>
      <div className="chart-label-row">
        {chartSessions.map((session) => (
          <span key={session.id}>{formatTimelineDate(session.completed_at || session.started_at)}</span>
        ))}
      </div>
    </article>
  );
}


function EmotionBreakdown({ items }) {
  if (!items.length) {
    return (
      <div className="progress-card empty-state compact-empty-state">
        <p>As completed sessions build up, this panel will highlight the expression patterns that show up most often.</p>
      </div>
    );
  }

  return (
    <article className="progress-card emotion-breakdown-card">
      <div className="progress-card-head">
        <div>
          <p className="panel-label">Dominant expression mix</p>
          <h3>What your history leans toward</h3>
        </div>
      </div>
      <div className="emotion-breakdown-list">
        {items.map((item) => (
          <div className="emotion-breakdown-item" key={item.emotion}>
            <div className="emotion-breakdown-meta">
              <strong>{item.emotion}</strong>
              <span>{item.count} sessions</span>
            </div>
            <div className="emotion-breakdown-bar-track">
              <div className="emotion-breakdown-bar-fill" style={{ width: `${item.share}%` }} />
            </div>
            <p>{item.share}% of completed interviews</p>
          </div>
        ))}
      </div>
    </article>
  );
}


export default App;