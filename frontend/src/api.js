const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function request(path, { token, method = "GET", body } = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(payload.detail || "Request failed");
  }

  return response.json();
}

export function register(body) {
  return request("/api/auth/register", { method: "POST", body });
}

export function login(body) {
  return request("/api/auth/login", { method: "POST", body });
}

export function fetchDashboard(token) {
  return request("/api/dashboard/summary", { token });
}

export function fetchSessions(token) {
  return request("/api/sessions", { token });
}

export function createSession(token, body) {
  return request("/api/sessions", { token, method: "POST", body });
}

export function fetchSession(token, sessionId) {
  return request(`/api/sessions/${sessionId}`, { token });
}

export function addSessionMessages(token, sessionId, messages) {
  return request(`/api/sessions/${sessionId}/messages`, {
    token,
    method: "POST",
    body: { messages },
  });
}

export function addSessionSamples(token, sessionId, samples) {
  return request(`/api/sessions/${sessionId}/samples`, {
    token,
    method: "POST",
    body: { samples },
  });
}

export function completeSession(token, sessionId, body) {
  return request(`/api/sessions/${sessionId}/complete`, {
    token,
    method: "POST",
    body,
  });
}

export function startInterview(token, sessionId) {
  return request(`/api/sessions/${sessionId}/interview/start`, {
    token,
    method: "POST",
  });
}

export async function respondToInterview(token, sessionId, audioBlob) {
  const formData = new FormData();
  formData.append("audio", audioBlob, "interview-response.webm");
  const response = await fetch(`${API_URL}/api/sessions/${sessionId}/interview/respond`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(payload.detail || "Request failed");
  }
  return response.json();
}

export async function analyzeSessionFrame(token, sessionId, imageBlob) {
  const formData = new FormData();
  formData.append("frame", imageBlob, "frame.jpg");
  const response = await fetch(`${API_URL}/api/sessions/${sessionId}/vision/analyze`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(payload.detail || "Request failed");
  }
  return response.json();
}

export function requestInterviewReview(token, sessionId) {
  return request(`/api/sessions/${sessionId}/interview/review`, {
    token,
    method: "POST",
  });
}