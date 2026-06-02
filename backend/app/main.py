from __future__ import annotations

import calendar
import logging
from pathlib import Path
import time

import cv2
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi import File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import numpy as np
import requests

from .config import DAILY_LLM_CALL_LIMIT, FRONTEND_ORIGIN_REGEX, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL, frontend_origins
from .db import database
from .interview import GroqInterviewService, InterviewMessage
from .session_cache import active_interview_store
from .schemas import (
    AuthRequest,
    AuthResponse,
    DashboardSummary,
    EmotionSampleInput,
    InterviewReviewResponse,
    InterviewTurnResponse,
    OAuthLoginRequest,
    RegisterRequest,
    SessionCompleteRequest,
    SessionCreateRequest,
    SessionDetail,
    SessionMessageInput,
    SessionMessageResponse,
    SessionReviewInput,
    SessionMessagesRequest,
    SessionSamplesRequest,
    SessionSummary,
    UserSummary,
    VisionAnalysisResponse,
    VisionAnalysisResult,
)
from .vision import FaceAnalyzer

LOGGER = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parents[2]


app = FastAPI(title="Mood Mirror API", version="0.1.0")
interview_service = GroqInterviewService(project_root)
face_analyzer = FaceAnalyzer(project_root / "models")

allowed_origins = [*frontend_origins(), "http://127.0.0.1:5173", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(allowed_origins)),
    allow_origin_regex=FRONTEND_ORIGIN_REGEX or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "name": app.title,
            "status": "ok",
            "docs": "/docs",
            "health": "/health",
        }
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def get_current_user(authorization: str | None = Header(default=None)) -> UserSummary:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    user = database.get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return user


def _ensure_interview_session(detail: SessionDetail) -> None:
    if detail.mode != "interview":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This action is only available for interview sessions")
    if detail.status != "active":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This session is already completed")


def _expression_summary(detail: SessionDetail) -> str:
    if not detail.total_samples:
        return "No recent expression data yet."
    return (
        f"Recent dominant expression: {detail.dominant_emotion} for the current session, "
        f"average calmness {detail.calmness_percent}%, smiles {detail.smile_events}, "
        f"surprise moments {detail.surprise_events}."
    )


def _conversation(detail: SessionDetail) -> list[InterviewMessage]:
    return [InterviewMessage(role=message.role, content=message.content) for message in detail.messages if message.role in {"user", "assistant"}]


def _require_interview_service() -> None:
    if interview_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Interview features are unavailable in this deployment.",
        )


def _require_face_analyzer() -> None:
    if face_analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision analysis is unavailable in this deployment.",
        )


def _detail_with_messages(detail: SessionDetail, messages: list[SessionMessageResponse]) -> SessionDetail:
    return detail.model_copy(update={"messages": messages, "transcript_turns": len(messages)})


def _detail_with_cached_messages(detail: SessionDetail) -> SessionDetail:
    if detail.mode != "interview":
        return detail
    messages = active_interview_store.load_messages(detail.id, detail.messages)
    return _detail_with_messages(detail, messages)


def _pending_messages(
    persisted_messages: list[SessionMessageResponse],
    cached_messages: list[SessionMessageResponse],
) -> list[SessionMessageResponse]:
    persisted_payload = [message.model_dump() for message in persisted_messages]
    cached_payload = [message.model_dump() for message in cached_messages]
    if cached_payload[: len(persisted_payload)] == persisted_payload:
        return cached_messages[len(persisted_messages) :]
    return cached_messages


def _flush_interview_messages(user_id: int, detail: SessionDetail) -> None:
    if detail.mode != "interview":
        return
    cached_messages = active_interview_store.flush_messages(detail.id, detail.messages)
    pending_messages = _pending_messages(detail.messages, cached_messages)
    if pending_messages:
        database.store_messages(user_id, detail.id, pending_messages)


def _vision_response(detail: SessionDetail, result) -> VisionAnalysisResponse:
    face_box = list(result.face_box) if result.face_box is not None else None
    return VisionAnalysisResponse(
        session=detail,
        analysis=VisionAnalysisResult(
            emotion=result.emotion,
            confidence=result.confidence,
            status=result.status,
            scores=result.scores,
            metrics=result.metrics,
            face_box=face_box,
        ),
    )


def _analyze_uploaded_frame(frame_bytes: bytes):
    if not frame_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Frame upload was empty")

    np_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not decode the uploaded frame")

    return face_analyzer.analyze(image)


def _start_of_utc_day(timestamp: float | None = None) -> float:
    current = timestamp or time.time()
    utc = time.gmtime(current)
    return float(calendar.timegm((utc.tm_year, utc.tm_mon, utc.tm_mday, 0, 0, 0, 0, 0, 0)))


def _consume_llm_quota(user_id: int, call_types: list[str]) -> None:
    if DAILY_LLM_CALL_LIMIT <= 0 or not call_types:
        return

    if interview_service is None:
        return

    interview_service.reload()
    if not interview_service.api_key:
        return

    day_start = _start_of_utc_day()
    used_calls = database.count_llm_calls_since(user_id, day_start)
    if used_calls + len(call_types) > DAILY_LLM_CALL_LIMIT:
        remaining = max(DAILY_LLM_CALL_LIMIT - used_calls, 0)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Daily LLM quota exceeded. Limit: {DAILY_LLM_CALL_LIMIT} calls per UTC day. "
                f"Remaining today: {remaining}."
            ),
        )

    database.record_llm_calls(user_id, call_types)


def _supabase_oauth_user(access_token: str) -> dict[str, object]:
    response = requests.get(
        f"{SUPABASE_URL.rstrip('/')}/auth/v1/user",
        headers={
            "Authorization": f"Bearer {access_token}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
        },
        timeout=20,
    )
    if response.status_code == status.HTTP_401_UNAUTHORIZED:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="OAuth session is invalid or expired")
    response.raise_for_status()
    return response.json()


def _oauth_display_name(oauth_user: dict[str, object]) -> str:
    metadata = oauth_user.get("user_metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    for key in ("full_name", "name", "user_name", "preferred_username"):
        candidate = metadata.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()[:120]

    email = oauth_user.get("email")
    if isinstance(email, str) and email.strip():
        return email.split("@", 1)[0][:120]

    return "Mood Mirror User"


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest) -> AuthResponse:
    try:
        user, token = database.create_user(payload.email, payload.display_name, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return AuthResponse(token=token, user=user)


@app.post("/api/auth/login", response_model=AuthResponse)
def login(payload: AuthRequest) -> AuthResponse:
    result = database.authenticate_user(payload.email, payload.password)
    if result is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    user, token = result
    return AuthResponse(token=token, user=user)


@app.post("/api/auth/oauth", response_model=AuthResponse)
def oauth_login(payload: OAuthLoginRequest) -> AuthResponse:
    try:
        oauth_user = _supabase_oauth_user(payload.access_token)
    except requests.RequestException as exc:
        LOGGER.exception("OAuth user lookup failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to verify OAuth session") from exc

    email = oauth_user.get("email")
    if not isinstance(email, str) or not email.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="OAuth account is missing an email address")

    user, token = database.authenticate_oauth_user(email, _oauth_display_name(oauth_user))
    return AuthResponse(token=token, user=user)


@app.get("/api/me", response_model=UserSummary)
def me(user: UserSummary = Depends(get_current_user)) -> UserSummary:
    return user


@app.get("/api/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary(user: UserSummary = Depends(get_current_user)) -> DashboardSummary:
    return database.dashboard_summary(user.id)


@app.get("/api/sessions", response_model=list[SessionSummary])
def list_sessions(user: UserSummary = Depends(get_current_user)) -> list[SessionSummary]:
    return database.list_sessions(user.id)


@app.post("/api/sessions", response_model=SessionSummary, status_code=status.HTTP_201_CREATED)
def create_session(payload: SessionCreateRequest, user: UserSummary = Depends(get_current_user)) -> SessionSummary:
    return database.create_session(user.id, payload)


@app.get("/api/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, user: UserSummary = Depends(get_current_user)) -> SessionDetail:
    try:
        detail = database.get_session(user.id, session_id)
        return _detail_with_cached_messages(detail)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc


@app.delete("/api/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str, user: UserSummary = Depends(get_current_user)) -> Response:
    try:
        database.delete_session(user.id, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

    active_interview_store.clear(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/sessions/{session_id}/samples", response_model=SessionDetail)
def add_samples(
    session_id: str,
    payload: SessionSamplesRequest,
    user: UserSummary = Depends(get_current_user),
) -> SessionDetail:
    try:
        return database.add_samples(user.id, session_id, payload.samples)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/messages", response_model=SessionDetail)
def add_messages(
    session_id: str,
    payload: SessionMessagesRequest,
    user: UserSummary = Depends(get_current_user),
) -> SessionDetail:
    try:
        return database.add_messages(user.id, session_id, payload.messages)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/complete", response_model=SessionDetail)
def complete_session(
    session_id: str,
    payload: SessionCompleteRequest,
    user: UserSummary = Depends(get_current_user),
) -> SessionDetail:
    try:
        detail = database.get_session(user.id, session_id)
        _flush_interview_messages(user.id, detail)
        return database.complete_session(user.id, session_id, payload.review, payload.completed_at)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc


@app.post("/api/sessions/{session_id}/interview/start", response_model=InterviewTurnResponse)
def start_interview(session_id: str, user: UserSummary = Depends(get_current_user)) -> InterviewTurnResponse:
    _require_interview_service()
    try:
        detail = database.get_session(user.id, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

    _ensure_interview_session(detail)
    detail = _detail_with_cached_messages(detail)
    _consume_llm_quota(user.id, ["interview_start"])

    try:
        assistant_message = interview_service.generate_turn(_conversation(detail), _expression_summary(detail), "start")
    except requests.RequestException as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to generate the opening interview question") from exc

    updated_messages = active_interview_store.append_messages(
        session_id,
        detail.messages,
        [SessionMessageResponse(role="assistant", content=assistant_message, created_at=time.time())],
    )
    updated = _detail_with_messages(detail, updated_messages)
    return InterviewTurnResponse(session=updated, assistant_message=assistant_message)


@app.post("/api/sessions/{session_id}/interview/respond", response_model=InterviewTurnResponse)
async def respond_to_interview(
    session_id: str,
    audio: UploadFile = File(...),
    user: UserSummary = Depends(get_current_user),
) -> InterviewTurnResponse:
    _require_interview_service()
    try:
        detail = database.get_session(user.id, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

    _ensure_interview_session(detail)
    detail = _detail_with_cached_messages(detail)
    _consume_llm_quota(user.id, ["transcription", "interview_reply"])

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio upload was empty")

    try:
        transcript = interview_service.transcribe_audio_file(
            audio_bytes,
            audio.filename or "interview-response.webm",
            audio.content_type or "audio/webm",
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to transcribe the recorded answer") from exc

    if not transcript.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No clear speech was detected in the recording")

    user_message = SessionMessageResponse(role="user", content=transcript, created_at=time.time())
    detail = _detail_with_messages(
        detail,
        active_interview_store.append_messages(session_id, detail.messages, [user_message]),
    )

    try:
        assistant_message = interview_service.generate_turn(_conversation(detail), _expression_summary(detail), "reply")
    except requests.RequestException as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to generate the follow-up interview question") from exc

    assistant_record = SessionMessageResponse(role="assistant", content=assistant_message, created_at=time.time())
    updated = _detail_with_messages(
        detail,
        active_interview_store.append_messages(session_id, detail.messages, [assistant_record]),
    )
    return InterviewTurnResponse(session=updated, assistant_message=assistant_message, transcript=transcript)


@app.post("/api/sessions/{session_id}/vision/analyze", response_model=VisionAnalysisResponse)
async def analyze_session_frame(
    session_id: str,
    frame: UploadFile = File(...),
    user: UserSummary = Depends(get_current_user),
) -> VisionAnalysisResponse:
    _require_face_analyzer()
    try:
        detail = database.get_session(user.id, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

    if detail.status != "active":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This session is already completed")

    frame_bytes = await frame.read()
    result = _analyze_uploaded_frame(frame_bytes)
    updated = _detail_with_cached_messages(detail)
    if result.face_box is not None:
        updated = _detail_with_cached_messages(
            database.add_samples(
                user.id,
                session_id,
                [
                    EmotionSampleInput(
                        emotion=result.emotion,
                        confidence=result.confidence,
                        metrics=result.metrics,
                        scores=result.scores,
                    )
                ],
            )
        )

    return _vision_response(updated, result)


@app.post("/api/sessions/{session_id}/vision/analyze-batch", response_model=VisionAnalysisResponse)
async def analyze_session_frames_batch(
    session_id: str,
    frames: list[UploadFile] = File(...),
    recorded_at: list[float] = Form(default=[]),
    user: UserSummary = Depends(get_current_user),
) -> VisionAnalysisResponse:
    _require_face_analyzer()
    try:
        detail = database.get_session(user.id, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

    if detail.status != "active":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This session is already completed")

    if not frames:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No frames were uploaded")

    samples: list[EmotionSampleInput] = []
    latest_result = None
    for index, frame in enumerate(frames):
        frame_bytes = await frame.read()
        result = _analyze_uploaded_frame(frame_bytes)
        latest_result = result
        if result.face_box is None:
            continue
        sample_recorded_at = recorded_at[index] if index < len(recorded_at) else None
        samples.append(
            EmotionSampleInput(
                emotion=result.emotion,
                confidence=result.confidence,
                recorded_at=sample_recorded_at,
                metrics=result.metrics,
                scores=result.scores,
            )
        )

    if latest_result is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No frames were uploaded")

    updated = _detail_with_cached_messages(detail)
    if samples:
        updated = _detail_with_cached_messages(database.add_samples(user.id, session_id, samples))

    return _vision_response(updated, latest_result)


@app.post("/api/sessions/{session_id}/interview/review", response_model=InterviewReviewResponse)
def review_interview(session_id: str, user: UserSummary = Depends(get_current_user)) -> InterviewReviewResponse:
    _require_interview_service()
    try:
        detail = database.get_session(user.id, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

    detail = _detail_with_cached_messages(detail)
    _consume_llm_quota(user.id, ["interview_review"])
    conversation = _conversation(detail)
    review = interview_service.review_session(
        conversation,
        _expression_summary(detail),
        {
            "calmness_percent": detail.calmness_percent,
            "smiles_per_minute": detail.smiles_per_minute,
            "surprise_events": detail.surprise_events,
            "dominant_emotion": detail.dominant_emotion,
            "total_samples": detail.total_samples,
        },
    )
    _flush_interview_messages(user.id, detail)
    updated = database.complete_session(
        user.id,
        session_id,
        SessionReviewInput(
            overall_score=review.overall_score,
            answer_score=review.answer_score,
            expression_score=review.expression_score,
            summary=review.summary,
            strengths=review.strengths,
            brush_up_topics=review.brush_up_topics,
            answer_feedback=review.answer_feedback,
            expression_feedback=review.expression_feedback,
        ),
        None,
    )
    return InterviewReviewResponse(session=updated, review=updated.review)