from __future__ import annotations

from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import numpy as np
import requests

from .config import FRONTEND_ORIGIN
from .db import database
from .schemas import (
    AuthRequest,
    AuthResponse,
    DashboardSummary,
    EmotionSampleInput,
    InterviewReviewResponse,
    InterviewTurnResponse,
    RegisterRequest,
    SessionCompleteRequest,
    SessionCreateRequest,
    SessionDetail,
    SessionMessageInput,
    SessionReviewInput,
    SessionMessagesRequest,
    SessionSamplesRequest,
    SessionSummary,
    UserSummary,
    VisionAnalysisResponse,
    VisionAnalysisResult,
)
from app.interview import GroqInterviewService, InterviewMessage
from app.vision import FaceAnalyzer


app = FastAPI(title="Mood Mirror API", version="0.1.0")
project_root = Path(__file__).resolve().parents[2]
interview_service = GroqInterviewService(project_root)
face_analyzer = FaceAnalyzer(project_root / "models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://127.0.0.1:5173", "http://localhost:5173"],
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
        return database.get_session(user.id, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc


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
        return database.complete_session(user.id, session_id, payload.review, payload.completed_at)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc


@app.post("/api/sessions/{session_id}/interview/start", response_model=InterviewTurnResponse)
def start_interview(session_id: str, user: UserSummary = Depends(get_current_user)) -> InterviewTurnResponse:
    try:
        detail = database.get_session(user.id, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

    _ensure_interview_session(detail)

    try:
        assistant_message = interview_service.generate_turn(_conversation(detail), _expression_summary(detail), "start")
    except requests.RequestException as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to generate the opening interview question") from exc

    updated = database.add_messages(user.id, session_id, [SessionMessageInput(role="assistant", content=assistant_message)])
    return InterviewTurnResponse(session=updated, assistant_message=assistant_message)


@app.post("/api/sessions/{session_id}/interview/respond", response_model=InterviewTurnResponse)
async def respond_to_interview(
    session_id: str,
    audio: UploadFile = File(...),
    user: UserSummary = Depends(get_current_user),
) -> InterviewTurnResponse:
    try:
        detail = database.get_session(user.id, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

    _ensure_interview_session(detail)

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

    detail = database.add_messages(user.id, session_id, [SessionMessageInput(role="user", content=transcript)])

    try:
        assistant_message = interview_service.generate_turn(_conversation(detail), _expression_summary(detail), "reply")
    except requests.RequestException as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to generate the follow-up interview question") from exc

    updated = database.add_messages(user.id, session_id, [SessionMessageInput(role="assistant", content=assistant_message)])
    return InterviewTurnResponse(session=updated, assistant_message=assistant_message, transcript=transcript)


@app.post("/api/sessions/{session_id}/vision/analyze", response_model=VisionAnalysisResponse)
async def analyze_session_frame(
    session_id: str,
    frame: UploadFile = File(...),
    user: UserSummary = Depends(get_current_user),
) -> VisionAnalysisResponse:
    try:
        detail = database.get_session(user.id, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

    if detail.status != "active":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This session is already completed")

    frame_bytes = await frame.read()
    if not frame_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Frame upload was empty")

    np_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not decode the uploaded frame")

    result = face_analyzer.analyze(image)
    updated = detail
    if result.face_box is not None:
        updated = database.add_samples(
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

    return _vision_response(updated, result)


@app.post("/api/sessions/{session_id}/interview/review", response_model=InterviewReviewResponse)
def review_interview(session_id: str, user: UserSummary = Depends(get_current_user)) -> InterviewReviewResponse:
    try:
        detail = database.get_session(user.id, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc

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