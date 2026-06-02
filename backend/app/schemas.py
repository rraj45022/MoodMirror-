from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SessionMode = Literal["mirror", "interview", "streamer"]
SessionStatus = Literal["active", "completed"]
MessageRole = Literal["system", "assistant", "user"]


class UserSummary(BaseModel):
    id: int
    email: str
    display_name: str
    created_at: float


class AuthRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=255)


class RegisterRequest(AuthRequest):
    display_name: str = Field(min_length=2, max_length=120)


class AuthResponse(BaseModel):
    token: str
    user: UserSummary


class OAuthLoginRequest(BaseModel):
    access_token: str = Field(min_length=20, max_length=8000)
    provider: str | None = Field(default=None, max_length=40)


class EmotionSampleInput(BaseModel):
    emotion: str = Field(min_length=2, max_length=32)
    confidence: float = Field(ge=0.0, le=1.0)
    recorded_at: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    scores: dict[str, float] = Field(default_factory=dict)


class SessionMessageInput(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1, max_length=4000)
    created_at: float | None = None


class SessionReviewInput(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    answer_score: int = Field(ge=0, le=100)
    expression_score: int = Field(ge=0, le=100)
    summary: str = Field(min_length=1, max_length=4000)
    strengths: list[str] = Field(default_factory=list)
    brush_up_topics: list[str] = Field(default_factory=list)
    answer_feedback: list[str] = Field(default_factory=list)
    expression_feedback: list[str] = Field(default_factory=list)


class SessionCreateRequest(BaseModel):
    mode: SessionMode = "interview"
    title: str = Field(min_length=2, max_length=160)


class SessionSamplesRequest(BaseModel):
    samples: list[EmotionSampleInput] = Field(min_length=1, max_length=500)


class SessionMessagesRequest(BaseModel):
    messages: list[SessionMessageInput] = Field(min_length=1, max_length=100)


class SessionCompleteRequest(BaseModel):
    completed_at: float | None = None
    review: SessionReviewInput | None = None


class SessionReviewResponse(SessionReviewInput):
    pass


class SessionMessageResponse(BaseModel):
    role: MessageRole
    content: str
    created_at: float


class SessionSummary(BaseModel):
    id: str
    user_id: int
    mode: SessionMode
    title: str
    status: SessionStatus
    started_at: float
    completed_at: float | None = None
    duration_seconds: int = 0
    calmness_percent: int = 0
    smile_events: int = 0
    surprise_events: int = 0
    reaction_spikes: int = 0
    smiles_per_minute: float = 0.0
    dominant_emotion: str = "neutral"
    mood_mix: dict[str, int] = Field(default_factory=dict)
    total_samples: int = 0
    transcript_turns: int = 0
    latest_expression: str = "No samples yet"
    overall_score: int | None = None


class SessionDetail(SessionSummary):
    messages: list[SessionMessageResponse] = Field(default_factory=list)
    review: SessionReviewResponse | None = None


class VisionAnalysisResult(BaseModel):
    emotion: str
    confidence: float
    status: str
    scores: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    face_box: list[int] | None = None


class VisionAnalysisResponse(BaseModel):
    session: SessionDetail
    analysis: VisionAnalysisResult


class InterviewTurnResponse(BaseModel):
    session: SessionDetail
    assistant_message: str
    transcript: str | None = None


class InterviewReviewResponse(BaseModel):
    session: SessionDetail
    review: SessionReviewResponse


class DashboardSummary(BaseModel):
    user: UserSummary
    total_sessions: int
    completed_sessions: int
    active_sessions: int
    daily_llm_call_limit: int
    llm_calls_used_today: int
    llm_calls_remaining_today: int
    average_calmness: int
    average_score: int | None = None
    total_smiles: int
    total_surprises: int
    total_reaction_spikes: int
    dominant_mode: str | None = None
    dominant_emotion: str | None = None
    latest_sessions: list[SessionSummary] = Field(default_factory=list)