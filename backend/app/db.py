from __future__ import annotations

import calendar
import json
import threading
import time
import uuid
from collections import Counter

from postgrest.exceptions import APIError
from supabase import Client, create_client

from .analytics import SessionAnalytics
from .config import DAILY_LLM_CALL_LIMIT, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL, TOKEN_TTL_SECONDS, validate_supabase_config
from .schemas import (
    DashboardSummary,
    EmotionSampleInput,
    SessionCreateRequest,
    SessionDetail,
    SessionMessageInput,
    SessionMessageResponse,
    SessionReviewInput,
    SessionReviewResponse,
    SessionSummary,
    UserSummary,
)
from .security import create_token, hash_password, verify_password


class AppDatabase:
    def __init__(self) -> None:
        validate_supabase_config()
        self._thread_local = threading.local()

    @property
    def client(self) -> Client:
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            self._thread_local.client = client
        return client

    def create_user(self, email: str, display_name: str, password: str) -> tuple[UserSummary, str]:
        normalized_email = email.strip().lower()
        existing = self._users_by_email(normalized_email, "id")
        if existing:
            raise ValueError("Email already registered")
        now = time.time()
        payload = {
            "email": normalized_email,
            "display_name": display_name.strip(),
            "password_hash": hash_password(password),
            "created_at": now,
        }
        created = self.client.table("users").insert(payload).execute().data or []
        if not created:
            raise RuntimeError("Failed to create user")
        user_id = int(created[0]["id"])
        token = self._issue_token(user_id)
        return self.get_user(user_id), token

    def authenticate_user(self, email: str, password: str) -> tuple[UserSummary, str] | None:
        rows = self._users_by_email(email.strip().lower(), "id, email, display_name, password_hash, created_at")
        if not rows or not verify_password(password, rows[0]["password_hash"]):
            return None
        token = self._issue_token(int(rows[0]["id"]))
        return self._user_from_row(rows[0]), token

    def authenticate_oauth_user(self, email: str, display_name: str) -> tuple[UserSummary, str]:
        normalized_email = email.strip().lower()
        normalized_display_name = display_name.strip() or normalized_email.split("@", 1)[0]
        rows = self._users_by_email(normalized_email, "id, email, display_name, created_at")

        if rows:
            user_row = rows[0]
            if normalized_display_name and normalized_display_name != user_row["display_name"]:
                self.client.table("users").update({"display_name": normalized_display_name}).eq("id", int(user_row["id"])).execute()
                user_row = {**user_row, "display_name": normalized_display_name}
            token = self._issue_token(int(user_row["id"]))
            return self._user_from_row(user_row), token

        now = time.time()
        created = self.client.table("users").insert(
            {
                "email": normalized_email,
                "display_name": normalized_display_name,
                "password_hash": hash_password(create_token()),
                "created_at": now,
            }
        ).execute().data or []
        if not created:
            raise RuntimeError("Failed to create user")
        user_id = int(created[0]["id"])
        token = self._issue_token(user_id)
        return self.get_user(user_id), token

    def get_user(self, user_id: int) -> UserSummary:
        rows = self.client.table("users").select("id, email, display_name, created_at").eq("id", user_id).limit(1).execute().data or []
        if not rows:
            raise KeyError("User not found")
        return self._user_from_row(rows[0])

    def get_user_by_token(self, token: str) -> UserSummary | None:
        now = time.time()
        rows = self.client.table("auth_tokens").select("user_id, expires_at").eq("token", token).gte("expires_at", now).limit(1).execute().data or []
        if not rows:
            return None
        return self.get_user(int(rows[0]["user_id"]))

    def create_session(self, user_id: int, payload: SessionCreateRequest) -> SessionSummary:
        session_id = str(uuid.uuid4())
        now = time.time()
        self.client.table("interview_sessions").insert(
            {
                "id": session_id,
                "user_id": user_id,
                "mode": payload.mode,
                "title": payload.title.strip(),
                "status": "active",
                "started_at": now,
                "completed_at": None,
                "review_json": None,
            }
        ).execute()
        return self.get_session(user_id, session_id)

    def add_samples(self, user_id: int, session_id: str, samples: list[EmotionSampleInput]) -> SessionDetail:
        session = self._require_session_row(user_id, session_id)
        self._ensure_active(session)
        now = time.time()
        self.client.table("session_samples").insert(
            [
                {
                    "session_id": session_id,
                    "recorded_at": sample.recorded_at or now,
                    "emotion": sample.emotion,
                    "confidence": sample.confidence,
                    "metrics_json": sample.metrics,
                    "scores_json": sample.scores,
                }
                for sample in samples
            ]
        ).execute()
        return self.get_session(user_id, session_id)

    def add_messages(self, user_id: int, session_id: str, messages: list[SessionMessageInput]) -> SessionDetail:
        session = self._require_session_row(user_id, session_id)
        self._ensure_active(session)
        self.store_messages(user_id, session_id, messages)
        return self.get_session(user_id, session_id)

    def store_messages(self, user_id: int, session_id: str, messages: list[SessionMessageInput | SessionMessageResponse]) -> None:
        session = self._require_session_row(user_id, session_id)
        self._ensure_active(session)
        now = time.time()
        self.client.table("session_messages").insert(
            [
                {
                    "session_id": session_id,
                    "role": message.role,
                    "content": message.content.strip(),
                    "created_at": message.created_at or now,
                }
                for message in messages
            ]
        ).execute()

    def complete_session(self, user_id: int, session_id: str, review: SessionReviewInput | None, completed_at: float | None) -> SessionDetail:
        session = self._require_session_row(user_id, session_id)
        review_json = review.model_dump() if review is not None else session["review_json"]
        end_at = completed_at or time.time()
        self.client.table("interview_sessions").update(
            {
                "status": "completed",
                "completed_at": end_at,
                "review_json": review_json,
            }
        ).eq("id", session_id).eq("user_id", user_id).execute()
        return self.get_session(user_id, session_id)

    def list_sessions(self, user_id: int) -> list[SessionSummary]:
        rows = self.client.table("interview_sessions").select("*").eq("user_id", user_id).order("started_at", desc=True).execute().data or []
        return [self._session_summary_from_row(user_id, row) for row in rows]

    def get_session(self, user_id: int, session_id: str) -> SessionDetail:
        row = self._require_session_row(user_id, session_id)
        messages = self._get_messages(session_id)
        review = self._get_review(row["review_json"])
        summary = self._session_summary_from_row(user_id, row, messages=messages, review=review)
        return SessionDetail(**summary.model_dump(), messages=messages, review=review)

    def delete_session(self, user_id: int, session_id: str) -> None:
        self._require_session_row(user_id, session_id)
        self.client.table("interview_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()

    def dashboard_summary(self, user_id: int) -> DashboardSummary:
        user = self.get_user(user_id)
        sessions = self.list_sessions(user_id)
        completed_sessions = [session for session in sessions if session.status == "completed"]
        active_sessions = [session for session in sessions if session.status == "active"]
        day_start = self._start_of_utc_day()
        llm_calls_used_today = self.count_llm_calls_since(user_id, day_start)
        llm_calls_remaining_today = max(DAILY_LLM_CALL_LIMIT - llm_calls_used_today, 0) if DAILY_LLM_CALL_LIMIT > 0 else 0
        average_calmness = int(sum(session.calmness_percent for session in sessions) / len(sessions)) if sessions else 0
        scores = [session.overall_score for session in completed_sessions if session.overall_score is not None]
        dominant_mode = Counter(session.mode for session in sessions).most_common(1)[0][0] if sessions else None
        dominant_emotion = Counter(session.dominant_emotion for session in sessions if session.total_samples).most_common(1)[0][0] if any(session.total_samples for session in sessions) else None
        return DashboardSummary(
            user=user,
            total_sessions=len(sessions),
            completed_sessions=len(completed_sessions),
            active_sessions=len(active_sessions),
            daily_llm_call_limit=DAILY_LLM_CALL_LIMIT,
            llm_calls_used_today=llm_calls_used_today,
            llm_calls_remaining_today=llm_calls_remaining_today,
            average_calmness=average_calmness,
            average_score=int(sum(scores) / len(scores)) if scores else None,
            total_smiles=sum(session.smile_events for session in sessions),
            total_surprises=sum(session.surprise_events for session in sessions),
            total_reaction_spikes=sum(session.reaction_spikes for session in sessions),
            dominant_mode=dominant_mode,
            dominant_emotion=dominant_emotion,
            latest_sessions=sessions[:5],
        )

    def count_llm_calls_since(self, user_id: int, started_at: float) -> int:
        response = self.client.table("llm_usage_events").select("id", count="exact").eq("user_id", user_id).gte("created_at", started_at).execute()
        return response.count or 0

    def record_llm_calls(self, user_id: int, call_types: list[str], recorded_at: float | None = None) -> None:
        if not call_types:
            return
        event_time = recorded_at or time.time()
        self.client.table("llm_usage_events").insert(
            [
                {
                    "user_id": user_id,
                    "call_type": call_type,
                    "created_at": event_time,
                }
                for call_type in call_types
            ]
        ).execute()

    def _issue_token(self, user_id: int) -> str:
        token = create_token()
        now = time.time()
        self.client.table("auth_tokens").insert(
            {
                "token": token,
                "user_id": user_id,
                "created_at": now,
                "expires_at": now + TOKEN_TTL_SECONDS,
            }
        ).execute()
        return token

    def _users_by_email(self, email: str, columns: str) -> list[dict[str, object]]:
        return self.client.table("users").select(columns).eq("email", email).limit(1).execute().data or []

    @staticmethod
    def _start_of_utc_day(timestamp: float | None = None) -> float:
        current = timestamp or time.time()
        utc = time.gmtime(current)
        return float(calendar.timegm((utc.tm_year, utc.tm_mon, utc.tm_mday, 0, 0, 0, 0, 0, 0)))

    def _require_session_row(self, user_id: int, session_id: str) -> dict[str, object]:
        rows = self.client.table("interview_sessions").select("*").eq("id", session_id).eq("user_id", user_id).limit(1).execute().data or []
        if not rows:
            raise KeyError("Session not found")
        return rows[0]

    @staticmethod
    def _ensure_active(session: dict[str, object]) -> None:
        if session["status"] != "active":
            raise ValueError("Session is already completed")

    def _get_samples(self, session_id: str) -> list[EmotionSampleInput]:
        rows = self.client.table("session_samples").select("emotion, confidence, recorded_at, metrics_json, scores_json").eq(
            "session_id", session_id
        ).order("recorded_at").execute().data or []
        return [
            EmotionSampleInput(
                emotion=row["emotion"],
                confidence=float(row["confidence"]),
                recorded_at=float(row["recorded_at"]),
                metrics=row["metrics_json"] or {},
                scores=row["scores_json"] or {},
            )
            for row in rows
        ]

    def _get_messages(self, session_id: str) -> list[SessionMessageResponse]:
        rows = self.client.table("session_messages").select("role, content, created_at").eq("session_id", session_id).order(
            "created_at"
        ).execute().data or []
        return [
            SessionMessageResponse(
                role=row["role"],
                content=row["content"],
                created_at=float(row["created_at"]),
            )
            for row in rows
        ]

    @staticmethod
    def _get_review(review_json: dict[str, object] | str | None) -> SessionReviewResponse | None:
        if not review_json:
            return None
        if isinstance(review_json, str):
            return SessionReviewResponse(**json.loads(review_json))
        return SessionReviewResponse(**review_json)

    def _session_summary_from_row(
        self,
        user_id: int,
        row: dict[str, object],
        *,
        messages: list[SessionMessageResponse] | None = None,
        review: SessionReviewResponse | None = None,
    ) -> SessionSummary:
        samples = self._get_samples(row["id"])
        resolved_messages = messages if messages is not None else self._get_messages(row["id"])
        resolved_review = review if review is not None else self._get_review(row["review_json"])
        aggregate = SessionAnalytics.summarize(
            started_at=float(row["started_at"]),
            completed_at=float(row["completed_at"]) if row["completed_at"] is not None else None,
            samples=samples,
            messages=resolved_messages,
            review=resolved_review,
        )
        return SessionSummary(
            id=row["id"],
            user_id=user_id,
            mode=row["mode"],
            title=row["title"],
            status=row["status"],
            started_at=float(row["started_at"]),
            completed_at=float(row["completed_at"]) if row["completed_at"] is not None else None,
            duration_seconds=aggregate.duration_seconds,
            calmness_percent=aggregate.calmness_percent,
            smile_events=aggregate.smile_events,
            surprise_events=aggregate.surprise_events,
            reaction_spikes=aggregate.reaction_spikes,
            smiles_per_minute=aggregate.smiles_per_minute,
            dominant_emotion=aggregate.dominant_emotion,
            mood_mix=aggregate.mood_mix,
            total_samples=aggregate.total_samples,
            transcript_turns=aggregate.transcript_turns,
            latest_expression=aggregate.latest_expression,
            overall_score=aggregate.overall_score,
        )

    @staticmethod
    def _user_from_row(row: dict[str, object]) -> UserSummary:
        return UserSummary(
            id=int(row["id"]),
            email=row["email"],
            display_name=row["display_name"],
            created_at=float(row["created_at"]),
        )


try:
    database = AppDatabase()
except APIError as exc:
    raise RuntimeError("Supabase is reachable but the required tables do not exist yet. Run backend/supabase/schema.sql in Supabase SQL Editor.") from exc