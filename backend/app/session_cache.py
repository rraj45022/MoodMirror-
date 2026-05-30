from __future__ import annotations

import json
import logging
import threading

from redis import Redis
from redis.exceptions import RedisError

from .config import INTERVIEW_MESSAGE_TTL_SECONDS, REDIS_URL
from .schemas import SessionMessageResponse


LOGGER = logging.getLogger(__name__)


class _InMemorySessionCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._messages: dict[str, list[SessionMessageResponse]] = {}

    def get_messages(self, session_id: str) -> list[SessionMessageResponse] | None:
        with self._lock:
            messages = self._messages.get(session_id)
            if messages is None:
                return None
            return [message.model_copy(deep=True) for message in messages]

    def set_messages(self, session_id: str, messages: list[SessionMessageResponse]) -> None:
        with self._lock:
            self._messages[session_id] = [message.model_copy(deep=True) for message in messages]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._messages.pop(session_id, None)


class _RedisSessionCache:
    def __init__(self, redis_url: str) -> None:
        self._client = Redis.from_url(redis_url, decode_responses=True)

    @staticmethod
    def _key(session_id: str) -> str:
        return f"mood-mirror:interview-messages:{session_id}"

    def get_messages(self, session_id: str) -> list[SessionMessageResponse] | None:
        raw_payload = self._client.get(self._key(session_id))
        if raw_payload is None:
            return None
        payload = json.loads(raw_payload)
        return [SessionMessageResponse(**item) for item in payload]

    def set_messages(self, session_id: str, messages: list[SessionMessageResponse]) -> None:
        payload = json.dumps([message.model_dump() for message in messages])
        self._client.setex(self._key(session_id), INTERVIEW_MESSAGE_TTL_SECONDS, payload)

    def clear(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))


class ActiveInterviewStore:
    def __init__(self) -> None:
        self._fallback = _InMemorySessionCache()
        self._primary = _RedisSessionCache(REDIS_URL) if REDIS_URL else None
        self._warned = False

    def load_messages(self, session_id: str, persisted_messages: list[SessionMessageResponse]) -> list[SessionMessageResponse]:
        cached = self._safe_get(session_id)
        if cached is not None:
            return cached
        seeded = [message.model_copy(deep=True) for message in persisted_messages]
        self._safe_set(session_id, seeded)
        return seeded

    def append_messages(
        self,
        session_id: str,
        persisted_messages: list[SessionMessageResponse],
        new_messages: list[SessionMessageResponse],
    ) -> list[SessionMessageResponse]:
        current = self.load_messages(session_id, persisted_messages)
        updated = current + [message.model_copy(deep=True) for message in new_messages]
        self._safe_set(session_id, updated)
        return updated

    def flush_messages(self, session_id: str, persisted_messages: list[SessionMessageResponse]) -> list[SessionMessageResponse]:
        messages = self.load_messages(session_id, persisted_messages)
        self.clear(session_id)
        return messages

    def clear(self, session_id: str) -> None:
        self._safe_clear(session_id)

    def _safe_get(self, session_id: str) -> list[SessionMessageResponse] | None:
        if self._primary is None:
            return self._fallback.get_messages(session_id)
        try:
            cached = self._primary.get_messages(session_id)
            if cached is not None:
                self._fallback.set_messages(session_id, cached)
            return cached
        except RedisError as exc:
            self._warn_once(exc)
            return self._fallback.get_messages(session_id)

    def _safe_set(self, session_id: str, messages: list[SessionMessageResponse]) -> None:
        self._fallback.set_messages(session_id, messages)
        if self._primary is None:
            return
        try:
            self._primary.set_messages(session_id, messages)
        except RedisError as exc:
            self._warn_once(exc)

    def _safe_clear(self, session_id: str) -> None:
        self._fallback.clear(session_id)
        if self._primary is None:
            return
        try:
            self._primary.clear(session_id)
        except RedisError as exc:
            self._warn_once(exc)

    def _warn_once(self, exc: Exception) -> None:
        if self._warned:
            return
        LOGGER.warning("Redis session cache unavailable, falling back to in-memory storage: %s", exc)
        self._warned = True


active_interview_store = ActiveInterviewStore()