from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7
DEFAULT_FRONTEND_ORIGIN = "http://localhost:5173"
DEFAULT_INTERVIEW_MESSAGE_TTL_SECONDS = 60 * 60 * 6


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


_DOTENV = _read_dotenv(REPO_ROOT / ".env")


def env_value(*names: str, default: str = "") -> str:
    for name in names:
        candidate = os.getenv(name)
        if candidate:
            return candidate
        if name in _DOTENV and _DOTENV[name]:
            return _DOTENV[name]
    return default


SUPABASE_URL = env_value("SUPABASE_URL", "supabase_url")
SUPABASE_SERVICE_ROLE_KEY = env_value("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_KEY", "supabase_key")
FRONTEND_ORIGIN = env_value("FRONTEND_ORIGIN", default=DEFAULT_FRONTEND_ORIGIN)
REDIS_URL = env_value("REDIS_URL")
INTERVIEW_MESSAGE_TTL_SECONDS = int(env_value("INTERVIEW_MESSAGE_TTL_SECONDS", default=str(DEFAULT_INTERVIEW_MESSAGE_TTL_SECONDS)))


def validate_supabase_config() -> None:
    missing: list[str] = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_SERVICE_ROLE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    if missing:
        raise RuntimeError(f"Missing Supabase configuration: {', '.join(missing)}")

