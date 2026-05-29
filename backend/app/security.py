from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets


PBKDF2_ITERATIONS = 310_000


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"{PBKDF2_ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        iterations_text, salt_text, digest_text = password_hash.split("$", 2)
        iterations = int(iterations_text)
        salt = base64.b64decode(salt_text.encode())
        expected = base64.b64decode(digest_text.encode())
    except (TypeError, ValueError):
        return False

    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, expected)


def create_token() -> str:
    return secrets.token_urlsafe(32)