"""Demo accounts and role-tagged session tokens (HS256 JWT)."""

import secrets
import time
from dataclasses import dataclass

import jwt

from medibot.rbac import ROLES

# Demo accounts, one per role. Passwords are deliberately simple and are published in the
# README; a real deployment would hand login to the hospital's SSO and keep only the roles.
DEMO_USERS: dict[str, tuple[str, str]] = {
    "dr.mehta": ("doctor123", "doctor"),
    "nurse.priya": ("nurse123", "nurse"),
    "billing.ravi": ("billing123", "billing_executive"),
    "tech.anand": ("tech123", "technician"),
    "admin.sys": ("admin123", "admin"),
}


class InvalidTokenError(ValueError):
    pass


@dataclass(frozen=True)
class TokenClaims:
    username: str
    role: str


def authenticate(username: str, password: str) -> str | None:
    """Return the role for valid credentials, else None. Constant-time compare and a
    dummy compare for unknown users, so timing does not reveal which usernames exist."""
    stored_password, role = DEMO_USERS.get(username, ("", ""))
    if secrets.compare_digest(stored_password.encode(), password.encode()) and role:
        return role
    return None


def create_token(username: str, role: str, secret: str, ttl_hours: int) -> str:
    now = int(time.time())
    payload = {"sub": username, "role": role, "iat": now, "exp": now + ttl_hours * 3600}
    return jwt.encode(payload, secret, algorithm="HS256")


def verify_token(token: str, secret: str) -> TokenClaims:
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError as e:
        raise InvalidTokenError("Session token is invalid or has expired. Please log in again.") from e
    role = payload.get("role")
    if role not in ROLES or not payload.get("sub"):
        raise InvalidTokenError("Session token does not carry a valid role.")
    return TokenClaims(username=str(payload["sub"]), role=str(role))
