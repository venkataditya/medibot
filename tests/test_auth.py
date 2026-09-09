import time

import pytest

from medibot import auth

SECRET = "unit-test-secret-that-is-long-enough-for-hs256"
OTHER = "another-unit-test-secret-long-enough-for-hs256"


def test_every_role_has_a_demo_account() -> None:
    roles = {role for _, role in auth.DEMO_USERS.values()}
    assert roles == {"doctor", "nurse", "billing_executive", "technician", "admin"}


def test_authenticate_returns_role_for_good_credentials_and_none_otherwise() -> None:
    assert auth.authenticate("dr.mehta", "doctor123") == "doctor"
    assert auth.authenticate("dr.mehta", "wrong") is None
    assert auth.authenticate("nobody", "doctor123") is None


def test_token_round_trip() -> None:
    token = auth.create_token("nurse.priya", "nurse", secret=SECRET, ttl_hours=1)
    claims = auth.verify_token(token, secret=SECRET)
    assert claims == auth.TokenClaims(username="nurse.priya", role="nurse")


def test_token_signed_with_another_secret_is_rejected() -> None:
    token = auth.create_token("nurse.priya", "nurse", secret=SECRET, ttl_hours=1)
    with pytest.raises(auth.InvalidTokenError):
        auth.verify_token(token, secret=OTHER)


def test_expired_token_is_rejected(monkeypatch) -> None:
    now = time.time()
    monkeypatch.setattr(auth.time, "time", lambda: now - 7200)
    token = auth.create_token("nurse.priya", "nurse", secret=SECRET, ttl_hours=1)
    monkeypatch.setattr(auth.time, "time", lambda: now)
    with pytest.raises(auth.InvalidTokenError):
        auth.verify_token(token, secret=SECRET)


def test_garbage_and_role_tampering_are_rejected() -> None:
    with pytest.raises(auth.InvalidTokenError):
        auth.verify_token("not.a.token", secret=SECRET)
    forged = auth.create_token("x", "ceo", secret=SECRET, ttl_hours=1)
    with pytest.raises(auth.InvalidTokenError):
        auth.verify_token(forged, secret=SECRET)
