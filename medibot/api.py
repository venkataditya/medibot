"""FastAPI backend: login, role-scoped chat, collections per role, health.

    uv run uvicorn medibot.api:app --reload
"""

import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

from medibot import auth, rbac
from medibot.config import Settings, get_settings
from medibot.llm import LLMError
from medibot.pipeline import build_medibot
from medibot.sql_rag import SQLRAGError

LOGIN_LIMIT_PER_MINUTE = 20
CHAT_LIMIT_PER_MINUTE = 30


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class LoginResponse(BaseModel):
    token: str
    username: str
    role: str
    collections: list[str]
    sql_access: bool


class ChatRequest(BaseModel):
    question: str = Field(max_length=2000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class SourceOut(BaseModel):
    source_document: str
    section_title: str
    collection: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    retrieval_type: str
    role: str
    access_denied: bool


class CollectionsResponse(BaseModel):
    role: str
    collections: list[str]
    sql_access: bool


class RateLimiter:
    """Small in-memory sliding window per client address. Enough to blunt brute-force
    logins and runaway clients on a single-process deployment."""

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
        hits.append(now)


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def create_app(settings: Settings | None = None, bot_factory: Callable[[], Any] = build_medibot) -> FastAPI:
    settings = settings or get_settings()
    login_limiter = RateLimiter(LOGIN_LIMIT_PER_MINUTE)
    chat_limiter = RateLimiter(CHAT_LIMIT_PER_MINUTE)
    bearer = HTTPBearer(auto_error=False)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bot = bot_factory()  # loads the index and models once
        yield

    app = FastAPI(title="MediBot", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> auth.TokenClaims:
        if credentials is None:
            raise HTTPException(status_code=401, detail="Log in to use MediBot.")
        try:
            return auth.verify_token(credentials.credentials, settings.medibot_secret)
        except auth.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=str(e)) from e

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {"status": "ok", "model": settings.groq_model, "collection": settings.collection_name}

    @app.post("/login", response_model=LoginResponse)
    def login(body: LoginRequest, request: Request) -> LoginResponse:
        login_limiter.check(_client_key(request))
        role = auth.authenticate(body.username, body.password)
        if role is None:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        token = auth.create_token(body.username, role, settings.medibot_secret, settings.token_ttl_hours)
        return LoginResponse(
            token=token, username=body.username, role=role,
            collections=rbac.collections_for(role), sql_access=rbac.can_use_sql(role),
        )

    @app.get("/collections/{role}", response_model=CollectionsResponse)
    def collections(role: str) -> CollectionsResponse:
        try:
            return CollectionsResponse(role=role, collections=rbac.collections_for(role), sql_access=rbac.can_use_sql(role))
        except rbac.UnknownRoleError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request, user: auth.TokenClaims = Depends(current_user)) -> ChatResponse:
        chat_limiter.check(user.username)
        # The role comes from the signed token only. Anything in the body is ignored.
        try:
            result = request.app.state.bot.chat(body.question, user.role)
        except (LLMError, SQLRAGError) as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        return ChatResponse(
            answer=result.answer,
            sources=[SourceOut(**vars(s)) for s in result.sources],
            retrieval_type=result.retrieval_type,
            role=result.role,
            access_denied=result.access_denied,
        )

    return app


app = create_app()
