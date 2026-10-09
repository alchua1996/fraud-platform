"""The scoring service.

Run locally (needs Redis on localhost:6379):
    uvicorn app.main:app --reload
Then open http://localhost:8000/docs to send test requests from the browser.
"""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from redis import Redis
from sqlalchemy import Engine, text

from app import __version__
from app.config import Settings, get_settings
from app.db import init_db, log_decision, make_engine
from app.models import load_model
from app.schemas import ScoreResponse, Transaction
from app.scoring import Scorer, decision_row

logger = logging.getLogger("fraud")


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    expected = request.app.state.settings.api_key
    if expected and not secrets.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key header")


def create_app(
    settings: Settings | None = None,
    redis_client: Redis | None = None,
    engine: Engine | None = None,
) -> FastAPI:
    """Build the app. Tests pass in their own Redis and database; production uses settings."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Expensive setup happens once, at startup, never per request.
        logging.basicConfig(
            level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        r = redis_client or Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=settings.redis_timeout_s,
            socket_connect_timeout=settings.redis_timeout_s,
        )
        db = engine or make_engine(settings.database_url)
        init_db(db)
        model = load_model(settings.model_path) if settings.scoring_mode == "rules_then_model" else None

        app.state.settings = settings
        app.state.redis = r
        app.state.engine = db
        app.state.scorer = Scorer(r, model, settings)
        logger.info("ready: scoring_mode=%s model=%s", settings.scoring_mode, app.state.scorer.model_version)
        yield
        if redis_client is None:
            r.close()
        if engine is None:
            db.dispose()

    app = FastAPI(title="Fraud scoring service", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> JSONResponse:
        checks: dict[str, str] = {}
        try:
            request.app.state.redis.ping()
            checks["redis"] = "ok"
        except Exception as exc:
            checks["redis"] = f"error: {exc.__class__.__name__}"
        try:
            with request.app.state.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            checks["database"] = "ok"
        except Exception as exc:
            checks["database"] = f"error: {exc.__class__.__name__}"
        healthy = all(value == "ok" for value in checks.values())
        return JSONResponse(
            {"status": "ok" if healthy else "degraded", **checks}, status_code=200 if healthy else 503
        )

    @app.get("/version")
    def version(request: Request) -> dict:
        s = request.app.state.settings
        return {
            "app_version": __version__,
            "model_version": request.app.state.scorer.model_version,
            "scoring_mode": s.scoring_mode,
            "review_threshold": s.review_threshold,
            "block_threshold": s.block_threshold,
        }

    @app.post("/score", response_model=ScoreResponse, dependencies=[Depends(require_api_key)])
    def score(txn: Transaction, request: Request, background: BackgroundTasks) -> ScoreResponse:
        # A plain `def` (not `async def`) endpoint: FastAPI runs it in a thread pool, which is
        # what makes the blocking Redis and database clients safe to use here.
        result = request.app.state.scorer.score(txn)
        if request.app.state.settings.decision_log_enabled:
            # Runs after the response is sent, so the caller doesn't wait on the database.
            # It still takes up this worker's time, though; see ticket FRAUD-3.
            background.add_task(log_decision, request.app.state.engine, decision_row(txn, result))
        return result.response

    return app


app = create_app()
