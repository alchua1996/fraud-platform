"""The decision log: one row per scored transaction, in Postgres (or SQLite for local runs).

This table is the source of truth for what the service decided and why. It's also the
training data for the next model: it stores the exact feature values used at scoring time,
so training never has to recompute them (and can't drift from what serving saw).
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    Engine,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

logger = logging.getLogger(__name__)

metadata = MetaData()

decisions = Table(
    "decisions",
    metadata,
    Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True),
    Column("transaction_id", String(64), nullable=False, unique=True),
    Column("card_id", String(64), nullable=False, index=True),
    Column("merchant_id", String(64), nullable=False),
    Column("merchant_category", String(32), nullable=False),
    Column("device_id", String(64)),
    Column("amount", Float, nullable=False),
    Column("country", String(2), nullable=False),
    Column("channel", String(16), nullable=False),
    Column("event_time", DateTime(timezone=True), nullable=False, index=True),
    Column("decision", String(16), nullable=False),
    Column("decided_by", String(16), nullable=False),
    Column("score", Float),
    Column("model_version", String(64), nullable=False),
    Column("rule_hits", JSON, nullable=False),
    Column("features", JSON, nullable=False),
    Column("latency_ms", Float, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)


def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    # Creates the table if it doesn't exist. A real service would use migrations (Alembic);
    # that's a stretch ticket.
    metadata.create_all(engine)


def log_decision(engine: Engine, row: dict[str, Any]) -> None:
    """Insert one decision. Replaying the same transaction twice is a no-op, not an error."""
    insert = pg_insert if engine.dialect.name == "postgresql" else sqlite_insert
    stmt = insert(decisions).values(**row).on_conflict_do_nothing(index_elements=["transaction_id"])
    try:
        with engine.begin() as conn:
            conn.execute(stmt)
    except Exception:
        logger.exception("failed to log decision for %s", row.get("transaction_id"))
