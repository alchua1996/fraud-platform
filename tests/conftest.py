from datetime import datetime, timedelta, timezone

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import make_engine
from app.main import create_app
from app.schemas import Transaction

T0 = datetime(2026, 9, 24, 14, 0, tzinfo=timezone.utc)


def make_txn(i: int = 0, minutes: float = 0, **overrides) -> Transaction:
    fields = {
        "transaction_id": f"t_test_{i:05d}",
        "card_id": "c_000001",
        "merchant_id": "m_00_001",
        "merchant_category": "grocery",
        "amount": 42.0,
        "country": "US",
        "channel": "card_present",
        "device_id": None,
        "event_time": T0 + timedelta(minutes=minutes),
    }
    fields.update(overrides)
    return Transaction(**fields)


@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def engine(tmp_path):
    return make_engine(f"sqlite:///{tmp_path / 'test.db'}")


@pytest.fixture
def settings():
    return Settings(_env_file=None, api_key="test-key", database_url="sqlite://")


@pytest.fixture
def client(settings, redis_client, engine):
    app = create_app(settings=settings, redis_client=redis_client, engine=engine)
    with TestClient(app, headers={"X-API-Key": "test-key"}) as c:
        yield c
