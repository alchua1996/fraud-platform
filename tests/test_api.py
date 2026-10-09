from sqlalchemy import select

from app.db import decisions

from .conftest import make_txn


def payload(txn):
    return txn.model_dump(mode="json")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "redis": "ok", "database": "ok"}


def test_score_returns_a_decision_and_logs_it(client, engine):
    txn = make_txn()
    r = client.post("/score", json=payload(txn))

    assert r.status_code == 200
    body = r.json()
    assert body["transaction_id"] == txn.transaction_id
    assert body["decision"] == "allow"
    assert body["decided_by"] == "model"
    assert body["model_version"] == "heuristic-v0"
    assert body["timings_ms"]["total"] > 0

    with engine.connect() as conn:
        rows = conn.execute(select(decisions)).mappings().all()
    assert len(rows) == 1
    assert rows[0]["decision"] == "allow"
    assert "log_amount" in rows[0]["features"]


def test_replaying_a_transaction_does_not_duplicate_the_log(client, engine):
    txn = make_txn()
    client.post("/score", json=payload(txn))
    client.post("/score", json=payload(txn))

    with engine.connect() as conn:
        rows = conn.execute(select(decisions)).all()
    assert len(rows) == 1


def test_blocklisted_card_is_blocked_by_rules(client, redis_client):
    redis_client.sadd("blocklist:cards", "c_000001")
    body = client.post("/score", json=payload(make_txn())).json()
    assert body["decision"] == "block"
    assert body["decided_by"] == "rules"
    assert body["score"] is None


def test_api_key_is_required_when_configured(client):
    r = client.post("/score", json=payload(make_txn()), headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_invalid_payload_is_rejected(client):
    bad = payload(make_txn())
    bad["amount"] = -5
    assert client.post("/score", json=bad).status_code == 422
