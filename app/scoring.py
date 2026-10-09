"""Ties the tiers together: features -> rules (tier 0) -> model (tier 1) -> decision."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from redis import Redis

from app.config import Settings
from app.features import fetch_features, record_transaction, to_vector
from app.models import ScoringModel
from app.rules import evaluate_rules
from app.schemas import ScoreResponse, Transaction


@dataclass
class ScoreResult:
    response: ScoreResponse
    features: dict[str, float]
    rule_hits: list[str]


class Scorer:
    def __init__(self, redis: Redis, model: ScoringModel | None, settings: Settings):
        self.redis = redis
        self.model = model
        self.settings = settings

    @property
    def model_version(self) -> str:
        return self.model.version if self.model is not None else "rules-only"

    def score(self, txn: Transaction) -> ScoreResult:
        t0 = perf_counter()
        features = fetch_features(self.redis, txn)
        t1 = perf_counter()
        rules = evaluate_rules(txn, features)
        t2 = perf_counter()

        score = None
        if rules.blocked:
            decision, decided_by = "block", "rules"
        elif self.model is None:
            decision, decided_by = ("review" if rules.hits else "allow"), "rules"
        else:
            score = self.model.predict_proba(to_vector(features, self.model.feature_names))
            decided_by = "model"
            if score >= self.settings.block_threshold:
                decision = "block"
            elif score >= self.settings.review_threshold:
                decision = "review"
            else:
                decision = "allow"
        t3 = perf_counter()

        # Recorded after scoring, so a transaction's features never include itself.
        record_transaction(self.redis, txn)
        t4 = perf_counter()

        timings = {
            "features": (t1 - t0) * 1000,
            "rules": (t2 - t1) * 1000,
            "model": (t3 - t2) * 1000,
            "record": (t4 - t3) * 1000,
            "total": (t4 - t0) * 1000,
        }
        response = ScoreResponse(
            transaction_id=txn.transaction_id,
            decision=decision,
            decided_by=decided_by,
            score=round(score, 6) if score is not None else None,
            reasons=rules.hits,
            model_version=self.model_version,
            timings_ms={k: round(v, 3) for k, v in timings.items()},
        )
        return ScoreResult(response=response, features=features, rule_hits=rules.hits)


def decision_row(txn: Transaction, result: ScoreResult) -> dict[str, Any]:
    """The decision-log row for one scored transaction."""
    r = result.response
    return {
        "transaction_id": txn.transaction_id,
        "card_id": txn.card_id,
        "merchant_id": txn.merchant_id,
        "merchant_category": txn.merchant_category,
        "device_id": txn.device_id,
        "amount": txn.amount,
        "country": txn.country,
        "channel": txn.channel,
        "event_time": txn.event_time,
        "decision": r.decision,
        "decided_by": r.decided_by,
        "score": r.score,
        "model_version": r.model_version,
        "rule_hits": result.rule_hits,
        "features": result.features,
        "latency_ms": r.timings_ms["total"],
    }
