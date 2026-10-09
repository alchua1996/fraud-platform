"""Tier 1: the model. Anything that turns a feature vector into a fraud probability.

The service only depends on the ScoringModel interface below, so changing models is a
configuration change: point MODEL_PATH at a different model and restart.

Today the only implementation is a hand-weighted placeholder. Training a real model and
teaching load_model() to load it is ticket FRAUD-1 in TICKETS.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from app.features import FEATURE_NAMES


class ScoringModel(Protocol):
    version: str
    feature_names: list[str]

    def predict_proba(self, x: Sequence[float]) -> float:
        """Probability that the transaction should be flagged, for one feature vector."""
        ...


class HeuristicModel:
    """A logistic function over hand-picked weights. A stand-in, not a trained model."""

    version = "heuristic-v0"
    BIAS = -4.0
    WEIGHTS = {
        "log_amount": 0.4,
        "is_online": 0.5,
        "is_night": 0.6,
        "is_high_risk_category": 0.8,
        "card_txn_count_10m": 0.6,
        "card_txn_count_1h": 0.2,
        "card_txn_count_24h": 0.05,
        "card_amount_sum_24h_log": 0.1,
        "seconds_since_last_txn_log": -0.15,
        "device_other_cards_24h": 0.9,
        "is_new_device": 1.2,
        "has_profile": -1.0,
        "amount_to_avg_30d": 0.12,
        "profile_txn_count_30d": -0.01,
        "country_mismatch": 1.0,
    }

    def __init__(self) -> None:
        self.feature_names = list(FEATURE_NAMES)
        self._weights = [self.WEIGHTS.get(name, 0.0) for name in self.feature_names]

    def predict_proba(self, x: Sequence[float]) -> float:
        z = self.BIAS + sum(w * v for w, v in zip(self._weights, x))
        z = max(min(z, 50.0), -50.0)
        return 1.0 / (1.0 + math.exp(-z))


def load_model(path: str) -> ScoringModel:
    """Load the model named by MODEL_PATH. Called once, at startup."""
    if path == "builtin:heuristic":
        return HeuristicModel()

    # TODO(FRAUD-1): load a trained model artifact, for example a joblib file holding the
    # fitted estimator, the feature names it was trained on and a version string, and return
    # an object that satisfies ScoringModel. Refuse to load it if its feature names don't
    # match what the service can compute.
    raise NotImplementedError(
        f"Can't load {path!r} yet: loading trained models is ticket FRAUD-1 in TICKETS.md"
    )
