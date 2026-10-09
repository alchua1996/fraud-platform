"""Tier 0: hand-written rules. Fast, explainable, and, for now, the source of training labels.

Hard rules block outright, whatever the model says.
Soft rules don't decide anything by themselves when a model is running. They're returned as
reasons, logged with every decision, and used as labels when training the next model
(jobs/export_training_data.py). In rules_only mode, any soft hit sends the transaction to review.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas import Transaction

# Thresholds, gathered here so they're easy to find and tune.
EXTREME_VELOCITY_10M = 10          # earlier charges on the card in the last 10 minutes
CARD_TESTING_COUNT_10M = 4
CARD_TESTING_MAX_AMOUNT = 5.00
NEW_CARD_HIGH_AMOUNT = 500.00      # no profile yet and a big ticket
SPIKE_MIN_AMOUNT = 200.00
SPIKE_RATIO = 8.0                  # amount vs the card's 30-day average
SHARED_DEVICE_OTHER_CARDS = 3      # other cards seen on this device in the last 24h
FOREIGN_ONLINE_AMOUNT = 300.00
NEW_DEVICE_AMOUNT = 300.00
NEW_DEVICE_RATIO = 5.0

HARD_RULES = ("blocklisted_card", "extreme_velocity")


@dataclass
class RuleResult:
    hits: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(hit in HARD_RULES for hit in self.hits)


def evaluate_rules(txn: Transaction, f: dict[str, float]) -> RuleResult:
    hits: list[str] = []

    # Hard rules
    if f["blocklisted"]:
        hits.append("blocklisted_card")
    if f["card_txn_count_10m"] >= EXTREME_VELOCITY_10M:
        hits.append("extreme_velocity")

    # Soft rules
    if f["card_txn_count_10m"] >= CARD_TESTING_COUNT_10M and txn.amount < CARD_TESTING_MAX_AMOUNT:
        hits.append("card_testing_pattern")
    if not f["has_profile"] and txn.amount >= NEW_CARD_HIGH_AMOUNT:
        hits.append("new_card_high_amount")
    if f["has_profile"] and txn.amount >= SPIKE_MIN_AMOUNT and f["amount_to_avg_30d"] >= SPIKE_RATIO:
        hits.append("amount_spike")
    if f["device_other_cards_24h"] >= SHARED_DEVICE_OTHER_CARDS:
        hits.append("shared_device")
    if f["country_mismatch"] and txn.channel == "online" and txn.amount >= FOREIGN_ONLINE_AMOUNT:
        hits.append("foreign_online_high_amount")
    if f["is_new_device"] and txn.amount >= NEW_DEVICE_AMOUNT and f["amount_to_avg_30d"] >= NEW_DEVICE_RATIO:
        hits.append("new_device_high_amount")

    return RuleResult(hits)
