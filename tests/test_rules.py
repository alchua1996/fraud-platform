from app.features import build_features
from app.rules import evaluate_rules

from .conftest import make_txn

PROFILE = {"avg_amount_30d": "40.0", "txn_count_30d": "45", "home_country": "US"}


def rules_for(txn, recent=(), profile=PROFILE, blocklisted=False, **device):
    return evaluate_rules(txn, build_features(txn, list(recent), profile, blocklisted, **device))


def test_blocklisted_card_is_blocked():
    result = rules_for(make_txn(), blocklisted=True)
    assert result.blocked
    assert "blocklisted_card" in result.hits


def test_extreme_velocity_is_blocked():
    txn = make_txn(minutes=10)
    recent = [(f"t_{i}|2.00", txn.event_time.timestamp() - 30 * (i + 1)) for i in range(10)]
    assert rules_for(txn, recent).blocked


def test_card_testing_pattern_is_a_soft_hit():
    txn = make_txn(amount=1.50, minutes=10)
    recent = [(f"t_{i}|2.00", txn.event_time.timestamp() - 60 * (i + 1)) for i in range(5)]
    result = rules_for(txn, recent)
    assert not result.blocked
    assert "card_testing_pattern" in result.hits


def test_amount_spike_needs_a_profile():
    big = make_txn(amount=900.0)
    assert "amount_spike" in rules_for(big).hits
    assert "amount_spike" not in rules_for(big, profile={}).hits
    assert "new_card_high_amount" in rules_for(big, profile={}).hits


def test_ordinary_purchase_has_no_hits():
    assert rules_for(make_txn()).hits == []
