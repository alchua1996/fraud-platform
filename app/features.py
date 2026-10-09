"""Online features: what the service knows about a card at the moment a transaction arrives.

Three kinds of features end up in the vector the model sees:
    request-time  computed from the request itself (amount, channel, hour)
    real-time     kept in Redis and updated by this service on every transaction
    batch         computed nightly by jobs/build_profiles.py and loaded into Redis

Redis keys
    card:{card_id}:txns       sorted set  member "txn_id|amount", score = event time (epoch seconds).
                                          Trimmed to the last 24 hours of event time.
    device:{device_id}:cards  sorted set  member card_id, score = last time that card used the device
    card:{card_id}:devices    set         devices the card has used (seeded by the batch job)
    card:{card_id}:profile    hash        avg_amount_30d, txn_count_30d, n_countries_30d,
                                          home_country, as_of (written by the batch job)
    blocklist:cards           set         cards reported stolen (written by the batch job)

All reads for one transaction go out in a single pipeline, so they cost one network round trip.
Recording the transaction afterwards is a second pipeline (a second round trip).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import timezone

from redis import Redis

from app.schemas import Transaction

WINDOW_S = 24 * 3600                  # how far back real-time features look, in event time
REALTIME_TTL_S = 48 * 3600            # wall-clock expiry so idle keys don't pile up
DEVICE_HISTORY_TTL_S = 35 * 24 * 3600
HIGH_RISK_CATEGORIES = frozenset({"gift_cards", "electronics", "digital_goods"})
MAX_AMOUNT_RATIO = 50.0
BLOCKLIST_KEY = "blocklist:cards"

# The model's input, in order. Training code must build vectors in exactly this order;
# to_vector() is the one place that does it, for serving and training alike.
FEATURE_NAMES = [
    "log_amount",
    "is_online",
    "is_night",
    "is_high_risk_category",
    "card_txn_count_10m",
    "card_txn_count_1h",
    "card_txn_count_24h",
    "card_amount_sum_24h_log",
    "seconds_since_last_txn_log",
    "device_other_cards_24h",
    "is_new_device",
    "has_profile",
    "amount_to_avg_30d",
    "profile_txn_count_30d",
    "country_mismatch",
]


def card_txns_key(card_id: str) -> str:
    return f"card:{card_id}:txns"


def card_devices_key(card_id: str) -> str:
    return f"card:{card_id}:devices"


def card_profile_key(card_id: str) -> str:
    return f"card:{card_id}:profile"


def device_cards_key(device_id: str) -> str:
    return f"device:{device_id}:cards"


def fetch_features(r: Redis, txn: Transaction) -> dict[str, float]:
    """Read everything Redis knows about this transaction's card and device, in one round trip."""
    ts = txn.event_time.timestamp()
    pipe = r.pipeline(transaction=False)
    pipe.zrangebyscore(card_txns_key(txn.card_id), ts - WINDOW_S, ts, withscores=True)
    pipe.hgetall(card_profile_key(txn.card_id))
    pipe.sismember(BLOCKLIST_KEY, txn.card_id)
    if txn.device_id:
        pipe.zcount(device_cards_key(txn.device_id), ts - WINDOW_S, ts)
        pipe.zscore(device_cards_key(txn.device_id), txn.card_id)
        pipe.sismember(card_devices_key(txn.card_id), txn.device_id)
    results = pipe.execute()

    recent, profile, blocklisted = results[0], results[1], results[2]
    if txn.device_id:
        device_cards_24h, card_last_on_device, device_known = results[3], results[4], bool(results[5])
    else:
        device_cards_24h, card_last_on_device, device_known = 0, None, True
    return build_features(
        txn, recent, profile, bool(blocklisted), device_cards_24h, card_last_on_device, device_known
    )


def build_features(
    txn: Transaction,
    recent: Sequence[tuple[str, float]],
    profile: dict[str, str],
    blocklisted: bool,
    device_cards_24h: int = 0,
    card_last_on_device: float | None = None,
    device_known: bool = True,
) -> dict[str, float]:
    """Turn raw Redis values into features. Pure function, so it's easy to test."""
    ts = txn.event_time.timestamp()
    times = [score for _, score in recent]
    amounts = [float(member.rsplit("|", 1)[1]) for member, _ in recent]
    seconds_since_last = ts - max(times) if times else WINDOW_S

    has_profile = bool(profile)
    avg_amount = float(profile.get("avg_amount_30d", 0.0)) if has_profile else 0.0

    # The device's distinct cards in the last day, not counting this card itself.
    other_cards = device_cards_24h
    if card_last_on_device is not None and card_last_on_device >= ts - WINDOW_S:
        other_cards -= 1

    return {
        # request-time
        "log_amount": math.log1p(txn.amount),
        "is_online": float(txn.channel == "online"),
        "is_night": float(txn.event_time.astimezone(timezone.utc).hour < 6),
        "is_high_risk_category": float(txn.merchant_category in HIGH_RISK_CATEGORIES),
        # real-time
        "card_txn_count_10m": float(sum(1 for t in times if t >= ts - 600)),
        "card_txn_count_1h": float(sum(1 for t in times if t >= ts - 3600)),
        "card_txn_count_24h": float(len(times)),
        "card_amount_sum_24h_log": math.log1p(sum(amounts)),
        "seconds_since_last_txn_log": math.log1p(max(seconds_since_last, 0.0)),
        "device_other_cards_24h": float(max(other_cards, 0)),
        "is_new_device": float(txn.device_id is not None and not device_known),
        # batch, and features derived from batch values. A missing profile is itself a signal,
        # so it gets its own flag instead of being filled in with "average customer" values.
        "has_profile": float(has_profile),
        "amount_to_avg_30d": min(txn.amount / (avg_amount + 1.0), MAX_AMOUNT_RATIO) if has_profile else 0.0,
        "profile_txn_count_30d": float(profile.get("txn_count_30d", 0.0)) if has_profile else 0.0,
        "country_mismatch": float(has_profile and txn.country != profile.get("home_country")),
        # used by the rules, not part of the model input
        "blocklisted": float(blocklisted),
    }


def to_vector(features: dict[str, float], names: Sequence[str] = FEATURE_NAMES) -> list[float]:
    return [float(features[name]) for name in names]


def record_transaction(r: Redis, txn: Transaction) -> None:
    """Add this transaction to the card's and device's recent history."""
    ts = txn.event_time.timestamp()
    card_key = card_txns_key(txn.card_id)
    pipe = r.pipeline(transaction=False)
    pipe.zadd(card_key, {f"{txn.transaction_id}|{txn.amount:.2f}": ts})
    pipe.zremrangebyscore(card_key, "-inf", f"({ts - WINDOW_S}")  # drop events older than 24h
    pipe.expire(card_key, REALTIME_TTL_S)
    if txn.device_id:
        device_key = device_cards_key(txn.device_id)
        pipe.zadd(device_key, {txn.card_id: ts}, gt=True)  # gt: keep the latest time if events arrive out of order
        pipe.zremrangebyscore(device_key, "-inf", f"({ts - WINDOW_S}")
        pipe.expire(device_key, REALTIME_TTL_S)
        devices_key = card_devices_key(txn.card_id)
        pipe.sadd(devices_key, txn.device_id)
        pipe.expire(devices_key, DEVICE_HISTORY_TTL_S)
    pipe.execute()
