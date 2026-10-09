"""Simulate a card issuer's transactions, with fraud mixed in.

Run:
    python -m simulator.generate                     # 5,000 cards, 30 days, last 7 days are "live"
    python -m simulator.generate --cards 20000 --days 60 --live-days 14

Writes to data/:
    history.parquet       Transactions before the cutoff. Plays the role of the data warehouse:
                          batch jobs read it, the API never sees it.
    live.parquet          Transactions after the cutoff, in time order. scripts/replay.py sends
                          these to the API as if they were happening now.
    ground_truth.parquet  transaction_id -> is_fraud, scenario. The answer key. Only evaluation
                          code may read it; nothing in app/ or jobs/ is allowed to.
    blocklist.txt         Cards reported stolen during the history window.

Fraud scenarios, each designed so a different part of the system has a chance to catch it:
    card_testing      Burst of tiny charges within minutes, sometimes followed by a big one.
                      Velocity rules territory.
    stolen_card       A few large purchases in risky categories within hours, from a new device,
                      often abroad. Amount-vs-profile features.
    account_takeover  One to three charges from a new device at night, several times the card's
                      usual amount. Hard: looks a lot like a legitimate splurge.
    device_ring       A few devices used across many different cards within a day. Shared-device
                      features now, graph features later.
    blocklist_retry   A card that was already reported stolen gets used again. Rules territory.

Fraud is deliberately over-represented compared with real life (well under 1% of transactions
in most portfolios) so there are enough positives to learn from. Tune it with --fraud-rate.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = np.array([
    "grocery", "restaurants", "gas", "pharmacy", "entertainment",
    "online_retail", "digital_goods", "electronics", "gift_cards", "travel",
])
CAT = {name: i for i, name in enumerate(CATEGORIES)}

# Spend relative to the card's baseline, and how likely a purchase in the category happens
# online (multiplied by each customer's own online habit).
AMOUNT_MULT = np.array([1.0, 0.8, 1.0, 0.7, 1.0, 1.5, 0.4, 4.0, 2.0, 6.0])
ONLINE_FACTOR = np.array([0.15, 0.2, 0.0, 0.2, 1.0, 2.0, 2.5, 1.0, 1.2, 1.5])
# Which categories tend to be a customer's regulars.
FAVORITE_WEIGHTS = np.array([0.20, 0.15, 0.12, 0.08, 0.10, 0.12, 0.06, 0.06, 0.04, 0.07])

FOREIGN_COUNTRIES = np.array(["CA", "GB", "MX", "DE", "FR", "BR", "NG", "RO", "VN", "PH"])
MERCHANTS_PER_CATEGORY = 40
MERCHANT_IDS = np.array(
    [f"m_{c:02d}_{j:03d}" for c in range(len(CATEGORIES)) for j in range(MERCHANTS_PER_CATEGORY)]
)

# Purchases cluster in the daytime and evening.
HOUR_WEIGHTS = np.array(
    [1, 0.5, 0.3, 0.2, 0.2, 0.4, 1, 2, 3, 4, 5, 6, 7, 6, 5, 5, 6, 7, 8, 7, 6, 4, 3, 2], dtype=float
)
HOUR_WEIGHTS /= HOUR_WEIGHTS.sum()

DAY = 86_400
HOUR = 3_600

OUTPUT_COLUMNS = [
    "transaction_id", "card_id", "merchant_id", "merchant_category", "amount",
    "country", "channel", "device_id", "event_time",
]


def new_device_ids(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.array([f"d_{x:010x}" for x in rng.integers(0, 2**40, n)], dtype=object)


def merchant_ids(rng: np.random.Generator, cat_idx: np.ndarray) -> np.ndarray:
    return MERCHANT_IDS[cat_idx * MERCHANTS_PER_CATEGORY + rng.integers(0, MERCHANTS_PER_CATEGORY, len(cat_idx))]


@dataclass
class Population:
    card_ids: np.ndarray
    home_country: np.ndarray
    daily_rate: np.ndarray     # average transactions per day
    amount_scale: np.ndarray   # typical purchase size in dollars
    online_ratio: np.ndarray   # how much this customer shops online
    favorites: np.ndarray      # (n_cards, 3) category indices
    devices: np.ndarray        # (n_cards, 2) the customer's own devices


def make_population(rng: np.random.Generator, n_cards: int) -> Population:
    home = np.where(rng.random(n_cards) < 0.9, "US", rng.choice(FOREIGN_COUNTRIES[:5], n_cards))
    favorites = np.stack(
        [rng.choice(len(CATEGORIES), 3, replace=False, p=FAVORITE_WEIGHTS) for _ in range(n_cards)]
    )
    return Population(
        card_ids=np.array([f"c_{i:06d}" for i in range(n_cards)]),
        home_country=home,
        daily_rate=rng.gamma(shape=2.0, scale=0.8, size=n_cards),
        amount_scale=rng.lognormal(mean=np.log(40), sigma=0.5, size=n_cards),
        online_ratio=rng.beta(2, 3, n_cards),
        favorites=favorites,
        devices=new_device_ids(rng, n_cards * 2).reshape(n_cards, 2),
    )


def legit_transactions(rng: np.random.Generator, pop: Population, days: int) -> pd.DataFrame:
    """Everyday spending. Times are seconds since the simulation start."""
    n_cards = len(pop.card_ids)
    card_idx = np.repeat(np.arange(n_cards), rng.poisson(pop.daily_rate * days))
    n = len(card_idx)

    day = rng.integers(0, days, n)
    seconds = day * DAY + rng.choice(24, n, p=HOUR_WEIGHTS) * HOUR + rng.integers(0, HOUR, n)

    use_favorite = rng.random(n) < 0.7
    cat = np.where(
        use_favorite,
        pop.favorites[card_idx, rng.integers(0, 3, n)],
        rng.integers(0, len(CATEGORIES), n),
    )
    amount = pop.amount_scale[card_idx] * AMOUNT_MULT[cat] * rng.lognormal(0, 0.6, n)
    online = rng.random(n) < np.clip(pop.online_ratio[card_idx] * ONLINE_FACTOR[cat], 0, 1)

    # Online purchases come from one of the customer's devices, occasionally a new one
    # (new phone, a partner's laptop). Legitimate "new device" traffic keeps that signal honest.
    pick = rng.random(n)
    device = np.where(pick < 0.7, pop.devices[card_idx, 0], pop.devices[card_idx, 1]).astype(object)
    brand_new = pick > 0.95
    device[brand_new] = new_device_ids(rng, int(brand_new.sum()))
    device[~online] = None

    # About 5% of customers take a trip; in-person purchases during it happen abroad.
    country = pop.home_country[card_idx].astype(object)
    trip_start = np.where(rng.random(n_cards) < 0.05, rng.integers(0, days, n_cards), -10_000)
    trip_len = rng.integers(3, 7, n_cards)
    trip_country = rng.choice(FOREIGN_COUNTRIES[:5], n_cards)
    abroad = (
        (day >= trip_start[card_idx]) & (day < trip_start[card_idx] + trip_len[card_idx]) & ~online
    )
    country[abroad] = trip_country[card_idx][abroad]

    return pd.DataFrame({
        "card_idx": card_idx,
        "t": seconds.astype(float),
        "merchant_id": merchant_ids(rng, cat),
        "merchant_category": CATEGORIES[cat],
        "amount": np.round(np.maximum(amount, 1.0), 2),
        "country": country,
        "channel": np.where(online, "online", "card_present"),
        "device_id": device,
        "is_fraud": 0,
        "scenario": "legit",
    })


class FraudBuilder:
    """Collects fraudulent transactions episode by episode."""

    def __init__(self, rng: np.random.Generator, pop: Population):
        self.rng = rng
        self.pop = pop
        self.rows: list[dict] = []
        self.episode_end: dict[int, float] = {}  # card_idx -> time of the last fraudulent charge

    def _add(self, card_idx, t, amount, category, device, country, scenario):
        cat_idx = CAT[category]
        self.rows.append({
            "card_idx": card_idx,
            "t": float(t),
            "merchant_id": merchant_ids(self.rng, np.array([cat_idx]))[0],
            "merchant_category": category,
            "amount": round(max(float(amount), 0.5), 2),
            "country": country,
            "channel": "online",
            "device_id": device,
            "is_fraud": 1,
            "scenario": scenario,
        })
        self.episode_end[card_idx] = max(self.episode_end.get(card_idx, 0.0), float(t))

    def _fraud_country(self, card_idx, p_foreign):
        if self.rng.random() < p_foreign:
            return str(self.rng.choice(FOREIGN_COUNTRIES))
        return str(self.pop.home_country[card_idx])

    def card_testing(self, card_idx, t0):
        rng = self.rng
        device = new_device_ids(rng, 1)[0]
        country = self._fraud_country(card_idx, 0.5)
        t = t0
        for _ in range(rng.integers(5, 16)):
            t += rng.integers(20, 91)
            self._add(card_idx, t, rng.uniform(0.5, 3.0), "digital_goods", device, country, "card_testing")
        if rng.random() < 0.5:  # the card works, so cash out
            for _ in range(rng.integers(1, 3)):
                t += rng.uniform(10 * 60, 2 * HOUR)
                category = str(rng.choice(["electronics", "gift_cards"]))
                self._add(card_idx, t, rng.uniform(300, 1500), category, device, country, "card_testing")

    def stolen_card(self, card_idx, t0):
        rng = self.rng
        device = new_device_ids(rng, 1)[0]
        country = self._fraud_country(card_idx, 0.6)
        for t in np.sort(t0 + rng.uniform(0, 6 * HOUR, rng.integers(3, 9))):
            category = str(rng.choice(["electronics", "gift_cards", "online_retail"]))
            self._add(card_idx, t, rng.uniform(200, 2000), category, device, country, "stolen_card")

    def account_takeover(self, card_idx, t0):
        rng = self.rng
        device = new_device_ids(rng, 1)[0]
        night = (t0 // DAY) * DAY + rng.integers(0, 6) * HOUR + rng.integers(0, HOUR)
        usual = self.pop.amount_scale[card_idx]
        for t in np.sort(night + rng.uniform(0, HOUR, rng.integers(1, 4))):
            category = str(rng.choice(["online_retail", "electronics", "gift_cards", "travel", "digital_goods"]))
            self._add(card_idx, t, usual * rng.uniform(5, 20), category, device,
                      str(self.pop.home_country[card_idx]), "account_takeover")

    def device_ring(self, card_idxs, t0):
        rng = self.rng
        ring_devices = new_device_ids(rng, int(rng.integers(2, 4)))
        for card_idx in card_idxs:
            country = self._fraud_country(card_idx, 0.3)
            for _ in range(rng.integers(1, 4)):
                category = str(rng.choice(["online_retail", "gift_cards", "electronics"]))
                self._add(card_idx, t0 + rng.uniform(0, DAY), rng.uniform(80, 400), category,
                          str(rng.choice(ring_devices)), country, "device_ring")

    def blocklist_retry(self, card_idx, t0):
        rng = self.rng
        device = new_device_ids(rng, 1)[0]
        country = self._fraud_country(card_idx, 0.5)
        for t in np.sort(t0 + rng.uniform(0, 2 * HOUR, rng.integers(1, 4))):
            category = str(rng.choice(["electronics", "gift_cards", "online_retail"]))
            self._add(card_idx, t, rng.uniform(100, 800), category, device, country, "blocklist_retry")

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def inject_fraud(rng, pop, fraud, window_start_day, window_days, fraud_rate, eligible):
    """Add single-card episodes and device rings within one time window."""
    lo, hi = window_start_day * DAY, (window_start_day + window_days) * DAY
    n_single = max(5, round(len(pop.card_ids) * 0.0012 * window_days * fraud_rate))
    n_rings = max(2, round(len(pop.card_ids) * 0.00015 * window_days * fraud_rate))

    victims = rng.choice(eligible, size=min(len(eligible), n_single + n_rings * 15), replace=False)
    singles, ring_pool = victims[:n_single], list(victims[n_single:])

    scenarios = rng.choice(["card_testing", "stolen_card", "account_takeover"], n_single, p=[0.35, 0.35, 0.30])
    for card_idx, scenario in zip(singles, scenarios):
        getattr(fraud, scenario)(int(card_idx), rng.uniform(lo, hi - HOUR))

    for _ in range(n_rings):
        size = int(rng.integers(6, 16))
        members, ring_pool = ring_pool[:size], ring_pool[size:]
        if members:
            fraud.device_ring([int(c) for c in members], rng.uniform(lo, hi - DAY))
    return set(int(v) for v in victims)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cards", type=int, default=5_000)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--live-days", type=int, default=7)
    parser.add_argument("--start", default="2026-09-01", help="first day of the simulation (UTC)")
    parser.add_argument("--fraud-rate", type=float, default=1.0, help="multiplier on the amount of fraud")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    if not 0 < args.live_days < args.days:
        parser.error("--live-days must be between 0 and --days")

    began = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    history_days = args.days - args.live_days
    start = pd.Timestamp(args.start, tz="UTC")

    pop = make_population(rng, args.cards)
    legit = legit_transactions(rng, pop, args.days)
    fraud = FraudBuilder(rng, pop)

    all_cards = np.arange(args.cards)
    history_victims = inject_fraud(rng, pop, fraud, 0, history_days, args.fraud_rate, all_cards)

    # Most victims in the history window notice and report the card a day or few later.
    # The bank closes it (adds it to the blocklist) and the customer moves to a new card,
    # so the old card's legitimate spending stops at the report.
    reported_at: dict[int, float] = {}
    for card_idx in history_victims:
        if card_idx in fraud.episode_end and rng.random() < 0.7:
            t_report = fraud.episode_end[card_idx] + rng.uniform(1, 3) * DAY
            if t_report < history_days * DAY:
                reported_at[card_idx] = t_report

    live_eligible = np.setdiff1d(all_cards, list(history_victims))
    inject_fraud(rng, pop, fraud, history_days, args.live_days, args.fraud_rate, live_eligible)

    # Fraudsters keep trying some of the cards that were already reported.
    lo, hi = history_days * DAY, args.days * DAY
    for card_idx in reported_at:
        if rng.random() < 0.3:
            fraud.blocklist_retry(card_idx, rng.uniform(lo, hi - 2 * HOUR))

    report_time = legit["card_idx"].map(reported_at)
    legit = legit[report_time.isna() | (legit["t"] < report_time)]

    txns = pd.concat([legit, fraud.frame()], ignore_index=True)
    txns = txns[txns["t"] < args.days * DAY].sort_values("t", kind="stable").reset_index(drop=True)
    txns["event_time"] = start + pd.to_timedelta(txns["t"], unit="s")
    txns["event_time"] = txns["event_time"].dt.floor("s")
    txns["card_id"] = pop.card_ids[txns["card_idx"].to_numpy()]
    txns["transaction_id"] = [f"t_{i:08d}" for i in range(len(txns))]

    cutoff = start + pd.Timedelta(days=history_days)
    is_live = txns["event_time"] >= cutoff

    args.out.mkdir(parents=True, exist_ok=True)
    txns.loc[~is_live, OUTPUT_COLUMNS].to_parquet(args.out / "history.parquet", index=False)
    txns.loc[is_live, OUTPUT_COLUMNS].to_parquet(args.out / "live.parquet", index=False)
    txns.loc[:, ["transaction_id", "card_id", "event_time", "is_fraud", "scenario"]].to_parquet(
        args.out / "ground_truth.parquet", index=False
    )
    blocklisted = sorted(pop.card_ids[list(reported_at)])
    (args.out / "blocklist.txt").write_text("\n".join(blocklisted) + "\n")

    print(f"cards: {args.cards:,}   days: {args.days} ({history_days} history + {args.live_days} live)")
    print(f"history: {int((~is_live).sum()):,} transactions   live: {int(is_live.sum()):,} transactions")
    print(f"blocklisted cards: {len(blocklisted)}")
    summary = (
        txns[txns["is_fraud"] == 1]
        .assign(split=np.where(is_live[txns["is_fraud"] == 1], "live", "history"))
        .groupby(["split", "scenario"]).size().unstack(0, fill_value=0)
    )
    print("\nfraudulent transactions by scenario:")
    print(summary.to_string())
    print(f"\nfraud rate: {txns['is_fraud'].mean():.2%}   wrote {args.out}/ in {time.perf_counter() - began:.1f}s")


if __name__ == "__main__":
    main()
