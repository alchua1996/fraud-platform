"""Nightly batch job: compute 30-day card profiles from the warehouse and load them into Redis.

Run:
    python -m jobs.build_profiles
    python -m jobs.build_profiles --history data/history.parquet --window-days 30

For every card active in the window it writes:
    card:{id}:profile   hash   avg_amount_30d, txn_count_30d, n_countries_30d, home_country, as_of
    card:{id}:devices   set    devices the card used (merged with what the service has seen live)
and replaces blocklist:cards with the cards in data/blocklist.txt.

Profiles are overwritten in place, key by key. While the job runs, some cards have tonight's
profile and some still have yesterday's; each card's own profile is never half-written,
because it's written with a single command. Replacing this with versioned keys and an atomic
switch is ticket FRAUD-4.

The blocklist, by contrast, is already swapped atomically: it's built under a temporary key and
renamed over the live one, so there's never a moment when the blocklist is empty.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from redis import Redis

from app.config import get_settings
from app.features import BLOCKLIST_KEY, DEVICE_HISTORY_TTL_S, card_devices_key, card_profile_key


def compute_profiles(history: pd.DataFrame, as_of: pd.Timestamp, window_days: int) -> pd.DataFrame:
    window = history[(history["event_time"] > as_of - pd.Timedelta(days=window_days))
                     & (history["event_time"] <= as_of)]
    by_card = window.groupby("card_id")
    profiles = pd.DataFrame({
        "avg_amount_30d": by_card["amount"].mean().round(2),
        "txn_count_30d": by_card.size(),
        "n_countries_30d": by_card["country"].nunique(),
    })
    # Home country: where the card transacts most often.
    country_counts = window.groupby(["card_id", "country"]).size().reset_index(name="n")
    home = country_counts.sort_values("n", ascending=False).drop_duplicates("card_id")
    profiles["home_country"] = home.set_index("card_id")["country"]
    profiles["as_of"] = as_of.isoformat()
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", type=Path, default=Path("data/history.parquet"))
    parser.add_argument("--blocklist", type=Path, default=Path("data/blocklist.txt"))
    parser.add_argument("--as-of", help="profile cutoff (default: the last timestamp in the history)")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1_000, help="cards per Redis pipeline")
    args = parser.parse_args()

    began = time.perf_counter()
    r = Redis.from_url(get_settings().redis_url, decode_responses=True)
    history = pd.read_parquet(args.history)
    as_of = pd.Timestamp(args.as_of, tz="UTC") if args.as_of else history["event_time"].max()

    profiles = compute_profiles(history, as_of, args.window_days)
    devices = (
        history[history["device_id"].notna() & (history["event_time"] <= as_of)]
        .groupby("card_id")["device_id"].unique()
    )

    # Write in batches: one pipeline (one network round trip) per batch of cards, instead of
    # one round trip per command.
    records = list(profiles.iterrows())
    for start in range(0, len(records), args.batch_size):
        pipe = r.pipeline(transaction=False)
        for card_id, row in records[start:start + args.batch_size]:
            pipe.hset(card_profile_key(card_id), mapping={k: str(v) for k, v in row.items()})
            if card_id in devices.index:
                pipe.sadd(card_devices_key(card_id), *devices[card_id])
                pipe.expire(card_devices_key(card_id), DEVICE_HISTORY_TTL_S)
        pipe.execute()

    blocklist = [line.strip() for line in args.blocklist.read_text().splitlines() if line.strip()] \
        if args.blocklist.exists() else []
    staging = f"{BLOCKLIST_KEY}:staging"
    pipe = r.pipeline(transaction=True)
    pipe.delete(staging)
    if blocklist:
        pipe.sadd(staging, *blocklist)
        pipe.rename(staging, BLOCKLIST_KEY)
    else:
        pipe.delete(BLOCKLIST_KEY)
    pipe.execute()

    r.hset("profiles:meta", mapping={
        "as_of": as_of.isoformat(),
        "cards": str(len(profiles)),
        "built_at": pd.Timestamp.now(tz="UTC").isoformat(),
    })
    elapsed = time.perf_counter() - began
    print(f"profiles as of {as_of}: {len(profiles):,} cards, {int(devices.map(len).sum()):,} card-device pairs")
    print(f"blocklist: {len(blocklist):,} cards   took {elapsed:.1f}s ({len(profiles) / elapsed:,.0f} cards/s)")


if __name__ == "__main__":
    main()
