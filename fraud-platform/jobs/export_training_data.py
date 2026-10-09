"""Export the decision log as a training set for the next model (ticket FRAUD-1).

Run:
    python -m jobs.export_training_data
    python -m jobs.export_training_data --out data/training/decisions.parquet

There are no fraud labels yet, so the target is "should this transaction have been flagged?",
and rules provide the answer:

    label = 1 if any rule fired when the transaction was scored (logged in rule_hits), or
            if an offline-only rule fires. Offline rules can look at more data than the
            hot path can afford in a few milliseconds:
              device_ring      the device was used by 5+ distinct cards on the same day
              country_hopping  the card was used in 3+ countries on the same day

Training a model on this teaches it to approximate the full rule set, including the offline
part, cheaply at scoring time. The simulator's ground truth is deliberately not used here;
it's only for evaluating how well that works (scripts/replay.py --truth).

Features come straight from the log, exactly as the service computed them at scoring time,
so there's no way for training features to drift from serving features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from app.config import get_settings
from app.db import decisions, make_engine
from app.features import FEATURE_NAMES

DEVICE_RING_CARDS = 5
COUNTRY_HOPPING_COUNTRIES = 3


def offline_labels(log: pd.DataFrame) -> pd.DataFrame:
    day = log["event_time"].dt.floor("D")
    cards_per_device_day = (
        log[log["device_id"].notna()].groupby([log["device_id"], day])["card_id"].transform("nunique")
    )
    countries_per_card_day = log.groupby([log["card_id"], day])["country"].transform("nunique")
    return pd.DataFrame({
        "device_ring": cards_per_device_day.reindex(log.index).fillna(0) >= DEVICE_RING_CARDS,
        "country_hopping": countries_per_card_day >= COUNTRY_HOPPING_COUNTRIES,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/training/decisions.parquet"))
    args = parser.parse_args()

    engine = make_engine(get_settings().database_url)
    with engine.connect() as conn:
        log = pd.DataFrame(conn.execute(select(decisions).order_by(decisions.c.event_time)).mappings().all())
    if log.empty:
        raise SystemExit("The decision log is empty. Replay some transactions first (make replay).")

    log["event_time"] = pd.to_datetime(log["event_time"], utc=True)
    for column in ("features", "rule_hits"):  # some drivers hand JSON back as text
        log[column] = log[column].map(lambda v: json.loads(v) if isinstance(v, str) else v)

    features = pd.DataFrame(list(log["features"]), index=log.index)[FEATURE_NAMES]
    offline = offline_labels(log)
    online_hit = log["rule_hits"].map(len) > 0

    out = pd.concat([log[["transaction_id", "event_time", "model_version"]], features], axis=1)
    out["rule_hits"] = log["rule_hits"].map(lambda hits: ",".join(hits))
    out["label"] = (online_hit | offline.any(axis=1)).astype(int)
    out["label_source"] = "none"
    out.loc[offline["country_hopping"], "label_source"] = "offline:country_hopping"
    out.loc[offline["device_ring"], "label_source"] = "offline:device_ring"
    out.loc[online_hit, "label_source"] = "rules"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"wrote {len(out):,} rows to {args.out}")
    print(f"label rate: {out['label'].mean():.2%}")
    print(out["label_source"].value_counts().to_string())


if __name__ == "__main__":
    main()
