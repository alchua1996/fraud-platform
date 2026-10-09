"""Replay simulated live transactions against the scoring API, in event-time order.

Run:
    python -m scripts.replay --limit 2000 --rps 100
    python -m scripts.replay --url http://api:8000 --limit 10000 --rps 300 --truth data/ground_truth.parquet

Prints the decision mix, errors and latency percentiles. Latency is measured from the moment
each request was *scheduled* to go out, so if the service falls behind and requests queue up,
that waiting counts too. (Measuring only from when a request actually left would hide it; this
mistake is known as "coordinated omission".)

With --truth it also compares decisions with the simulator's answer key. That's evaluation
only: the API never sees the answer key.

Requests are sent concurrently, so two charges on the same card a few milliseconds apart can be
scored out of order. Use --concurrency 1 when exact ordering matters.

Replaying the same transactions twice double-counts them in Redis. Reset first (make reset).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd


def to_payloads(frame: pd.DataFrame) -> list[dict]:
    payloads = []
    for row in frame.itertuples(index=False):
        payloads.append({
            "transaction_id": row.transaction_id,
            "card_id": row.card_id,
            "merchant_id": row.merchant_id,
            "merchant_category": row.merchant_category,
            "amount": float(row.amount),
            "country": row.country,
            "channel": row.channel,
            "device_id": row.device_id if isinstance(row.device_id, str) else None,
            "event_time": row.event_time.isoformat(),
        })
    return payloads


async def replay(payloads, url, rps, concurrency, api_key) -> list[dict]:
    results: list[dict] = []
    gate = asyncio.Semaphore(concurrency)
    headers = {"X-API-Key": api_key} if api_key else {}
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(base_url=url, headers=headers, timeout=10.0, limits=limits) as client:

        async def send(payload: dict, scheduled: float) -> None:
            async with gate:
                result = {"transaction_id": payload["transaction_id"]}
                try:
                    response = await client.post("/score", json=payload)
                    result["status"] = str(response.status_code)
                    if response.status_code == 200:
                        body = response.json()
                        result.update(
                            decision=body["decision"],
                            decided_by=body["decided_by"],
                            score=body["score"],
                            server_ms=body["timings_ms"]["total"],
                        )
                except httpx.HTTPError as exc:
                    result["status"] = type(exc).__name__
                result["latency_ms"] = (time.perf_counter() - scheduled) * 1000
                results.append(result)

        start = time.perf_counter()
        tasks = []
        for i, payload in enumerate(payloads):
            scheduled = start + i / rps
            delay = scheduled - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(send(payload, scheduled)))
        await asyncio.gather(*tasks)
    return results


def percentiles(values) -> str:
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return f"p50 {p50:6.1f}   p95 {p95:6.1f}   p99 {p99:6.1f}   max {np.max(values):6.1f}"


def report(results: pd.DataFrame, elapsed: float, truth_path: Path | None) -> None:
    ok = results[results["status"] == "200"]
    print(f"\nsent {len(results):,} in {elapsed:.1f}s ({len(results) / elapsed:,.0f} req/s)")
    print("status:  " + "   ".join(f"{k}: {v:,}" for k, v in results["status"].value_counts().items()))
    if ok.empty:
        return

    mix = ok["decision"].value_counts(normalize=True)
    print("decision:  " + "   ".join(f"{d} {mix.get(d, 0):.1%}" for d in ("allow", "review", "block")))
    print("decided by:  " + "   ".join(f"{k} {v:,}" for k, v in ok["decided_by"].value_counts().items()))
    print(f"latency ms (client):  {percentiles(results['latency_ms'])}")
    print(f"latency ms (server):  {percentiles(ok['server_ms'])}")

    if truth_path is None:
        return
    truth = pd.read_parquet(truth_path, columns=["transaction_id", "is_fraud", "scenario"])
    scored = ok.merge(truth, on="transaction_id", how="inner")
    flagged = scored["decision"] != "allow"
    fraud = scored["is_fraud"] == 1
    n_fraud, n_flagged = int(fraud.sum()), int(flagged.sum())
    caught = int((flagged & fraud).sum())
    print("\nagainst the answer key (flagged = review or block):")
    print(f"  fraud transactions: {n_fraud:,}   flagged: {n_flagged:,}")
    print(f"  fraud caught (recall): {caught / max(n_fraud, 1):.1%}   "
          f"flags that were fraud (precision): {caught / max(n_flagged, 1):.1%}")
    print(f"  legitimate transactions flagged: {int((flagged & ~fraud).sum()):,} "
          f"({(flagged & ~fraud).sum() / max((~fraud).sum(), 1):.2%} of legitimate)")
    by_scenario = scored[fraud].assign(caught=flagged[fraud]).groupby("scenario")["caught"].agg(["size", "mean"])
    if not by_scenario.empty:
        print("  recall by scenario:")
        for scenario, row in by_scenario.iterrows():
            print(f"    {scenario:<18} {row['mean']:6.1%} of {int(row['size'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("SCORING_URL", "http://localhost:8000"))
    parser.add_argument("--data", type=Path, default=Path("data/live.parquet"))
    parser.add_argument("--offset", type=int, default=0, help="skip this many transactions first")
    parser.add_argument("--limit", type=int, default=2_000)
    parser.add_argument("--rps", type=float, default=100.0, help="target requests per second")
    parser.add_argument("--concurrency", type=int, default=50, help="max requests in flight")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY") or None)
    parser.add_argument("--truth", type=Path, help="answer key to evaluate against")
    parser.add_argument("--save", type=Path, help="write per-transaction results to this CSV")
    args = parser.parse_args()

    frame = pd.read_parquet(args.data).iloc[args.offset:args.offset + args.limit]
    payloads = to_payloads(frame)
    print(f"replaying {len(payloads):,} transactions to {args.url} at {args.rps:g} req/s "
          f"({frame['event_time'].min()} to {frame['event_time'].max()})")

    began = time.perf_counter()
    results = pd.DataFrame(asyncio.run(replay(payloads, args.url, args.rps, args.concurrency, args.api_key)))
    report(results, time.perf_counter() - began, args.truth)
    if args.save:
        results.to_csv(args.save, index=False)
        print(f"\nsaved per-transaction results to {args.save}")


if __name__ == "__main__":
    main()
