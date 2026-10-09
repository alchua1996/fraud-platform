# Fraud scoring platform

A practice codebase that works like the transaction-scoring service at a card issuer:
simulated card traffic, a real-time scoring API (FastAPI), online features in Redis, and a
decision log in Postgres. It runs end to end today with hand-written rules and a placeholder
model.

Treat it like a team you just joined. The service works, the code is yours now, and the
backlog in [TICKETS.md](TICKETS.md) is the work a real team would do next: automate testing
and deployment, train and ship a model, add monitoring and alerts, make it fast, swap
features and models without downtime, and add an LLM review lane.

## How it fits together

```
simulator/generate.py
 ├─ data/history.parquet ─► jobs/build_profiles.py ─► Redis: card profiles, blocklist    (nightly)
 │
 └─ data/live.parquet ──► scripts/replay.py ──► POST /score ──► Redis: velocity, devices (every request)
                                                    │
                                                    └─► Postgres: decision log ─► jobs/export_training_data.py ─► next model
```

- **The simulator** generates 30 days of card transactions with five fraud patterns mixed in.
  The first 23 days are "history" (the data warehouse); the last 7 are "live" traffic that
  gets replayed against the API. The answer key (`data/ground_truth.parquet`) is only for
  evaluation; the service never sees it.
- **The API** scores each transaction in two tiers: rules first, then a model, and returns
  allow, review or block.
- **Redis** holds what the service needs to know about a card in under a millisecond: recent
  activity (written on every request) and a 30-day profile (written by the nightly job).
- **Postgres** stores every decision with the exact features used, which doubles as training
  data for the next model.

## Quick start

You need Docker (Docker Desktop on Mac or Windows) and `make`.

```bash
cp .env.example .env    # then set POSTGRES_PASSWORD to anything made of letters and digits
make setup              # build and start everything, simulate data, load batch features
make replay             # score 2,000 live transactions and report what happened
```

Then open http://localhost:8000/docs, expand `POST /score`, click "Try it out" and send the
example transaction. If you set `API_KEY` in `.env`, put it in the `x-api-key` field.

`make replay` prints the decision mix, latency, and how the decisions compare with the answer
key. For example, 6,000 transactions at 100 requests/second on a 2-vCPU test machine:

```
sent 6,000 in 60.1s (100 req/s)
status:  200: 6,000
decision:  allow 99.2%   review 0.4%   block 0.4%
decided by:  model 5,977   rules 23
latency ms (client):  p50    5.6   p95    8.2   p99   16.0   max  226.7
latency ms (server):  p50    0.9   p95    1.4   p99    3.5   max   31.7

against the answer key (flagged = review or block):
  fraud transactions: 72   flagged: 48
  fraud caught (recall): 65.3%   flags that were fraud (precision): 97.9%
  legitimate transactions flagged: 1 (0.02% of legitimate)
```

Notice the gap between client and server latency: the service spends under a millisecond on
the actual scoring, but the caller waits about five. Ticket FRAUD-3 is about where the rest goes.

## How a transaction gets scored

`POST /score` in [app/main.py](app/main.py) runs these steps (orchestrated in
[app/scoring.py](app/scoring.py)):

1. **Validate.** FastAPI checks the request against `Transaction` in
   [app/schemas.py](app/schemas.py) and rejects anything malformed with a 422.
2. **Fetch features.** One pipelined Redis call reads the card's last 24 hours, its profile,
   the blocklist and the device history ([app/features.py](app/features.py)). That's one
   network round trip, however many keys.
3. **Tier 0: rules** ([app/rules.py](app/rules.py)). Hard rules (blocklisted card, extreme
   velocity) block immediately. Soft rules (card-testing pattern, amount spike, shared device
   and so on) are returned as reasons and later become training labels.
4. **Tier 1: model** ([app/models.py](app/models.py)). A hand-weighted logistic function for
   now. Its probability is compared with `REVIEW_THRESHOLD` and `BLOCK_THRESHOLD`.
5. **Record.** A second Redis round trip adds the transaction to the card's and device's
   recent history, so the next request sees it.
6. **Log.** After the response is sent, a background task writes the decision to Postgres
   ([app/db.py](app/db.py)).

Features are computed from the transaction's `event_time`, not the server clock. That's what
lets simulated history replay deterministically, and it's how real systems handle events that
arrive late.

Set `SCORING_MODE=rules_only` in `.env` (then `make restart`) to run without a model: any soft
rule hit goes to review. That's both the "before ML" baseline and a sensible fallback.

### Redis keys

| Key | Type | Written by | Holds |
|---|---|---|---|
| `card:{id}:txns` | sorted set | API, every request | the card's last 24h, scored by event time |
| `device:{id}:cards` | sorted set | API, every request | cards seen on the device, with last-seen time |
| `card:{id}:devices` | set | nightly job and API | devices the card has used |
| `card:{id}:profile` | hash | nightly job | 30-day average amount, count, home country |
| `blocklist:cards` | set | nightly job | cards reported stolen |

Look at them yourself with `make redis-cli`, e.g. `HGETALL card:c_000001:profile` or
`ZRANGE card:c_000001:txns 0 -1 WITHSCORES`.

## Baseline to beat

Both modes on the same window (the first 6,000 live transactions, 72 of them fraudulent),
100 requests/second, measured on a 2-vCPU test machine:

| Mode | Fraud caught | Flags that were fraud | Legitimate flagged | Server p50 / p99 | Client p50 / p99 |
|---|---|---|---|---|---|
| `rules_only` | 72.2% | 54.7% | 0.73% | 0.8 / 3.0 ms | 5.4 / 13.7 ms |
| `rules_then_model` (heuristic) | 65.3% | 97.9% | 0.02% | 0.9 / 3.5 ms | 5.6 / 16.0 ms |

The rules catch more fraud but flag 30 times more good customers; the heuristic is precise
but misses more. Your trained model (FRAUD-1) should be compared on the same window:
`make reset && make replay N=6000`.

## Commands

Run `make help` for the full list. The ones you'll use most:

| Command | What it does |
|---|---|
| `make setup` | first run: start services, generate data, load batch features |
| `make replay N=5000 RPS=200` | send live transactions to the API and report |
| `make reset` | empty Redis and the decision log, reload batch features |
| `make restart` | recreate the API container after editing `.env` |
| `make logs` | follow the API's logs |
| `make export` | turn the decision log into training data |
| `make test` | run the tests |
| `make redis-cli` / `make psql` | open a Redis or Postgres shell |

Replaying the same transactions twice without `make reset` double-counts them in Redis, and
every velocity feature comes out wrong.

## Running without Docker

Handy for quick iteration with a debugger:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
redis-server --daemonize yes     # or: docker run -d -p 6379:6379 redis:7-alpine
python -m simulator.generate
python -m jobs.build_profiles
uvicorn app.main:app --reload    # in another terminal
python -m scripts.replay --truth data/ground_truth.parquet
```

Without `DATABASE_URL` set, the decision log goes to SQLite at `data/decisions.db`.

## Deploying to a $24/month server (AWS Lightsail)

The whole stack fits on one small Linux server. On AWS Lightsail that's the $24/month plan
(4 GB RAM, 2 vCPUs, 80 GB SSD). Any provider's Ubuntu server with 4 GB of RAM works the same way.

1. **Put the code on GitHub** (a private repo is fine), from the project folder on your laptop:

   ```bash
   git init && git add . && git commit -m "Initial scaffold"
   git branch -M main
   git remote add origin https://github.com/<you>/fraud-platform.git   # create the empty repo on github.com first
   git push -u origin main
   ```

   The tests run on every push (`.github/workflows/ci.yml`). For a private repo, the server
   needs read access to clone it: run `ssh-keygen` on the server and add the public key under
   the repo's Settings → Deploy keys. Automating the deploy itself is ticket FRAUD-0.
2. **Set a billing alarm first** (AWS Billing → Budgets), for example at $30/month.
3. In Lightsail, **create an instance**: Linux/Unix → OS Only → Ubuntu 24.04 LTS → the
   $24/month plan.
4. Under the instance's **Networking** tab, keep SSH (port 22) and add a rule for TCP port
   8000, restricted to your own IP address. Don't open 5432 (Postgres) or 6379 (Redis).
5. **Connect with SSH** (the browser-based button works) and run:

   ```bash
   git clone https://github.com/<you>/fraud-platform.git
   cd fraud-platform
   bash deploy/bootstrap.sh   # installs Docker; log out and back in when it tells you to
   bash deploy/bootstrap.sh   # creates .env with random secrets and starts everything
   ```

6. From your laptop: `curl http://<server-ip>:8000/health`, or open
   `http://<server-ip>:8000/docs` with the API key printed at the end of the script.

To ship changes by hand: push to GitHub, then on the server `git pull && make up`. Once
FRAUD-0 is done, merging to `main` does this for you.

To edit code on the server directly, VS Code's Remote-SSH extension opens the server's folder
on your laptop, with the terminal running on the server. Commit and push from there too, so
GitHub stays the single source of truth.

A few things to know about the cost:

- Lightsail charges while the instance exists, even when it's stopped. To pause for a while,
  take a snapshot, delete the instance, and recreate it from the snapshot later.
- The CPUs are burstable: fine for normal use, but a long load test uses up the burst
  allowance and the server slows down. Keep load tests short, and note in your results if a
  run got throttled.

### Security basics

- Always set `API_KEY` on a server reachable from the internet (the bootstrap script does).
- Redis and Postgres have no published ports; only the API container can reach them.
- Restrict port 8000 to your IP in the firewall.
- `.env` holds secrets and is in `.gitignore`. Keep it that way.

## Experiments: break it on purpose

Knowing these tools in practice mostly means knowing how they fail. Try these while a replay
runs (`make replay N=20000 RPS=100` in one terminal), and write down what happened and what
you changed. That's the material interviewers ask about.

- **Redis goes away:** `docker compose stop redis`, then `docker compose start redis`. Then
  `docker compose pause redis` (Redis hangs instead of refusing connections) and compare.
- **Redis fills up:** in `make redis-cli`, run `INFO memory` to see `used_memory_human`, then
  `CONFIG SET maxmemory` to slightly more than that (e.g. `8mb`). Watch what happens as it
  fills. Why errors rather than silently missing data? (Look at `--maxmemory-policy` in
  `docker-compose.yml`.) `CONFIG SET maxmemory 512mb` puts it back.
- **Deploy during traffic:** `make restart` mid-replay. How many requests failed?
- **Overload:** `make replay N=20000 RPS=300`. Watch client latency pull away from server
  latency, and figure out why.
- **Replay twice:** run the same replay again without `make reset` and look at the velocity
  features in the decision log.

## Project layout

```
app/
  main.py          FastAPI app: endpoints, startup, API key check
  scoring.py       features -> rules -> model -> decision
  features.py      Redis reads and writes, feature definitions
  rules.py         tier 0 rules and thresholds
  models.py        tier 1 model interface and the placeholder heuristic
  db.py            decision log table
  schemas.py       request and response shapes
  config.py        settings from environment variables
simulator/
  generate.py      synthetic transactions, fraud scenarios, answer key
jobs/
  build_profiles.py        nightly batch features -> Redis
  export_training_data.py  decision log -> training set with rule-based labels
scripts/
  replay.py        load generator and evaluator
  reset.py         wipe Redis and the decision log
tests/             unit and API tests (fakeredis + SQLite, no services needed)
deploy/
  bootstrap.sh     one-time setup for a fresh Ubuntu server
```

## Troubleshooting

- **`set POSTGRES_PASSWORD in .env`**: you haven't created `.env` yet (`cp .env.example .env`).
- **`PermissionError` writing to `data/` on Linux**: the container runs as user 1000. Fix with
  `sudo chown -R 1000:1000 data models`.
- **Port 8000 is already in use**: change the left side of `"8000:8000"` in
  `docker-compose.yml`, e.g. `"8001:8000"`.
- **Velocity numbers look too high**: you replayed transactions that were already in Redis.
  Run `make reset`.
- **`NotImplementedError: Can't load ...`**: you pointed `MODEL_PATH` at a trained model before
  finishing FRAUD-1.
