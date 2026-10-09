"""Start over: empty Redis and the decision log.

Run:
    python -m scripts.reset
then reload the batch features with python -m jobs.build_profiles (make reset does both).

Needed before replaying the same transactions again; otherwise Redis already holds them and
every velocity count comes out too high.
"""

from redis import Redis
from sqlalchemy import delete

from app.config import get_settings
from app.db import decisions, init_db, make_engine


def main() -> None:
    settings = get_settings()
    Redis.from_url(settings.redis_url).flushdb()
    engine = make_engine(settings.database_url)
    init_db(engine)
    with engine.begin() as conn:
        deleted = conn.execute(delete(decisions)).rowcount
    print(f"flushed Redis and deleted {deleted:,} decisions. Reload batch features: python -m jobs.build_profiles")


if __name__ == "__main__":
    main()
