from app.features import (
    FEATURE_NAMES,
    card_devices_key,
    card_profile_key,
    fetch_features,
    record_transaction,
    to_vector,
)

from .conftest import make_txn


def test_velocity_counts_only_look_back_in_event_time(redis_client):
    for i, minutes in enumerate([0, 30, 55, 58]):
        record_transaction(redis_client, make_txn(i, minutes=minutes))

    f = fetch_features(redis_client, make_txn(99, minutes=60))

    assert f["card_txn_count_10m"] == 2      # minutes 55 and 58
    assert f["card_txn_count_1h"] == 4
    assert f["card_txn_count_24h"] == 4


def test_old_transactions_fall_out_of_the_24h_window(redis_client):
    record_transaction(redis_client, make_txn(1, minutes=0))
    record_transaction(redis_client, make_txn(2, minutes=25 * 60))

    f = fetch_features(redis_client, make_txn(3, minutes=25 * 60 + 1))

    assert f["card_txn_count_24h"] == 1


def test_device_shared_across_cards_counts_other_cards_only(redis_client):
    for i in range(4):
        record_transaction(
            redis_client, make_txn(i, minutes=i, card_id=f"c_{i}", channel="online", device_id="d_ring")
        )

    f = fetch_features(redis_client, make_txn(10, minutes=5, card_id="c_0", channel="online", device_id="d_ring"))

    assert f["device_other_cards_24h"] == 3


def test_new_device_flag_uses_the_cards_known_devices(redis_client):
    redis_client.sadd(card_devices_key("c_000001"), "d_known")

    known = fetch_features(redis_client, make_txn(1, channel="online", device_id="d_known"))
    new = fetch_features(redis_client, make_txn(2, channel="online", device_id="d_new"))

    assert known["is_new_device"] == 0
    assert new["is_new_device"] == 1


def test_profile_features_and_missing_profile(redis_client):
    no_profile = fetch_features(redis_client, make_txn(1, amount=500.0))
    assert no_profile["has_profile"] == 0
    assert no_profile["amount_to_avg_30d"] == 0

    redis_client.hset(
        card_profile_key("c_000001"),
        mapping={"avg_amount_30d": "49.0", "txn_count_30d": "40", "home_country": "US"},
    )
    with_profile = fetch_features(redis_client, make_txn(2, amount=500.0, country="GB"))
    assert with_profile["has_profile"] == 1
    assert with_profile["amount_to_avg_30d"] == 10.0
    assert with_profile["country_mismatch"] == 1


def test_vector_follows_feature_names(redis_client):
    f = fetch_features(redis_client, make_txn())
    assert len(to_vector(f)) == len(FEATURE_NAMES)
