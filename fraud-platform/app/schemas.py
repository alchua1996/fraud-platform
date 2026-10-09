"""Request and response shapes for the API. FastAPI validates every request against these."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Transaction(BaseModel):
    transaction_id: str
    card_id: str
    merchant_id: str
    merchant_category: str
    amount: float = Field(gt=0)
    country: str = Field(min_length=2, max_length=2, description="ISO country code of the merchant or IP")
    channel: Literal["card_present", "online"]
    device_id: str | None = None
    # When the transaction happened. Features are computed relative to this, not the server
    # clock, which is what makes replaying simulated history deterministic.
    event_time: datetime

    @field_validator("event_time")
    @classmethod
    def assume_utc(cls, value: datetime) -> datetime:
        # A naive datetime would be interpreted in the server's local time zone. Never do that.
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "t_demo_0001",
                "card_id": "c_000042",
                "merchant_id": "m_07_003",
                "merchant_category": "electronics",
                "amount": 899.99,
                "country": "US",
                "channel": "online",
                "device_id": "d_00000000ab",
                "event_time": "2026-09-24T03:15:00Z",
            }
        }
    )


class ScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    transaction_id: str
    decision: Literal["allow", "review", "block"]
    decided_by: Literal["rules", "model"]
    score: float | None = Field(description="Model probability; null when the rules decided")
    reasons: list[str]
    model_version: str
    timings_ms: dict[str, float]
