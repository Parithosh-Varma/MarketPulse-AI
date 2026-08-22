"""Timezone helpers — UTC internally, Asia/Kolkata for presentation (§17)."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def ensure_utc(ts: datetime) -> datetime:
    """Normalize any datetime to timezone-aware UTC (naive assumed UTC)."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def to_ist(ts: datetime) -> datetime:
    """Convert a timestamp to Asia/Kolkata for display only."""
    return ensure_utc(ts).astimezone(IST)


def ist_market_date(ts: datetime) -> str:
    """ISO date of the Indian trading day a timestamp falls on."""
    return to_ist(ts).date().isoformat()
