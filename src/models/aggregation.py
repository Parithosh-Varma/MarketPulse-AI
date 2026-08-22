"""Typed schemas for aggregated sentiment (Phase 3)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    else:
        raise TypeError(
            f"timestamp must be datetime or ISO-8601 str, got {type(value).__name__}"
        )
    return ts if ts.tzinfo else ts.replace(tzinfo=None)


def _parse_window(value: Any) -> timedelta:
    if isinstance(value, timedelta):
        return value
    if isinstance(value, (int, float)):
        return timedelta(seconds=value)
    raise TypeError(f"window must be timedelta, got {type(value).__name__}")


def _check_range(name: str, value: Optional[float], low: float, high: float) -> None:
    if value is None:
        return
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}")


@dataclass(frozen=True)
class AggregatedSentiment:
    """Summary of sentiment over one asset/window (or any grouping).

    Class proportions are the share of observations whose score is
    positive / negative / exactly zero — consistent across engines that do
    or do not emit calibrated probabilities.
    """

    window_start: datetime
    window_end: datetime
    window: timedelta
    n_observations: int
    mean_score: float
    weighted_score: Optional[float] = None
    median_score: Optional[float] = None
    mean_confidence: Optional[float] = None
    share_positive: Optional[float] = None
    share_negative: Optional[float] = None
    share_neutral: Optional[float] = None
    ticker: Optional[str] = None
    sector: Optional[str] = None
    market: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "window_start", _parse_ts(self.window_start))
        object.__setattr__(self, "window_end", _parse_ts(self.window_end))
        object.__setattr__(self, "window", _parse_window(self.window))

        # Zero-length windows are valid: every observation may share one
        # timestamp (point-in-time aggregation).
        if self.window_end < self.window_start:
            raise ValueError(
                f"window_end ({self.window_end}) cannot be before "
                f"window_start ({self.window_start})"
            )
        if self.n_observations <= 0:
            raise ValueError(f"n_observations must be > 0, got {self.n_observations}")

        for name in ("mean_score", "weighted_score", "median_score"):
            _check_range(name, getattr(self, name), -1.0, 1.0)
        for name in (
            "mean_confidence",
            "share_positive",
            "share_negative",
            "share_neutral",
        ):
            _check_range(name, getattr(self, name), 0.0, 1.0)

        shares = [self.share_positive, self.share_negative, self.share_neutral]
        if all(s is not None for s in shares):
            total = sum(shares)  # type: ignore[arg-type]
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"class shares must sum to 1.0, got {total}")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["window_start"] = (
            self.window_start.isoformat() if self.window_start else None
        )
        data["window_end"] = self.window_end.isoformat() if self.window_end else None
        data["window"] = self.window.total_seconds()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregatedSentiment":
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, payload: str) -> "AggregatedSentiment":
        return cls.from_dict(json.loads(payload))
