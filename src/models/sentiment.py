"""Typed schemas for individual sentiment observations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_PROB_TOLERANCE = 1e-6


def _parse_timestamp(value: Any) -> datetime:
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
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _check_range(name: str, value: Optional[float], low: float, high: float) -> None:
    if value is None:
        return
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}")


@dataclass(frozen=True)
class SentimentObservation:
    """A single scored text item (headline, post, event) for one asset.

    Score convention: -1.0 (maximally negative) .. +1.0 (maximally positive).
    Class probabilities are optional; when all three are provided they must
    sum to 1.0 within tolerance.
    """

    timestamp: datetime
    source: str
    score: float
    ticker: Optional[str] = None
    author_id: Optional[str] = None
    text: Optional[str] = None
    prob_positive: Optional[float] = None
    prob_negative: Optional[float] = None
    prob_neutral: Optional[float] = None
    model_name: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _parse_timestamp(self.timestamp))

        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")

        try:
            score = float(self.score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"score must be a number, got {self.score!r}") from exc
        _check_range("score", score, -1.0, 1.0)
        object.__setattr__(self, "score", score)

        for name in ("prob_positive", "prob_negative", "prob_neutral"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                object.__setattr__(self, name, value)
            _check_range(name, value, 0.0, 1.0)

        probs = [
            self.prob_positive,
            self.prob_negative,
            self.prob_neutral,
        ]
        if all(p is not None for p in probs):
            total = sum(probs)  # type: ignore[arg-type]
            if abs(total - 1.0) > _PROB_TOLERANCE:
                raise ValueError(
                    f"probabilities must sum to 1.0 (got {total:.9f}); "
                    "provide all three or none"
                )

        _check_range("confidence", self.confidence, 0.0, 1.0)

    @property
    def timestamp_iso(self) -> str:
        return self.timestamp.isoformat()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp_iso
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentimentObservation":
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    def to_json(self, **json_kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **json_kwargs)

    @classmethod
    def from_json(cls, payload: str) -> "SentimentObservation":
        return cls.from_dict(json.loads(payload))
