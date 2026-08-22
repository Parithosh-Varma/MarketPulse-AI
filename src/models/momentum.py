"""Typed schema for sentiment momentum (Phase 4)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from src.models.aggregation import _parse_ts, _parse_window

MomentumClass = str  # one of CLASS_IMPROVING / DETERIORATING / STABLE

CLASS_IMPROVING = "improving"
CLASS_DETERIORATING = "deteriorating"
CLASS_STABLE = "stable"
MOMENTUM_CLASSES = (CLASS_IMPROVING, CLASS_DETERIORATING, CLASS_STABLE)


@dataclass(frozen=True)
class SentimentMomentum:
    """Change in aggregate sentiment between two consecutive windows.

    momentum = current_score - previous_score
    rate_per_hour normalizes by elapsed wall-clock time between window
    midpoints so uneven sampling does not distort comparisons.
    """

    ticker: Optional[str]
    previous_window_end: datetime
    current_window_end: datetime
    window: timedelta
    previous_score: float
    current_score: float
    classification: MomentumClass
    threshold: float
    accelerating: Optional[bool] = None
    reversal: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "previous_window_end", _parse_ts(self.previous_window_end))
        object.__setattr__(self, "current_window_end", _parse_ts(self.current_window_end))
        object.__setattr__(self, "window", _parse_window(self.window))

        if self.current_window_end < self.previous_window_end:
            raise ValueError("current_window_end cannot precede previous_window_end")
        for name in ("previous_score", "current_score"):
            value = getattr(self, name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1], got {value}")
        if self.threshold < 0:
            raise ValueError("threshold must be >= 0")
        if self.classification not in MOMENTUM_CLASSES:
            raise ValueError(
                f"classification must be one of {MOMENTUM_CLASSES}, "
                f"got {self.classification!r}"
            )

    @property
    def momentum(self) -> float:
        return self.current_score - self.previous_score

    @property
    def elapsed(self) -> timedelta:
        return self.current_window_end - self.previous_window_end

    @property
    def rate_per_hour(self) -> float:
        hours = self.elapsed.total_seconds() / 3600.0
        if hours <= 0:
            raise ValueError("non-positive elapsed time between windows")
        return self.momentum / hours

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["previous_window_end"] = self.previous_window_end.isoformat()
        data["current_window_end"] = self.current_window_end.isoformat()
        data["window"] = self.window.total_seconds()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentimentMomentum":
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, payload: str) -> "SentimentMomentum":
        return cls.from_dict(json.loads(payload))
