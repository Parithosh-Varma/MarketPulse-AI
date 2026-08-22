"""Typed schema for price/sentiment divergence (Phase 6)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from src.models.aggregation import _parse_ts, _parse_window

CLASS_BEARISH_DIVERGENCE = "bearish_divergence"
CLASS_BULLISH_DIVERGENCE = "bullish_divergence"
CLASS_ALIGNED_BULLISH = "aligned_bullish"
CLASS_ALIGNED_BEARISH = "aligned_bearish"
CLASS_NEUTRAL = "neutral"

CLASSIFICATIONS = (
    CLASS_BEARISH_DIVERGENCE,
    CLASS_BULLISH_DIVERGENCE,
    CLASS_ALIGNED_BULLISH,
    CLASS_ALIGNED_BEARISH,
    CLASS_NEUTRAL,
)


@dataclass(frozen=True)
class DivergenceObservation:
    """Price move vs sentiment move over one look-back window.

    Sign convention:
        divergence_score > 0  sentiment outruns price   (bullish divergence
                              when price fell while sentiment rose)
        divergence_score < 0  price outruns sentiment   (bearish divergence
                              when price rose while sentiment fell)

    ``price_return`` is a simple return (0.031 == +3.1%).
    """

    symbol: str
    as_of: datetime
    window: timedelta
    price_return: float
    sentiment_change: float
    divergence_score: float
    classification: str
    confidence: float
    n_observations: int
    volatility_context: Optional[float] = None
    threshold_used: float = 0.30
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", _parse_ts(self.as_of))
        object.__setattr__(self, "window", _parse_window(self.window))
        if self.window.total_seconds() <= 0:
            raise ValueError("window must be positive")

        for name in ("price_return",):
            value = float(getattr(self, name))
            object.__setattr__(self, name, value)
        if self.n_observations <= 0:
            raise ValueError("n_observations must be > 0")
        for name in ("divergence_score", "confidence"):
            value = getattr(self, name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1], got {value}")
        if self.volatility_context is not None and self.volatility_context < 0:
            raise ValueError("volatility_context must be >= 0")
        if self.classification not in CLASSIFICATIONS:
            raise ValueError(
                f"classification must be one of {CLASSIFICATIONS}, "
                f"got {self.classification!r}"
            )

    @property
    def is_divergence(self) -> bool:
        return self.classification in (
            CLASS_BEARISH_DIVERGENCE,
            CLASS_BULLISH_DIVERGENCE,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["as_of"] = self.as_of.isoformat()
        data["window"] = self.window.total_seconds()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DivergenceObservation":
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, payload: str) -> "DivergenceObservation":
        return cls.from_dict(json.loads(payload))
