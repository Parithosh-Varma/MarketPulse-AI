"""Typed schemas for market regime & stress (Phase 7)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, Dict, Optional

from src.models.aggregation import _parse_ts

REGIME_BULLISH = "bullish"
REGIME_BEARISH = "bearish"
REGIME_NEUTRAL = "neutral"
REGIME_HIGH_VOL = "high_volatility"
REGIME_RISK_ON = "risk_on"
REGIME_RISK_OFF = "risk_off"
REGIME_TRANSITION = "transition"

REGIMES = (
    REGIME_BULLISH,
    REGIME_BEARISH,
    REGIME_NEUTRAL,
    REGIME_HIGH_VOL,
    REGIME_RISK_ON,
    REGIME_RISK_OFF,
    REGIME_TRANSITION,
)

STRESS_LOW = "low_stress"
STRESS_MEDIUM = "medium_stress"
STRESS_HIGH = "high_stress"


@dataclass(frozen=True)
class MarketStress:
    """Point-in-time market stress reading.

    ``level`` is on the native scale of ``source`` (e.g. FinSentinel GMSI
    is an expanding-z composite roughly centered on 0); ``percentile`` in
    [0, 1] when a reference distribution is available.
    """

    as_of: datetime
    source: str
    level: float
    classification: str
    percentile: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", _parse_ts(self.as_of))
        if self.classification not in (STRESS_LOW, STRESS_MEDIUM, STRESS_HIGH):
            raise ValueError(
                f"classification must be one of "
                f"({STRESS_LOW}, {STRESS_MEDIUM}, {STRESS_HIGH}), "
                f"got {self.classification!r}"
            )
        if self.percentile is not None and not 0.0 <= self.percentile <= 1.0:
            raise ValueError("percentile must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["as_of"] = self.as_of.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketStress":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass(frozen=True)
class MarketRegime:
    """Rule-derived market regime with full component transparency."""

    as_of: datetime
    symbol: Optional[str]
    regime: str
    confidence: float
    components: Dict[str, Optional[float]]
    stress_classification: Optional[str] = None
    risk_appetite: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", _parse_ts(self.as_of))
        if self.regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got {self.regime!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.risk_appetite not in (None, "risk_on", "risk_off", "unknown"):
            raise ValueError("risk_appetite invalid")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["as_of"] = self.as_of.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketRegime":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, payload: str) -> "MarketRegime":
        return cls.from_dict(json.loads(payload))
