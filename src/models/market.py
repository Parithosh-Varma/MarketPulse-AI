"""Typed schemas for market price/volatility observations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional

from src.models.sentiment import _parse_timestamp, _check_range


@dataclass(frozen=True)
class MarketObservation:
    """One bar of market data for a symbol (index, stock, crypto, index-vol).

    `volatility` is the realized volatility measure used downstream (e.g. 7d
    rolling std of returns) — units/annualization belong in `metadata`.
    `vix` carries an associated volatility-index close (CBOE VIX or India VIX)
    where applicable.
    """

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    volatility: Optional[float] = None
    vix: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _parse_timestamp(self.timestamp))

        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")

        for name in ("open", "high", "low", "close"):
            value = getattr(self, name)
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a number, got {value!r}") from exc
            _check_range(name, value, 0.0, float("inf"))
            object.__setattr__(self, name, value)

        if self.high < max(self.open, self.close) - 1e-9:
            raise ValueError(
                f"high ({self.high}) must be >= max(open, close) "
                f"({max(self.open, self.close)})"
            )
        if self.low > min(self.open, self.close) + 1e-9:
            raise ValueError(
                f"low ({self.low}) must be <= min(open, close) "
                f"({min(self.open, self.close)})"
            )
        if self.high < self.low - 1e-9:
            raise ValueError(f"high ({self.high}) must be >= low ({self.low})")

        if self.volume is not None:
            volume = float(self.volume)
            if volume < 0:
                raise ValueError(f"volume must be >= 0, got {volume}")
            object.__setattr__(self, "volume", volume)

        for name in ("volatility", "vix"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if value < 0:
                    raise ValueError(f"{name} must be >= 0, got {value}")
                object.__setattr__(self, name, value)

    @property
    def timestamp_iso(self) -> str:
        return self.timestamp.isoformat()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp_iso
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketObservation":
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    def to_json(self, **json_kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **json_kwargs)

    @classmethod
    def from_json(cls, payload: str) -> "MarketObservation":
        return cls.from_dict(json.loads(payload))
