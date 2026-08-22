"""Typed schema for backtest results (Phase 8)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, Dict, Optional

from src.models.aggregation import _parse_ts


@dataclass(frozen=True)
class BacktestResult:
    """Evaluation of a signal against forward market movement.

    All forward returns are simple returns measured from the first close
    STRICTLY AFTER the signal timestamp (no look-ahead). ``hit_rate`` is the
    share of signals whose predicted direction matched the realized
    forward-return sign. ``random_hit_rate`` is the empirical base rate of
    that horizon's positive returns — a coin flip scores this, so a signal
    must beat it meaningfully before any claim is made.
    """

    signal_name: str
    symbol: str
    horizon_bars: int
    period_start: datetime
    period_end: datetime
    n_signals: int
    n_evaluated: int
    hit_rate: float
    mean_forward_return: float
    median_forward_return: float
    std_forward_return: float
    trade_sharpe: float
    buy_hold_return: float
    excess_vs_buy_hold: float
    random_hit_rate: float
    max_drawdown: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "period_start", _parse_ts(self.period_start))
        object.__setattr__(self, "period_end", _parse_ts(self.period_end))
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        if self.n_evaluated < 0:
            raise ValueError("n_evaluated must be >= 0")
        if self.n_evaluated > self.n_signals:
            raise ValueError("n_evaluated cannot exceed n_signals")
        for name in ("hit_rate", "random_hit_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_drawdown > 0:
            raise ValueError("max_drawdown must be <= 0 (a loss)")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["period_start"] = self.period_start.isoformat()
        data["period_end"] = self.period_end.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestResult":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, payload: str) -> "BacktestResult":
        return cls.from_dict(json.loads(payload))
