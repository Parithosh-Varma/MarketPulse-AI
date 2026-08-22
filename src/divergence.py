"""Price/sentiment divergence engine (Phase 6).

Core idea: compare how far price moved vs how far sentiment moved over the
same window. Disagreement alone is NOT flagged as a signal — classification
requires |score| >= threshold AND opposing directions.

Definitions (documented in docs/API.md):

    p = clip(price_return     / price_scale,     -1, 1)
    s = clip(sentiment_change / sentiment_scale,  -1, 1)

    divergence_score = clip(s - p, -1, 1)      # sentiment minus price

    same-sign moves        -> aligned_bullish / aligned_bearish (or neutral
                              if both tiny)
    opposite-sign moves and
    |divergence_score| >= threshold -> bearish_divergence (price up /
                              sentiment down) or bullish_divergence
    otherwise              -> neutral

confidence combines sample support (log-scaled n_observations) with the
strength of the score; extreme volatility_context dampens confidence.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional, Sequence

from src.models import MarketObservation
from src.models.aggregation import AggregatedSentiment
from src.models.divergence import (
    CLASS_ALIGNED_BEARISH,
    CLASS_ALIGNED_BULLISH,
    CLASS_BEARISH_DIVERGENCE,
    CLASS_BULLISH_DIVERGENCE,
    CLASS_NEUTRAL,
    DivergenceObservation,
)

DEFAULT_THRESHOLD = 0.30
DEFAULT_PRICE_SCALE = 0.05  # ±5% daily-range maps to ±1
DEFAULT_SENTIMENT_SCALE = 0.50  # ±0.5 sentiment change maps to ±1
_EPS = 1e-12


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def compute_divergence(
    price_return: float,
    sentiment_change: float,
    *,
    symbol: str,
    as_of: datetime,
    window: timedelta,
    n_observations: int,
    threshold: float = DEFAULT_THRESHOLD,
    price_scale: float = DEFAULT_PRICE_SCALE,
    sentiment_scale: float = DEFAULT_SENTIMENT_SCALE,
    volatility_context: Optional[float] = None,
) -> DivergenceObservation:
    """Classify one price-vs-sentiment comparison."""
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be in [0, 1]")
    if price_scale <= 0 or sentiment_scale <= 0:
        raise ValueError("scales must be positive")

    p = _clip(price_return / price_scale)
    s = _clip(sentiment_change / sentiment_scale)
    score = _clip(s - p)

    if abs(p) < _EPS or abs(s) < _EPS:
        classification = CLASS_NEUTRAL
    elif p > 0 and s < 0:
        classification = (
            CLASS_BEARISH_DIVERGENCE if abs(score) >= threshold else CLASS_NEUTRAL
        )
    elif s > 0 > p:
        classification = (
            CLASS_BULLISH_DIVERGENCE if abs(score) >= threshold else CLASS_NEUTRAL
        )
    else:
        if abs(price_return) < _EPS and abs(sentiment_change) < _EPS:
            classification = CLASS_NEUTRAL
        elif price_return > 0 or sentiment_change > 0:
            classification = CLASS_ALIGNED_BULLISH
        else:
            classification = CLASS_ALIGNED_BEARISH

    sample_support = math.log1p(max(n_observations, 0)) / math.log1p(50)
    vol_dampener = (
        1.0 / (1.0 + volatility_context * 10.0) if volatility_context else 1.0
    )
    confidence = _clip(min(sample_support, 1.0) * abs(score) * vol_dampener)

    return DivergenceObservation(
        symbol=symbol,
        as_of=as_of,
        window=window,
        price_return=price_return,
        sentiment_change=sentiment_change,
        divergence_score=score,
        classification=classification,
        confidence=confidence,
        n_observations=n_observations,
        volatility_context=volatility_context,
        threshold_used=threshold,
    )


def simple_return(first_close: float, last_close: float) -> float:
    """(last/first - 1); raises on non-positive first close."""
    if first_close <= 0:
        raise ValueError(f"first_close must be > 0, got {first_close}")
    return last_close / first_close - 1.0


def price_return_over(
    prices: Sequence[MarketObservation], lookback_bars: int
) -> tuple[float, datetime]:
    """Return (simple return over trailing N bars, as-of timestamp)."""
    if len(prices) < 2:
        raise ValueError("need at least 2 price observations")
    window_rows = prices[-(lookback_bars + 1):]
    if len(window_rows) < 2:
        raise ValueError("lookback_bars exceeds available history")
    ret = simple_return(window_rows[0].close, window_rows[-1].close)
    return ret, window_rows[-1].timestamp


def detect_divergence(
    prices: Sequence[MarketObservation],
    aggregates: Sequence[AggregatedSentiment],
    *,
    symbol: str,
    lookback_bars: int = 5,
    threshold: float = DEFAULT_THRESHOLD,
    price_scale: float = DEFAULT_PRICE_SCALE,
    sentiment_scale: float = DEFAULT_SENTIMENT_SCALE,
    volatility_context: Optional[float] = None,
) -> Optional[DivergenceObservation]:
    """Convenience aligner for daily bars + daily aggregated sentiment.

    Compares the trailing ``lookback_bars`` price return against the change
    between the last two aggregate points whose windows fall inside that
    price span. Returns None when alignment is impossible (missing data).
    """
    if len(prices) < 2 or len(aggregates) < 2:
        return None

    price_return, as_of = price_return_over(list(prices), lookback_bars)
    window_span = as_of - prices[max(0, len(prices) - lookback_bars - 1)].timestamp

    inside = [
        agg for agg in aggregates if agg.window_end <= as_of
    ]
    if len(inside) < 2:
        return None
    prev, curr = inside[-2], inside[-1]
    sentiment_change = (
        curr.weighted_score if curr.weighted_score is not None else curr.mean_score
    ) - (
        prev.weighted_score if prev.weighted_score is not None else prev.mean_score
    )
    n_obs = sum(agg.n_observations for agg in inside[-2:])

    return compute_divergence(
        price_return=price_return,
        sentiment_change=sentiment_change,
        symbol=symbol,
        as_of=as_of,
        window=window_span if window_span > timedelta(0) else timedelta(days=1),
        n_observations=n_obs,
        threshold=threshold,
        price_scale=price_scale,
        sentiment_scale=sentiment_scale,
        volatility_context=volatility_context,
    )


__all__ = [
    "DEFAULT_THRESHOLD",
    "DivergenceObservation",
    "CLASS_ALIGNED_BEARISH",
    "CLASS_ALIGNED_BULLISH",
    "CLASS_BEARISH_DIVERGENCE",
    "CLASS_BULLISH_DIVERGENCE",
    "CLASS_NEUTRAL",
    "compute_divergence",
    "detect_divergence",
    "price_return_over",
    "simple_return",
]
