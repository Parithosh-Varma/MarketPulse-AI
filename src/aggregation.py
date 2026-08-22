"""Sentiment aggregation over observations and time windows (Phase 3).

Methodology (documented in docs/SENTIMENT_ENGINE.md consumers section):

- mean_score: arithmetic mean of observation scores in the group.
- weighted_score: sum(weight_i * score_i) / sum(weight_i) with per-source
  weights from a configurable mapping; unknown sources use
  ``default_weight``. Sources are NOT assumed equally reliable.
- median_score: robust central tendency.
- mean_confidence: mean of non-None confidences (None if no engine
  reported confidence, e.g. pure-VADER corpora).
- share_positive/negative/neutral: fraction of observations with
  score > 0 / < 0 / == 0 — consistent across engines.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from src.models import SentimentObservation
from src.models.aggregation import AggregatedSentiment

DEFAULT_SOURCE_WEIGHT = 1.0


class AggregationError(ValueError):
    pass


def _require_non_empty(observations: Sequence[SentimentObservation]) -> None:
    if not observations:
        raise AggregationError("cannot aggregate an empty observation set")


def _source_weights(
    weights: Optional[Mapping[str, float]], default_weight: float
) -> "Dict[str, float]":
    if weights is None:
        return {}
    cleaned: Dict[str, float] = {}
    for source, weight in weights.items():
        if weight <= 0:
            raise ValueError(f"weight for {source!r} must be > 0, got {weight}")
        cleaned[source] = float(weight)
    return cleaned


def classify(score: float, eps: float = 1e-9) -> str:
    if score > eps:
        return "positive"
    if score < -eps:
        return "negative"
    return "neutral"


def aggregate(
    observations: Sequence[SentimentObservation],
    *,
    source_weights: Optional[Mapping[str, float]] = None,
    default_source_weight: float = DEFAULT_SOURCE_WEIGHT,
    ticker: Optional[str] = None,
    sector: Optional[str] = None,
    market: Optional[str] = None,
    source: Optional[str] = None,
    window: Optional[timedelta] = None,
    metadata: Optional[dict] = None,
) -> AggregatedSentiment:
    """Aggregate observations into one :class:`AggregatedSentiment`."""
    _require_non_empty(observations)
    scores = [obs.score for obs in observations]
    weights_map = _source_weights(source_weights, default_source_weight)

    weighted_sum = 0.0
    weight_total = 0.0
    for obs in observations:
        w = weights_map.get(obs.source, default_source_weight)
        weighted_sum += w * obs.score
        weight_total += w

    confidences = [
        obs.confidence for obs in observations if obs.confidence is not None
    ]
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for s in scores:
        counts[classify(s)] += 1

    timestamps = [obs.timestamp for obs in observations]
    resolved_window = window or max(max(timestamps) - min(timestamps), timedelta(0))

    return AggregatedSentiment(
        window_start=min(timestamps),
        window_end=max(timestamps),
        window=resolved_window,
        n_observations=len(observations),
        mean_score=sum(scores) / len(scores),
        weighted_score=weighted_sum / weight_total,
        median_score=statistics.median(scores),
        mean_confidence=(
            sum(confidences) / len(confidences) if confidences else None
        ),
        share_positive=counts["positive"] / len(observations),
        share_negative=counts["negative"] / len(observations),
        share_neutral=counts["neutral"] / len(observations),
        ticker=ticker,
        sector=sector,
        market=market,
        source=source,
        metadata=metadata or {},
    )


def aggregate_by_ticker(
    observations: Sequence[SentimentObservation],
    **kwargs,
) -> dict[str, AggregatedSentiment]:
    """Group observations by ticker and aggregate each group."""
    _require_non_empty(observations)
    groups: dict[str, List[SentimentObservation]] = {}
    for obs in observations:
        groups.setdefault(obs.ticker or "_untagged", []).append(obs)
    return {
        ticker: aggregate(group, ticker=ticker if ticker != "_untagged" else None, **kwargs)
        for ticker, group in sorted(groups.items())
    }


def aggregate_by_source(
    observations: Sequence[SentimentObservation],
    **kwargs,
) -> dict[str, AggregatedSentiment]:
    """Group observations by source and aggregate each group."""
    _require_non_empty(observations)
    groups: dict[str, List[SentimentObservation]] = {}
    for obs in observations:
        groups.setdefault(obs.source, []).append(obs)
    return {
        src: aggregate(group, source=src, **kwargs)
        for src, group in sorted(groups.items())
    }


def floor_to_window(ts: datetime, window: timedelta) -> datetime:
    """UTC grid-floor a timestamp onto fixed windows since the epoch."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    seconds = int(window.total_seconds())
    if seconds <= 0:
        raise ValueError(f"window must be positive, got {window}")
    buckets = int((ts - epoch).total_seconds()) // seconds
    return epoch + timedelta(seconds=buckets * seconds)


def aggregate_by_time_window(
    observations: Sequence[SentimentObservation],
    window: timedelta,
    *,
    source_weights: Optional[Mapping[str, float]] = None,
    ticker: Optional[str] = None,
    min_observations: int = 1,
) -> List[AggregatedSentiment]:
    """Bucket observations into fixed UTC windows and aggregate each.

    Buckets containing fewer than ``min_observations`` items are dropped
    (never silently mixed into neighbours).
    """
    _require_non_empty(observations)
    if min_observations < 1:
        raise ValueError("min_observations must be >= 1")

    buckets: dict[datetime, List[SentimentObservation]] = {}
    for obs in observations:
        key = floor_to_window(obs.timestamp, window)
        buckets.setdefault(key, []).append(obs)

    results = []
    for start in sorted(buckets):
        group = buckets[start]
        if len(group) < min_observations:
            continue
        agg = aggregate(
            group,
            source_weights=source_weights,
            ticker=ticker,
        )
        results.append(
            AggregatedSentiment(
                window_start=start,
                window_end=start + window,
                window=window,
                n_observations=agg.n_observations,
                mean_score=agg.mean_score,
                weighted_score=agg.weighted_score,
                median_score=agg.median_score,
                mean_confidence=agg.mean_confidence,
                share_positive=agg.share_positive,
                share_negative=agg.share_negative,
                share_neutral=agg.share_neutral,
                ticker=ticker,
            )
        )
    return results
