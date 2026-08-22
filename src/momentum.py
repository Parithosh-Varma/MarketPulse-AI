"""Sentiment momentum computation over aggregated sentiment (Phase 4).

momentum = current_score - previous_score

Classification against a noise threshold ``t`` (default 0.10 on the [-1,1]
score scale):

- improving:     momentum >= +t
- deteriorating: momentum <= -t
- stable:        |momentum| < t

With three consecutive aggregates the detector additionally flags:

- accelerating: |Δ_now| > |Δ_prev| and both deltas push the same direction
- reversal:     Δ crosses zero relative to Δ_prev with both above threshold

These are descriptive research metrics — not trading signals.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from src.models.aggregation import AggregatedSentiment
from src.models.momentum import (
    CLASS_DETERIORATING,
    CLASS_IMPROVING,
    CLASS_STABLE,
    SentimentMomentum,
)

DEFAULT_THRESHOLD = 0.10


class MomentumError(ValueError):
    pass


def _classify(delta: float, threshold: float) -> str:
    if delta >= threshold:
        return CLASS_IMPROVING
    if delta <= -threshold:
        return CLASS_DETERIORATING
    return CLASS_STABLE


def _pair_momentum(
    previous: AggregatedSentiment,
    current: AggregatedSentiment,
    threshold: float,
    *,
    accelerating: Optional[bool] = None,
    reversal: Optional[bool] = None,
) -> SentimentMomentum:
    score_field = "weighted_score" if (
        current.weighted_score is not None and previous.weighted_score is not None
    ) else "mean_score"
    prev_score = getattr(previous, score_field)
    curr_score = getattr(current, score_field)
    return SentimentMomentum(
        ticker=current.ticker or previous.ticker,
        previous_window_end=previous.window_end,
        current_window_end=current.window_end,
        window=current.window or (current.window_end - previous.window_end),
        previous_score=prev_score,
        current_score=curr_score,
        classification=_classify(curr_score - prev_score, threshold),
        threshold=threshold,
        accelerating=accelerating,
        reversal=reversal,
        metadata={"score_basis": score_field},
    )


def momentum(
    aggregates: Sequence[AggregatedSentiment],
    *,
    ticker: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> SentimentMomentum:
    """Momentum between the last two consecutive aggregates."""
    if len(aggregates) < 2:
        raise MomentumError("need at least 2 aggregate points to compute momentum")
    result = _pair_momentum(aggregates[-2], aggregates[-1], threshold)
    if ticker is not None and result.ticker is None:
        result = SentimentMomentum(
            **{**result.to_dict(), "ticker": ticker}
        )
    return result


def momentum_series(
    aggregates: Sequence[AggregatedSentiment],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[SentimentMomentum]:
    """Pairwise momentum across a full series; third point enables flags.

    Element i compares aggregates[i-1] -> aggregates[i]. For i >= 2 the
    ``accelerating`` and ``reversal`` flags are populated from the prior
    delta.
    """
    if len(aggregates) < 2:
        raise MomentumError("need at least 2 aggregate points to compute momentum")

    out: List[SentimentMomentum] = []
    for i in range(1, len(aggregates)):
        prev_delta = None
        if i >= 2:
            p0 = aggregates[i - 2]
            p1 = aggregates[i - 1]
            basis_prev = (
                p0.weighted_score if p0.weighted_score is not None else p0.mean_score
            )
            basis_curr = (
                p1.weighted_score if p1.weighted_score is not None else p1.mean_score
            )
            prev_delta = basis_curr - basis_prev

        cur_delta = (
            aggregates[i].weighted_score
            if aggregates[i].weighted_score is not None
            else aggregates[i].mean_score
        ) - (
            aggregates[i - 1].weighted_score
            if aggregates[i - 1].weighted_score is not None
            else aggregates[i - 1].mean_score
        )

        accelerating: Optional[bool] = None
        reversal: Optional[bool] = None
        if prev_delta is not None:
            same_direction = prev_delta * cur_delta > 0
            accelerating = bool(same_direction and abs(cur_delta) > abs(prev_delta))
            reversal = bool(
                prev_delta * cur_delta < 0
                and abs(prev_delta) >= threshold
                and abs(cur_delta) >= threshold
            )

        out.append(
            _pair_momentum(
                aggregates[i - 1],
                aggregates[i],
                threshold,
                accelerating=accelerating,
                reversal=reversal,
            )
        )
    return out
