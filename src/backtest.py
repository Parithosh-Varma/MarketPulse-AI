"""Walk-forward signal backtesting (Phase 8).

Methodology (docs/BACKTESTING.md has the full write-up):

    information available at T
        -> signal value s(T) in [-1, 1]
        -> entry at the FIRST close strictly after T   (no look-ahead)
        -> exit  horizon_bars later
        -> forward return = close(exit) / close(entry) - 1

Direction convention: positive signal predicts positive forward return.
Signals with an incomplete forward window are excluded from evaluation and
counted in ``n_signals - n_evaluated``.

Metrics: hit rate vs the empirical base rate of positive returns at that
horizon ("random baseline"), mean/median/std of forward returns, per-trade
Sharpe (mean/std — not annualized), buy-and-hold comparison over the same
span, and max drawdown of a sign-weighted equity curve. This is research
evaluation only; it is not a P&L simulator (no costs, no sizing).
"""

from __future__ import annotations

import statistics
from bisect import bisect_right
from datetime import datetime
from typing import List, Sequence, Tuple

from src.models import MarketObservation
from src.models.backtest import BacktestResult


class BacktestError(ValueError):
    pass


def _closes_after(
    timestamps: List[datetime],
    closes: List[float],
    ts: datetime,
) -> int:
    """Index of first bar with timestamp STRICTLY greater than ts."""
    return bisect_right(timestamps, ts)


def max_drawdown(returns: Sequence[float]) -> float:
    """Max peak-to-trough loss of a compounded curve (<= 0)."""
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for r in returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        if peak > 0:
            worst = min(worst, equity / peak - 1.0)
    return round(worst, 10)


def _base_rate(forward_returns: Sequence[float]) -> float:
    if not forward_returns:
        return 0.0
    return sum(1 for r in forward_returns if r > 0) / len(forward_returns)


def backtest_signal(
    prices: Sequence[MarketObservation],
    signals: Sequence[Tuple[datetime, float]],
    *,
    signal_name: str,
    symbol: str,
    horizon_bars: int = 5,
    entry_threshold: float = 0.05,
) -> BacktestResult:
    """Evaluate ``signals`` against forward price movement.

    Signals with |value| < entry_threshold are treated as no-opinion and
    skipped (they still count toward n_signals). Duplicate timestamps are
    deduplicated keeping the last value.
    """
    if len(prices) < 2:
        raise BacktestError("need at least 2 price observations")
    if horizon_bars < 1:
        raise BacktestError("horizon_bars must be >= 1")
    if entry_threshold < 0 or entry_threshold > 1:
        raise ValueError("entry_threshold must be in [0, 1]")

    timestamps = [p.timestamp for p in prices]
    closes = [float(p.close) for p in prices]

    merged: dict[datetime, float] = {}
    for ts, value in signals:
        merged[ts] = float(value)
    ordered = sorted(merged.items())
    n_signals_total = len(ordered)

    evaluated_returns: List[float] = []
    strategy_returns: List[float] = []
    hits = 0
    skipped_incomplete = 0
    skipped_flat = 0

    for ts, value in ordered:
        entry_idx = _closes_after(timestamps, closes, ts)
        # entry_idx == 0 => the signal predates every bar; entering at bar 0
        # would fabricate a common window for all such signals, so skip them.
        if entry_idx == 0:
            skipped_incomplete += 1
            continue
        exit_idx = entry_idx + horizon_bars
        if exit_idx >= len(closes):
            skipped_incomplete += 1
            continue
        if abs(value) < entry_threshold:
            skipped_flat += 1
            continue
        forward_return = closes[exit_idx] / closes[entry_idx] - 1.0
        evaluated_returns.append(forward_return)
        strategy_returns.append(
            forward_return if value > 0 else -forward_return
        )
        if (value > 0 and forward_return > 0) or (
            value < 0 and forward_return < 0
        ):
            hits += 1

    if not evaluated_returns:
        raise BacktestError(
            "no evaluable signals: every signal was flat or lacked a "
            "complete forward window"
        )

    mean_ret = statistics.mean(evaluated_returns)
    std_ret = statistics.stdev(evaluated_returns) if len(evaluated_returns) > 1 else 0.0
    sharpe = mean_ret / std_ret if std_ret > 0 else 0.0

    span_start = prices[0].timestamp
    span_end = prices[-1].timestamp
    buy_hold = closes[-1] / closes[0] - 1.0
    # buy-hold return expected over the same bar count as one signal window
    n_bars = len(closes) - 1
    per_bar = (1.0 + buy_hold) ** (1.0 / n_bars) - 1.0 if n_bars > 0 else 0.0
    horizon_buy_hold = (1.0 + per_bar) ** horizon_bars - 1.0

    return BacktestResult(
        signal_name=signal_name,
        symbol=symbol,
        horizon_bars=horizon_bars,
        period_start=span_start,
        period_end=span_end,
        n_signals=n_signals_total,
        n_evaluated=len(evaluated_returns),
        hit_rate=round(hits / len(evaluated_returns), 6),
        mean_forward_return=round(mean_ret, 8),
        median_forward_return=round(statistics.median(evaluated_returns), 8),
        std_forward_return=round(std_ret, 8),
        trade_sharpe=round(sharpe, 6),
        buy_hold_return=round(buy_hold, 8),
        excess_vs_buy_hold=round(mean_ret - horizon_buy_hold, 8),
        random_hit_rate=round(_base_rate(evaluated_returns), 6),
        max_drawdown=max_drawdown(strategy_returns),
        metadata={
            "entry_threshold": entry_threshold,
            "skipped_incomplete_window": skipped_incomplete,
            "skipped_no_opinion": skipped_flat,
            "horizon_buy_hold_baseline": round(horizon_buy_hold, 8),
            "note": "research evaluation, not a trading simulation",
        },
    )


def daily_sentiment_signal(
    aggregates: Sequence,
) -> List[Tuple[datetime, float]]:
    """Convert aggregated sentiment into (window_end, score) signal points."""
    out: List[Tuple[datetime, float]] = []
    for agg in aggregates:
        score = agg.weighted_score if agg.weighted_score is not None else agg.mean_score
        out.append((agg.window_end, score))
    return out


def compare_with_baseline(result: BacktestResult) -> str:
    """Plain-language summary honoring §22 (association, not prediction)."""
    edge = result.hit_rate - result.random_hit_rate
    verdict = (
        "no meaningful association" if abs(edge) < 0.05
        else "historically associated with direction"
    )
    return (
        f"{result.signal_name} on {result.symbol}: hit rate "
        f"{result.hit_rate:.1%} vs base rate {result.random_hit_rate:.1%} "
        f"(n={result.n_evaluated}) — {verdict}. Research metric only."
    )


__all__ = [
    "BacktestError",
    "BacktestResult",
    "backtest_signal",
    "compare_with_baseline",
    "daily_sentiment_signal",
    "max_drawdown",
]
