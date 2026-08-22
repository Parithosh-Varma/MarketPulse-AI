# Backtesting Methodology (Phase 8)

Module: `src/backtest.py` · Schema: `BacktestResult` (`src/models/backtest.py`)

## Timeline discipline (no look-ahead)

```
signal computed at T  ──►  entry = FIRST close strictly AFTER T
                       ──►  exit  = entry + horizon_bars
forward return = close(exit) / close(entry) − 1
```

- `bisect_right` guarantees entry is *strictly after* the signal timestamp.
- Signals older than the first price bar are **skipped** (never clamped to
  bar 0 — that bug would fabricate identical windows; regression-tested).
- Signals whose forward window extends past the last bar are skipped as
  incomplete and counted in metadata.
- `|signal| < entry_threshold` marks a no-opinion point (also skipped).

## Metrics

| Metric | Definition |
|---|---|
| hit_rate | share of evaluated signals whose sign matched forward-return sign |
| random_hit_rate | empirical base rate of positive returns at that horizon — a coin flip scores this |
| mean/median/std_forward_return | distribution of forward returns across signals |
| trade_sharpe | mean/std of forward returns (per-trade, NOT annualized) |
| buy_hold_return | full-period close-to-close return |
| excess_vs_buy_hold | mean forward return minus geometrically scaled per-window buy-hold |
| max_drawdown | peak-to-trough of a sign-weighted compounded curve |

Signal direction: positive value predicts positive forward return;
negative predicts negative. Strategy return flips sign with the signal.

## Baselines

1. **Random/base-rate**: `random_hit_rate`. A signal must beat this by a
   margin that survives sample-size scrutiny before any claim is made.
2. **Buy-and-hold**: reported alongside excess.

## Interpretation rules (§22)

`compare_with_baseline()` renders the verdict language:

> "…hit rate X% vs base rate Y% (n=…) — no meaningful association /
> historically associated with direction. **Research metric only.**"

No P&L simulation, no transaction costs, no position sizing — those would
imply tradability this research framework deliberately does not claim.

## Known limitations

- Overlapping signal windows share future bars → metrics are correlated,
  not independent draws (report n but treat significance conservatively).
- Single-asset series only; portfolio effects out of scope by design.
- Survivorship bias does not apply to index-level data used here but would
  matter for stock-universe studies.

Run: `pytest tests/test_backtest.py -q`
