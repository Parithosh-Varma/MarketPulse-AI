# MarketPulse AI — API Reference (src/)

All public entry points. Schemas live in `src/models/` and round-trip via
`to_dict()/from_dict()/to_json()/from_json()`.

## Sentiment (`src/sentiment.py`)
```python
from src.sentiment import default_engine, FinBERTEngine, VADEREngine

engine = default_engine()                      # FinBERT if available else VADER
obs = engine.analyze(text, source=..., ticker=..., timestamp=None, metadata={})
batch = engine.analyze_batch([text | dict, ...])   # one forward pass per batch
```
- `score = P(pos) − P(neg)`, `confidence = max(P(pos), P(neu), P(neg))`
- Raises on empty/whitespace/non-string text; truncates >512 tokens.

## Aggregation (`src/aggregation.py`)
```python
aggregate(obs_list, source_weights={...}, default_source_weight=1.0, ...)
aggregate_by_ticker(...) / aggregate_by_source(...)
aggregate_by_time_window(obs_list, timedelta(hours=1), min_observations=1)
```

## Momentum (`src/momentum.py`)
```python
momentum(aggregates, threshold=0.10) -> SentimentMomentum
momentum_series(aggregates, threshold=0.10) -> [SentimentMomentum]
# classification: improving / deteriorating / stable (+ accelerating, reversal)
```

## Market data (`src/market/`)
```python
from src.market import YFinanceProvider, FREDProvider, resolve_ticker

provider = YFinanceProvider()
provider.get_history("NIFTY50", date(2024,1,1), date(2024,12,31))  # -> [MarketObservation]
fred = FREDProvider()   # FRED_API_KEY env var; get_history("FEDFUNDS", ...)
```
Canonical symbols: `NIFTY50, BANKNIFTY, INDIAVIX, NIFTY_*, BTC`.
FII/DII: `FiiDiiProvider` interface + `ManualCsvFiiDiiProvider`.

## Divergence (`src/divergence.py`)
```python
compute_divergence(price_return, sentiment_change, symbol=..., as_of=...,
                   window=..., n_observations=..., threshold=0.30) -> DivergenceObservation
detect_divergence(prices, aggregates, symbol="NIFTY50", lookback_bars=5)
```

## Regime (`src/regime.py`)
```python
classify_regime(RegimeInputs(as_of=..., price_momentum=..., vix_level=...,
              sentiment=..., gmsi=...), config=RegimeConfig()) -> MarketRegime
make_stress(gmsi_value, as_of, p20=..., p80=...) -> MarketStress  # FinSentinel buckets
```
Rule precedence: high_volatility → risk_off → transition → risk_on →
bullish/bearish → neutral. Missing inputs never count as False votes.

## Backtesting (`src/backtest.py`)
```python
backtest_signal(prices, [(ts, value)...], signal_name=..., symbol=...,
                horizon_bars=3, entry_threshold=0.05) -> BacktestResult
daily_sentiment_signal(aggregates)          # adapter for the above
compare_with_baseline(result)               # §22-safe summary string
```

## Data quality (`src/quality.py`)
```python
check_sentiment_observations(rows) -> [QualityIssue]
check_market_observations(rows, expected_symbol=..., staleness_limit=...)
check_aggregate_continuity(aggs)
build_report(source, n_records, issues)     # .ok, .summary(), .to_dict()
```

## Alerts (`src/alerts.py`)
```python
engine = AlertEngine(channels=[LogChannel(), JsonlFileChannel("alerts.jsonl")],
                     rules=["sentiment_drop", ...], cooldown_seconds=3600)
fired = engine.process(context_dict)
```
Context keys: `sentiment_momentum`, `divergence_score`,
`divergence_classification`, `previous_regime`, `current_regime`,
`vix_level`, `vix_baseline`.

## Timezones (`src/timeutils.py`)
`ensure_utc(ts)` · `to_ist(ts)` · `ist_market_date(ts)` — UTC stored,
Asia/Kolkata for display.
