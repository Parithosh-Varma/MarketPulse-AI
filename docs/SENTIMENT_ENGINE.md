# Sentiment Engine (Phase 2)

Module: `src/sentiment.py` · Schemas: `src/models/sentiment.py`

## Purpose

One canonical interface for scoring financial text, replacing per-notebook
FinBERT snippets. Every scored item becomes a typed `SentimentObservation`
(UTC timestamp, source, ticker, probabilities, model name, confidence).

## Conventions

```
score      = P(positive) - P(negative)        # -1 bearish … +1 bullish
confidence = max(P(positive), P(neutral), P(negative))
```

These match the original notebook methodology
(`notebooks/03_sentiment_analysis.ipynb`, cells 3–4).

## Engines

| Engine | Class | Model | Probabilities | Notes |
|---|---|---|---|---|
| Primary | `FinBERTEngine` | `ProsusAI/finbert` | yes (softmax) | torch/transformers loaded lazily; device priority cuda > mps > cpu |
| Fallback | `VADEREngine` | vaderSentiment lexicon | no — score is compound; confidence = \|compound\| | pure-Python, no GPU needed |

`default_engine()` returns a process-wide singleton: FinBERT when
transformers+torch are importable, otherwise VADER with a logged warning.

### Label-order safety

The legacy notebook assumed softmax row order `(negative, neutral,
positive)` for ProsusAI/finbert. This engine instead resolves labels at
runtime from `model.config.id2label` and raises if the checkpoint's labels
cannot be mapped to {positive, negative, neutral}. Both known orderings are
handled and covered by tests.

## API

```python
from src.sentiment import default_engine

engine = default_engine()

obs = engine.analyze(
    "The company reported record profits.",
    source="newsapi",
    ticker="RELIANCE",
)
# -> SentimentObservation(score≈0.7, confidence≈0.9, model_name=…)

batch = engine.analyze_batch([
    {"text": "Margins under pressure", "source": "news", "ticker": "NIFTY"},
    {"text": "Record profits", "source": "reddit"},
])
```

- Batched inference: one tokenizer pass / one forward pass per batch.
- Long text: truncated by tokenizer to 512 tokens (`max_length`).
- Invalid text (empty, whitespace, non-string): immediate `ValueError` /
  `TypeError` — never silently dropped.

## Testing

Tests run fully offline via a deterministic mock engine plus real-VADER
checks; FinBERT internals are exercised with injected fakes (including both
label layouts). No network or model download happens in CI:

```bash
pytest tests/test_sentiment_engine.py -q
```

A live end-to-end check (downloads ~440 MB on first run):

```bash
python -c "
from src.sentiment import FinBERTEngine
e = FinBERTEngine()
print(e.analyze('Markets rally as inflation cools', ticker='NIFTY').score)"
```

## Aggregation (Phase 3)

Module: `src/aggregation.py` · Schema: `AggregatedSentiment`

- `mean_score` — arithmetic mean of scores in the group.
- `weighted_score` — Σ(wᵢ·scoreᵢ)/Σwᵢ with configurable per-source weights
  (`source_weights`); unknown sources use `default_source_weight=1.0`.
  Sources are not assumed equally reliable.
- `median_score` — robust central tendency.
- `mean_confidence` — mean of engine-reported confidences (None for
  pure-VADER corpora).
- `share_positive/negative/neutral` — fraction of observations with
  score > 0 / < 0 / == 0 (sign rule, engine-agnostic).

Groupings: `aggregate_by_ticker`, `aggregate_by_source`, and fixed UTC-grid
buckets via `aggregate_by_time_window(obs, window)` supporting 15m/1h/4h/1d/7d.
Buckets below `min_observations` are dropped explicitly, never merged.

## Known limitations

- VADER provides a polarity proxy, not calibrated probabilities; aggregation
  code must treat its `prob_*` fields as absent (Phase 3 handles this).
- English-only models; non-English text scores unreliably.
- `default_engine()` caches one engine per process — construct
  `FinBERTEngine()` directly for a different checkpoint/device.
