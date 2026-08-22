# MarketPulse AI — Development Roadmap

Incremental evolution of the FinSentinel repository
(see `docs/ARCHITECTURE.md` for the as-found state). Each phase builds on the
previous one; existing validated research code is extended, never rewritten.
Scope guard: this is a **research and decision-support platform** — no order
execution, no broker connections, no price-prediction claims.

## Status (updated after autonomous build-out)

| Phase | Status | Key artifacts |
|---|---|---|
| 1 — Architecture & schemas | ✅ done | `docs/ARCHITECTURE.md`, `src/models/*`, first tests |
| 2 — Sentiment engine | ✅ done | `src/sentiment.py`, `docs/SENTIMENT_ENGINE.md` |
| 3 — Aggregation | ✅ done | `src/aggregation.py`, `AggregatedSentiment` |
| 4 — Momentum | ✅ done | `src/momentum.py`, `SentimentMomentum` |
| 5 — Market data pipeline | ✅ done | `src/market/*`, `docs/DATA_SOURCES.md`, `src/timeutils.py` |
| 6 — Divergence | ✅ done | `src/divergence.py`, `DivergenceObservation` |
| 7 — Regime engine | ✅ done | `src/regime.py`, `MarketRegime`/`MarketStress` |
| 8 — Backtesting | ✅ done | `src/backtest.py`, `docs/BACKTESTING.md` |
| 9 — Dashboard | ✅ v1 live-page | `dashboard/app.py` "MarketPulse Live", `scripts/build_pulse_summary.py` |
| 10 — Alerting | ✅ log/file channels | `src/alerts.py` (email/TG/webhook pending creds) |
| — Data quality layer | ✅ done | `src/quality.py` |
| — Offline demo | ✅ done | `scripts/demo_marketpulse.py` |

Known gaps carried forward: FinBERT weights not vendored (downloads on first
run); FII/DII manual CSV; alerts lack scheduler; root requirements.txt still
needs pin/prune pass; ruff/mypy wired but not yet enforced in CI.

## Phase 1 — Repository understanding and setup ✅

- Inspect structure, entry points, deps, data sources, storage, models.
- Write `docs/ARCHITECTURE.md` and this roadmap.
- Introduce typed schemas: `src/models/sentiment.py`, `src/models/market.py`.
- First test suite (`tests/`) runnable via pytest; Python 3.11 venv workflow.

**Exit criteria:** tests green; docs reviewed; nothing existing broken.

## Phase 2 — Unified sentiment engine

- Fill empty `src/sentiment.py`: one scoring interface over FinBERT
  (primary) and VADER (fallback), emitting `SentimentObservation` objects.
- Batch + single-text scoring; deterministic model versioning
  (`model_name`, confidence).
- Wire NewsAPI/Reddit/GDELT-text fetchers to produce observations instead of
  bare CSV rows (keep CSV export for backward compatibility).
- Pin/prune root `requirements.txt`; remove `django-pipeline`, decide
  torch-vs-tensorflow, drop `snscrape`.

**Exit criteria:** any fetched text row can be scored through one API;
golden-file regression test on a sample corpus.

## Phase 3 — Sentiment aggregation and sentiment momentum

- `src/aggregation/`: daily/per-window aggregation per ticker from raw
  observations (volume-weighted mean, source diversity count).
- Sentiment momentum: z-scored change vs trailing windows; formalize the
  "sentiment surprise" already used by GMSI (`reconstruct_gmsi.py`) on top of
  the new schema.
- Unify column names (`return` vs `log_return`, `volatility_7d` vs `vol_7d`)
  behind adapters so legacy scripts keep working.
- Normalize all timestamps to UTC-aware datetimes at schema level.

**Exit criteria:** GMSI reconstruction reproduces identical `pure_gmsi`
values when re-pointed at the aggregation layer.

## Phase 4 — Market/sentiment divergence detection

- New module `src/divergence/`: rolling rank/Spearman correlation between
  sentiment momentum and returns/volatility per asset; divergence flags with
  persistence rules (no single-day noise).
- Extend the placebo-test harness (`scripts/sanity_check_gmsi.py` pattern) to
  divergence signals — every signal ships with its empirical null.
- Dashboard page (static first): divergence timeline vs NIFTY/BANK NIFTY.

**Exit criteria:** divergence series computed leakage-free; placebo results
documented in `docs/methodology.md`.

## Phase 5 — Historical data pipeline

- Add sources following `data_pipeline/sources/fetch_<name>.py` +
  `SOURCE_REGISTRY`: India VIX (`^INDIAVIX`), BANK NIFTY (`^NSEBANK`),
  FII/DII activity (public exchange/depository disclosures — source TBD,
  no invented endpoints), keep CBOE VIX for cross-market comparison.
- Backfill orchestration: idempotent runs, manifest-based incremental pulls,
  repo-root-independent paths (`Path(__file__)`, not CWD).
- Landing zone layout documented; parquet for large history.

**Exit criteria:** `fetch_all.py --source all` populates a fresh machine;
manifest paths portable.

## Phase 6 — Backtesting engine

- `src/backtest/`: walk-forward evaluation of *signals* (sentiment momentum,
  MFI regimes, divergence) against forward volatility/vol-of-vol targets —
  **decision-support metrics, not P&L trading simulation**.
- Strict point-in-time data access (reuse expanding min-max / expanding
  z-score conventions); transaction-cost-free by design, clearly labeled.
- Replace synthetic-data fallbacks with hard failures + explicit
  `--synthetic-demo` opt-in flag.

**Exit criteria:** reproducible walk-forward reports; no look-ahead violations
(automated leakage check).

## Phase 7 — Market regime detection

- Build on `scripts/regime_analysis.py`: keep quantile baseline, add HMM
  regime fit (already flagged as Paper 3 next step in README status).
- Regime features: GMSI level, MFI components, India VIX term signals.
- Emit `RegimeLabel` observations consumable by dashboard/alerts; document in
  `docs/methodology.md`.

**Exit criteria:** regime labels stable under refit; agreement stats vs
quantile baseline reported.

## Phase 8 — Real-time dashboard

- Evolve `dashboard/app.py`: live pages reading cached observation stores
  (sentiment momentum, divergence, regimes, India VIX/FII-DII panels) while
  preserving existing research-figure pages.
- Refresh cadence bounded by upstream rate limits (NewsAPI free tier etc.);
  cache-first rendering.
- Deduplicate figure assets (`reports/figures` ↔ `dashboard/assets`).

**Exit criteria:** dashboard renders from the pipeline's current outputs with
no manual PNG copying.

## Phase 9 — Alerting

- Rule engine over stored signals: threshold/cross/persistence alerts
  (e.g., divergence confirmed N days, regime flip, VIX spike).
- Delivery channels decoupled from detection (webhook/email abstraction);
  rate-limited, deduplicated, full audit log.
- No trading actions ever — alerts are informational only.

**Exit criteria:** alert replay over historical window produces expected
events; audit trail queryable.

## Phase 10 — Production hardening

- CI: pytest + lint (ruff) + type-check (mypy) on push.
- Config validation at startup (fail fast on missing keys); secrets only via
  env/.env (rotate the leaked FRED key from `.env.example`).
- Structured logging everywhere (extend `data_pipeline/utils/logger.py`);
  data-quality gates using `manifest.json` (% missing thresholds).
- Packaging: `pyproject.toml`; pinned dependency lock; container recipe.
- Documentation: fill `docs/methodology.md`, runbooks, ADRs for schema
  decisions.

**Exit criteria:** green CI on clean clone; onboarding doc covers
setup → fetch → score → dashboard in <30 min.

## Dependency graph

```
P1 ─► P2 ─► P3 ─► P4 ─────┐
        └──► P5 ─► P6 ─► P7 ─► P8 ─► P9 ─► P10
```

(P5 can start once P2 lands; P6 needs P3+P5.)
