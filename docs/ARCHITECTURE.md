# FinSentinel — Architecture Report (Phase 1)

> Prepared as the foundation for **MarketPulse AI**, an incremental evolution of
> FinSentinel. This document describes the repository *as found*. No existing
> code was modified.

## 1. Project Summary

FinSentinel is a **financial-sentiment and market-dynamics research platform**
covering BTC and NIFTY 50. It is research-first (notebooks + scripts) with a
deployed read-only Streamlit dashboard. Core contributions already implemented:

| Contribution | What it does | Where it lives |
|---|---|---|
| **GMSI** | Global Market Stress Index — exogenous composite of GDELT event counts, negative event share, inverted Goldstein score, FinBERT/VADER sentiment, sentiment surprise, attention. Zero price-derived inputs. | `scripts/reconstruct_gmsi.py`, validated by `scripts/validate_gmsi.py`, `scripts/sanity_check_gmsi.py` |
| **MFI** | Market Fragility Index — mean of normalized AC₁(|r|), vol-of-vol, and tail-risk frequency, expanding min-max normalized (no look-ahead). | `scripts/08_market_dynamics_analysis.py` |
| **Shock Propagation** | Forward-vol decay after top-5% return shocks, conditioned on GMSI regime; half-life estimation. | `scripts/regime_analysis.py` (`shock_response`), notebook 08 |
| **Regime Analysis** | GMSI quantile regimes (Low/Medium/High stress), KS + Mann-Whitney tests, Wasserstein distances. | `scripts/regime_analysis.py` |
| **RF Vol Forecasting** | Random-forest next-week log-volatility model on sentiment-augmented features. | `pipeline/` + `run_pipeline.py`, artifact `models/rf_btc.pkl` |

**Explicit non-goal of the project:** price prediction. The README states the
research question is *"why and how markets move"*. MarketPulse AI must preserve
this framing (research / decision support, not trading).

## 2. Repository Layout

```
FinSentinel/
├── src/                     # EMPTY placeholder modules (fetch_data, preprocess,
│                            #   sentiment, correlation, utils — all 0 lines)
├── pipeline/                # RF modeling: dataset_builder, feature_engineering,
│                            #   train_rf_model, evaluate
├── data_pipeline/           # Production-grade ingestion layer
│   ├── fetch_all.py         # Master runner w/ --source registry
│   ├── sources/             # yfinance, fred, gdelt, trends, alphavantage,
│   │                        #   reddit, newsapi, quandl fetchers
│   └── utils/               # cache.py (24h TTL), logger.py, manifest.py
├── scripts/                 # One-off research pipelines (GMSI, MFI, regimes, VSI)
├── notebooks/               # 15 notebooks: collection → sentiment → alignment →
│                            #   correlation → RF → VSI → events → GMSI validation
├── dashboard/app.py         # 1,321-line Streamlit app (static research figures)
├── data/raw/                # Committed sample CSVs + manifest.json + pipeline.log
├── models/rf_btc.pkl        # Trained RF artifact
├── reports/figures/         # Research figures (duplicated in dashboard/assets/)
├── docs/methodology.md      # EMPTY placeholder
├── requirements.txt         # Unpinned full-project deps
├── .env.example             # API key template ⚠ contains a real-looking FRED key
└── run_pipeline.py          # RF training entry point (BTC only)
```

## 3. Entry Points

| Command | Purpose |
|---|---|
| `python run_pipeline.py` | Train/evaluate RF volatility model on `data/processed/{asset}_sentiment_aligned.csv` |
| `python data_pipeline/fetch_all.py [--source yfinance,fred,...]` | Ingest all/selected sources into `data/raw/`, writes `manifest.json` |
| `streamlit run dashboard/app.py` | Dashboard (7 pages: Overview, GMSI, Placebo, MFI, Shock, Regime, Methodology) |
| `python scripts/<name>.py` | Research pipelines (run from repo root; paths are CWD-relative) |

## 4. Runtime & Dependencies

- **Python 3.11** (pinned in `dashboard/runtime.txt`; repo badge says 3.11).
- Root `requirements.txt` is unpinned and includes questionable entries:
  `django-pipeline` (unrelated), `tensorflow` + `torch` both listed (only
  transformers/FinBERT used in notebooks), `snscrape` (abandoned upstream).
  Dashboard deps are separately pinned in `dashboard/requirements.txt`.
- No `pyproject.toml` / `setup.py` / `setup.cfg`. No linting or CI config.
- **No test suite exists** (`src/models/` schemas added in Phase 1 are the first
  tested code).

## 5. Configuration & Environment Variables

From `.env.example` (loaded via `python-dotenv` in `data_pipeline/fetch_all.py`):

| Variable | Source |
|---|---|
| `FRED_API_KEY` | FRED macro series |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCP service account JSON for GDELT BigQuery |
| `ALPHA_VANTAGE_KEY` | GDP / Fed Funds / CPI gap-fill |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` / `REDDIT_USER_AGENT` | PRAW social sentiment |
| `NEWSAPI_KEY` | Headlines for FinBERT/VADER scoring |
| `QUANDL_API_KEY` | Nasdaq Data Link |

Key-free sources: **yfinance** (BTC-USD, ^NSEI NIFTY 50, ^VIX, FX, bonds),
**Google Trends** via pytrends.

> ⚠️ **Security finding:** `.env.example` line 16 contains what appears to be a
> real FRED API key committed to git. Do not copy it anywhere; it should be
> revoked/rotated by the owner and replaced with a placeholder.

## 6. Data Sources (existing)

| Source | Data | Output |
|---|---|---|
| yfinance | OHLCV BTC-USD, ^NSEI, ^GSPC, ^VIX… since 2010 | CSV per ticker: `open, high, low, close, volume, adj_close, log_return, vol_7d/14d/30d/60d` |
| GDELT GKG | Event counts, avg Goldstein, negative share, themes (2015–) | Parquet cache → `events_daily_2015_2025.csv` |
| NewsAPI + Reddit + text corpus | Raw text rows | `text_data.csv`: `timestamp, text, source, asset, channel` |
| Google Trends | Search volume, 14 financial terms | `{asset}_google_trends.csv` |
| FRED / AlphaVantage / Quandl | Macro series | CSVs (FRED integration pending key) |

**Gaps vs. MarketPulse AI requirements:** no India-specific VIX feed (repo uses
CBOE `^VIX` only; India VIX will need e.g. Yahoo `^INDIAVIX`), no FII/DII
activity source, no BANK NIFTY ticker (^NSEBANK not yet pulled), no social-media
ingestion wired beyond raw Reddit fetcher.

## 7. Storage Model

**No database.** Everything is file-based:

- `data/raw/*.csv` — per-source pulls; a few sample files are committed to git.
- `data/raw/manifest.json` — auto-written quality manifest (rows, date range,
  % missing per column, pull timestamp). Note: historical entries reference a
  Windows path (`E:/...`), so manifests are machine-specific.
- `data/raw/*.parquet` — GDELT month-by-month cache (24 h TTL via
  `utils/cache.py`).
- `data/processed/*.csv` — aligned analysis tables (`gmsi_exogenous.csv`,
  `btc_vsi_full.csv`, `nifty_vsi_full.csv`, …); mostly gitignored/not present.
- `models/*.pkl` — joblib artifacts.

## 8. Data Flow

```
                ┌──────────────────── data_pipeline/fetch_all.py ────────────────────┐
                │  yfinance  fred  gdelt  trends  alphavantage  reddit  newsapi       │
                └──────────────────────────────┬─────────────────────────────────────┘
                                               ▼
                                     data/raw/ (+manifest.json)
                                               ▼
   notebooks 01–04: cleaning, FinBERT/VADER scoring, daily aggregation,
                    time-alignment            ▼
                                     data/processed/
                       (text_with_sentiment.csv, *_sentiment_aligned.csv)
                                               ▼
   scripts/reconstruct_gmsi.py ──► gmsi_exogenous.csv ──► scripts/validate/sanity_check
   scripts/construct_vsi_full.py ► {btc,nifty}_vsi_full.csv
   scripts/08_market_dynamics_analysis.py ► MFI + shock propagation figures
   scripts/regime_analysis.py ──► regime stats/figures
   pipeline/ + run_pipeline.py ──► RF vol model (models/rf_btc.pkl)
                                               ▼
                          dashboard/app.py (renders static PNGs + synthetic overlays)
```

Two parallel lineages exist:

1. **Notebook lineage** (01→13): exploratory, produced the published findings.
2. **Scripts lineage**: partially consolidated re-implementations
   (`reconstruct_gmsi.py`, `regime_analysis.py`) that read/write
   `data/processed/`.

The two agree on methodology but drift on file/column names (see §10).

## 9. Sentiment Stack

- **Models:** FinBERT (HuggingFace `transformers`) primary; VADER fallback.
  Applied inside notebooks 03/09/10 — **no reusable module exists**
  (`src/sentiment.py` is empty).
- **Downstream contract:** consumers expect columns `finbert_score`
  (−1…1 or prob-derived), `vader_score`, plus text rows keyed by
  `timestamp, asset, channel/source`.
- **Aggregation:** daily mean per asset (`sentiment_core` = finbert falling
  back to vader); "sentiment surprise" = deviation from 30-day rolling mean
  (this is the seed of Phase 3 *sentiment momentum*).

## 10. Important Modules

| Module | Role | Notes |
|---|---|---|
| `data_pipeline/fetch_all.py` | Registry-driven master runner | Well-structured; add new sources here |
| `data_pipeline/utils/{cache,logger,manifest}.py` | Cross-cutting infra | Reuse as-is for new fetchers |
| `scripts/reconstruct_gmsi.py` | Canonical exogenous GMSI build | Expanding z-scores, equal-weight baseline (README documents PCA-weight variant) |
| `scripts/08_market_dynamics_analysis.py` | MFI construction + shock propagation | Has **synthetic-data fallback** that can silently mask missing inputs |
| `scripts/regime_analysis.py` | Regimes, diagnostics, tail conditioning | Statistical core for future regime detection |
| `scripts/construct_vsi_full.py` | Merges prices+vol+sentiment+events into VSI tables | Template for the unified feature store |
| `pipeline/feature_engineering.py` | Sentiment×volatility interaction features | Feeds RF model |
| `dashboard/app.py` | Presentation layer | Static figures; real-time rewrite deferred to Phase 8 |

## 11. Where New Functionality Should Be Added

1. **`src/` is the designated home for reusable library code** — every module
   there is an empty placeholder awaiting implementation. Phase 1 adds
   `src/models/` (schemas). Later phases add `src/sentiment_engine/`,
   `src/aggregation/`, `src/divergence/`, `src/backtest/`, `src/regime/`.
2. **New data sources** follow the `data_pipeline/sources/fetch_<name>.py`
   pattern and get registered in `SOURCE_REGISTRY` in `fetch_all.py`
   (candidates: India VIX `^INDIAVIX`, BANK NIFTY `^NSEBANK`, FII/DII flows).
3. **New dashboard pages** extend `dashboard/app.py` until Phase 8 refactors it.
4. **Tests** live under `tests/` (new directory) runnable with pytest.

## 12. Technical Debt Register

| # | Debt | Impact | Suggested remediation phase |
|---|---|---|---|
| 1 | Real-looking FRED key committed in `.env.example` | Credential leak | Immediate (owner rotates key) |
| 2 | `src/*.py` all empty; logic trapped in notebooks/scripts | No importable library; duplication | Phases 2–4 fill these |
| 3 | Unpinned root `requirements.txt`; `django-pipeline`, dual TF+torch, dead `snscrape` | Fragile installs | Phase 2 (pin + prune) |
| 4 | CWD-relative paths in scripts; manifest records Windows paths | Breaks portability/CI | Phase 5 (path constants via `Path(__file__)`) |
| 5 | Column-name drift: `return` vs `log_return`, `volatility_7d` vs `vol_7d`, `sentiment_core` vs `finbert_score` | Silent merge failures | Phase 3 (single schema) |
| 6 | Synthetic-data fallbacks in analysis scripts | Fake results look real if inputs missing | Phase 6 (fail loudly) |
| 7 | Mixed timezone conventions (tz-aware text timestamps vs naive dates) | Alignment bugs | Phase 3 (UTC-aware schema) |
| 8 | No tests, lint, CI | Regression risk | Started Phase 1; complete Phase 10 |
| 9 | Duplicated figure assets (`reports/figures` ↔ `dashboard/assets`) | Drift | Phase 8 |
| 10 | `docs/methodology.md` empty despite rich README content | Knowledge siloed in notebooks | Phase 7 |

## 13. Components That Must NOT Be Rewritten

These encode validated research methodology (placebo-tested findings) and
production-quality plumbing — **extend, don't replace**:

1. `data_pipeline/` registry + `utils/cache|logger|manifest` infrastructure.
2. `scripts/reconstruct_gmsi.py` — exogenous GMSI math (expanding z-scores,
   no look-ahead normalization).
3. MFI component definitions (AC₁ persistence, CoV, tail frequency) in
   `08_market_dynamics_analysis.py`.
4. `scripts/regime_analysis.py` statistical tests (KS, Mann-Whitney,
   Wasserstein, shock-response local projections).
5. The dashboard's research-figure pages (Phase 8 adds live pages around them).
6. The "no price-derived inputs in GMSI" invariant — any new sentiment index
   must preserve exogeneity or explicitly mark itself as hybrid.

## 14. Schema Decision (Phase 1)

The project uses **plain pandas/dict/CSV conventions with no Pydantic
anywhere** (grep across code and requirements confirms). Per the instruction
"prefer Pydantic if the existing project already uses it", the new models in
`src/models/` use **stdlib `dataclasses` with explicit `__post_init__`
validation** — type-safe, zero new dependencies, and trivially convertible
to/from the dict/CSV shapes the existing pipeline already produces. Migration
to Pydantic later would be mechanical if the team opts in.
