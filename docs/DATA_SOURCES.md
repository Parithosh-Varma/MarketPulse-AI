# Data Sources (Phase 5)

Providers live in `src/market/`. Canonical symbols are provider-independent
(`src/market/base.py::CANONICAL_SYMBOLS`): `NIFTY50`, `BANKNIFTY`,
`INDIAVIX`, sector indices (`NIFTY_IT`, `NIFTY_AUTO`, …), `BTC`.

## Yahoo Finance — `YFinanceProvider`

| | |
|---|---|
| Tickers | ^NSEI (NIFTY 50), ^NSEBANK (BANK NIFTY), ^INDIAVIX (India VIX), ^CNX* sectors, BTC-USD, equities via `.NS` suffix |
| Auth | none |
| Rate limits | undocumented; retries with exponential backoff built in (`with_retries`) |
| Licensing | Yahoo Terms of Service: personal/research use OK; **not** for redistribution or commercial products |
| Tests | offline — fake module injected; no network in CI |

## FRED — `FREDProvider`

| | |
|---|---|
| Series | `FEDFUNDS`, `CPIAUCSL`, `UNRATE`, `DGS10`, `DGS2`, `T10Y2Y`, `GDP` (`FREQUENT_SERIES`) |
| Auth | `FRED_API_KEY` env var only — never hard-coded (the previously leaked key must be rotated at fred.stlouisfed.org) |
| Licensing | free API key under FRED terms of use; attribution requested |
| Missing data | observations with value `"."` are skipped explicitly |

## FII/DII flows — `FiiDiiProvider`

No reliable free API exists for daily Indian FII/DII flows. Authoritative
primary sources (HTML/PDF pages, no stable JSON):

- NSE FII/DII daily activity report
- NSDL / CDSL depository monthly summaries

Rather than scraping fragile pages, analytics code depends on the
`FiiDiiProvider` interface. `ManualCsvFiiDiiProvider` loads a researcher-
maintained CSV:

```
date,institution,buy_value_cr,sell_value_cr,net_value_cr
2026-08-18,FII,12000.5,11500.0,500.5
2026-08-18,DII,9000.0,8500.25,499.75
```

Values in ₹ crore as published. When an acceptable programmatic source
emerges, implement a new provider behind the same interface.

## Timezone standard

All timestamps stored as UTC (`SentimentObservation`, `MarketObservation`).
Presentation converts to `Asia/Kolkata` via `src/timeutils.py`
(`to_ist`, `ist_market_date`). Naive datetimes are interpreted as UTC.
