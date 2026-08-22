"""MarketPulse AI — end-to-end offline demo.

Runs the full intelligence pipeline on the sample data shipped in
data/raw/ (no API keys, no model downloads):

    text corpus -> sentiment scoring -> daily aggregation -> momentum
    -> divergence vs price -> regime classification -> mini backtest
    -> data-quality report

Usage:
    python scripts/demo_marketpulse.py [max_rows]
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aggregation import aggregate_by_ticker, aggregate_by_time_window  # noqa: E402
from src.backtest import backtest_signal, compare_with_baseline, daily_sentiment_signal  # noqa: E402
from src.divergence import detect_divergence  # noqa: E402
from src.market.base import resolve_ticker  # noqa: E402
from src.models import MarketObservation, SentimentObservation  # noqa: E402
from src.momentum import momentum_series  # noqa: E402
from src.quality import build_report, check_market_observations  # noqa: E402
from src.regime import RegimeInputs, classify_regime  # noqa: E402
from src.sentiment import default_engine  # noqa: E402

LINE = "=" * 64


def load_text_rows(limit: int):
    path = ROOT / "data/raw/text_data.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for i, row in enumerate(reader):
            if i >= limit:
                break
            if not row.get("text", "").strip():
                continue
            rows.append(row)
    return rows


def load_prices(symbol_key: str, limit: int | None = None):
    fname = {"BTC": "btc_prices.csv", "NIFTY50": "nifty_prices.csv"}[symbol_key]
    path = ROOT / "data/raw" / fname
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ts = datetime.fromisoformat(row["date"]).replace(tzinfo=timezone.utc)
            close = float(row["close"])
            rows.append(
                MarketObservation(
                    timestamp=ts,
                    symbol=symbol_key,
                    open=float(row["open"]),
                    high=max(close, float(row["high"])),
                    low=min(float(row["low"]), float(row["low"])),
                    close=close,
                    volume=float(row["volume"] or 0),
                )
            )
    return rows[-limit:] if limit else rows


def main(max_rows: int = 400) -> None:
    print(LINE)
    print(" MARKETPULSE AI — OFFLINE DEMO (sample data, no API keys)")
    print(LINE)

    # 1 ── sentiment scoring -------------------------------------------------
    engine = default_engine()
    print(f"\n[1] Scoring {max_rows} sample headlines with {engine.model_name} ...")
    raw_rows = load_text_rows(max_rows)
    observations = []
    for row in raw_rows:
        try:
            ts = datetime.fromisoformat(row["timestamp"].strip())
        except (KeyError, ValueError):
            ts = None
        try:
            observations.append(
                engine.analyze(
                    row["text"],
                    source=(row.get("source") or row.get("channel") or "").strip()
                    or "unlabeled_feed",
                    ticker=row.get("asset") or None,
                    timestamp=ts,
                )
            )
        except (ValueError, TypeError):
            continue
    scored = [o for o in observations if o.score != 0.0]
    bullish = sum(1 for o in scored if o.score > 0)
    bearish = sum(1 for o in scored if o.score < 0)
    print(f"    scored={len(scored)}  bullish={bullish}  bearish={bearish}")

    # normalize tickers to canonical names where we have a mapping
    canon = {"NIFTY": "NIFTY50"}
    for obs in scored:
        if obs.ticker and obs.ticker.upper() in canon:
            object.__setattr__(obs, "ticker", canon[obs.ticker.upper()])

    # 2 ── aggregation -------------------------------------------------------
    print("\n[2] Daily aggregated sentiment per asset:")
    by_ticker = aggregate_by_ticker(scored)
    for ticker, agg in sorted(by_ticker.items()):
        print(
            f"    {ticker:<10} n={agg.n_observations:>3}  "
            f"sentiment={agg.weighted_score:+.3f}  "
            f"pos/neg/neu = {agg.share_positive:.0%}/{agg.share_negative:.0%}/"
            f"{agg.share_neutral:.0%}"
        )

    focus = "NIFTY50" if "NIFTY50" in by_ticker else next(iter(by_ticker))
    daily = aggregate_by_time_window(
        [o for o in scored if o.ticker == focus],
        timedelta(days=1),
        min_observations=1,
    )

    # 3 ── momentum ----------------------------------------------------------
    print(f"\n[3] Sentiment momentum — {focus} (daily windows):")
    series = momentum_series(daily) if len(daily) >= 2 else []
    if not series:
        print("    fewer than 2 daily buckets in sample corpus — skipping")
    for m in series[-5:]:
        flags = []
        if m.accelerating:
            flags.append("accelerating")
        if m.reversal:
            flags.append("REVERSAL")
        suffix = f"  [{' ,'.join(flags)}]" if flags else ""
        print(
            f"    {m.current_window_end:%Y-%m-%d}  {m.previous_score:+.3f} -> "
            f"{m.current_score:+.3f}  Δ={m.momentum:+.3f}  {m.classification}{suffix}"
        )

    # 4 ── market data + quality --------------------------------------------
    print(f"\n[4] Price history — {focus}:")
    all_prices = load_prices(focus)
    report = build_report(
        focus,
        len(all_prices),
        check_market_observations(
            all_prices, expected_symbol=focus, staleness_limit=timedelta(days=3650)
        ),
    )
    span_days = (all_prices[-1].timestamp - all_prices[0].timestamp).days
    print(
        f"    bars={len(all_prices)}  span≈{span_days}d  "
        f"last_close={all_prices[-1].close:,.2f}  quality: {report.summary()}"
    )

    # trim price series to the sentiment observation span (+ margins) so the
    # analytics below compare like with like
    if daily:
        lo = daily[0].window_start - timedelta(days=15)
        hi = daily[-1].window_end + timedelta(days=60)
        prices = [p for p in all_prices if lo <= p.timestamp <= hi]
    else:
        prices = all_prices

    # 5 ── divergence --------------------------------------------------------
    print(f"\n[5] Price vs sentiment divergence — {focus}:")
    div = detect_divergence(prices, daily, symbol=focus, lookback_bars=10)
    if div:
        print(
            f"    window={div.window.days}d  price_return={div.price_return:+.2%}  "
            f"sentiment_change={div.sentiment_change:+.2f}\n"
            f"    divergence_score={div.divergence_score:+.2f}  "
            f"classification={div.classification}  confidence={div.confidence:.2f}"
        )
    else:
        print("    insufficient aligned data for a divergence reading")

    # 6 ── regime ------------------------------------------------------------
    print(f"\n[6] Market regime — {focus}:")
    tail = prices[-21:]
    mom = tail[-1].close / tail[0].close - 1.0
    rets = [
        b.close / a.close - 1.0
        for a, b in zip(tail[:-1], tail[1:])
        if a.close > 0
    ]
    vol = (sum(r * r for r in rets) / len(rets)) ** 0.5 if rets else None
    last_sent = daily[-1].mean_score if daily else None
    regime = classify_regime(
        RegimeInputs(
            as_of=prices[-1].timestamp,
            symbol=focus,
            price_momentum=mom,
            realized_volatility=vol,
            vix_level=None,
            sentiment=last_sent,
            gmsi=None,
            mfi=None,
        )
    )
    print(
        f"    20-bar momentum={mom:+.2%}  realized_vol≈{vol:.4f}  "
        f"sentiment={last_sent:+.3f}\n"
        f"    regime={regime.regime.upper()}  confidence={regime.confidence:.2f}  "
        f"votes={regime.metadata['rule_votes']}"
    )

    # 7 ── backtest ----------------------------------------------------------
    print(f"\n[7] Mini walk-forward backtest — sentiment signal on {focus}:")
    if len(daily) >= 6 and len(prices) >= 10:
        try:
            result = backtest_signal(
                prices,
                daily_sentiment_signal(daily),
                signal_name="daily_sentiment",
                symbol=focus,
                horizon_bars=3,
                entry_threshold=0.05,
            )
            print("    " + compare_with_baseline(result))
            print(
                f"    mean_fwd={result.mean_forward_return:+.4%}  "
                f"trade_sharpe={result.trade_sharpe:.2f}  "
                f"buy_hold={result.buy_hold_return:+.2%}  "
                f"max_dd={result.max_drawdown:.2%}"
            )
        except Exception as exc:
            print(f"    backtest skipped: {exc}")
    else:
        print("    need ≥6 aligned daily buckets for a meaningful mini-run")

    print("\n" + LINE)
    print(" Research metrics only — NOT trading advice or price prediction.")
    print(LINE)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
