"""Compute the MarketPulse intelligence snapshot for the dashboard page.

Runs the offline pipeline over data/raw sample feeds (or any future live
feeds) and writes:

    data/processed/marketpulse_summary.json   — one snapshot object
    data/processed/alerts.jsonl               — alert engine output

The Streamlit "MarketPulse Live" page renders this file; all analytics stay
in src/, never in UI code.

Usage:
    python scripts/build_pulse_summary.py [max_text_rows]
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aggregation import aggregate_by_ticker, aggregate_by_time_window  # noqa: E402
from src.alerts import AlertEngine, JsonlFileChannel  # noqa: E402
from src.backtest import backtest_signal, daily_sentiment_signal  # noqa: E402
from src.divergence import detect_divergence  # noqa: E402
from src.models import MarketObservation, SentimentObservation  # noqa: E402
from src.momentum import momentum_series  # noqa: E402
from src.regime import RegimeInputs, classify_regime  # noqa: E402
from src.sentiment import default_engine  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"


def load_text(limit: int) -> list[SentimentObservation]:
    engine = default_engine()
    out: list[SentimentObservation] = []
    with open(RAW / "text_data.csv", newline="", encoding="utf-8") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            if i >= limit:
                break
            try:
                ts = datetime.fromisoformat(row["timestamp"].strip())
                ticker = (row.get("asset") or "").upper()
                out.append(
                    engine.analyze(
                        row["text"],
                        source=(row.get("source") or row.get("channel") or "").strip()
                        or "unlabeled_feed",
                        ticker={"NIFTY": "NIFTY50"}.get(ticker, ticker or None),
                        timestamp=ts,
                    )
                )
            except (ValueError, TypeError):
                continue
    return out


def _price_file(symbol_key: str) -> Path:
    """Prefer freshly pulled live bars; fall back to bundled history."""
    live = RAW / "live" / f"{symbol_key.lower()}_prices.csv"
    bundled = {
        "BTC": "btc_prices.csv",
        "NIFTY50": "nifty_prices.csv",
    }.get(symbol_key)
    if live.exists():
        return live
    if bundled:
        return RAW / bundled
    raise FileNotFoundError(f"no price data for {symbol_key}")


def load_prices(symbol_key: str) -> list[MarketObservation]:
    rows: list[MarketObservation] = []
    with open(_price_file(symbol_key), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            close = float(row["close"])
            rows.append(
                MarketObservation(
                    timestamp=datetime.fromisoformat(row["date"]).replace(
                        tzinfo=timezone.utc
                    ),
                    symbol=symbol_key,
                    open=float(row["open"]),
                    high=max(close, float(row["high"])),
                    low=float(row["low"]),
                    close=close,
                    volume=float(row["volume"] or 0),
                )
            )
    return rows


def build_snapshot(max_rows: int) -> dict:
    observations = load_text(max_rows)
    by_ticker = aggregate_by_ticker(observations)

    focus = "NIFTY50" if "NIFTY50" in by_ticker else next(iter(by_ticker))
    scored_focus = [o for o in observations if o.ticker == focus]
    daily = aggregate_by_time_window(scored_focus, timedelta(days=1), min_observations=1)
    momentum = momentum_series(daily) if len(daily) >= 2 else []

    prices = load_prices(focus)
    if daily:
        lo = daily[0].window_start - timedelta(days=15)
        hi = daily[-1].window_end + timedelta(days=60)
        aligned_prices = [p for p in prices if lo <= p.timestamp <= hi]
    else:
        aligned_prices = prices

    divergence = (
        detect_divergence(aligned_prices, daily, symbol=focus, lookback_bars=10)
        if len(daily) >= 2 and aligned_prices
        else None
    )

    # Regime runs on the LATEST bars regardless of sentiment-span overlap,
    # and consumes real India VIX when live bars exist.
    vix_level = None
    try:
        vix_rows = load_prices("INDIAVIX")
        if vix_rows:
            vix_level = float(vix_rows[-1].vix or vix_rows[-1].close)
    except FileNotFoundError:
        pass

    regime_prices = aligned_prices or prices[-60:]
    tail = regime_prices[-21:]
    mom_value = tail[-1].close / tail[0].close - 1.0 if len(tail) >= 2 else None
    rets = [
        b.close / a.close - 1.0
        for a, b in zip(tail[:-1], tail[1:])
        if a.close > 0
    ]
    vol = (sum(r * r for r in rets) / len(rets)) ** 0.5 if rets else None
    # sentiment counts only if its freshest bucket is recent relative to bars
    last_sent = None
    sent_mom_value = None
    if daily and regime_prices:
        gap_days = (
            regime_prices[-1].timestamp - daily[-1].window_end
        ).days
        if abs(gap_days) <= 10:
            last_sent = daily[-1].mean_score
            if len(momentum) >= 1:
                sent_mom_value = momentum[-1].momentum
    regime = (
        classify_regime(
            RegimeInputs(
                as_of=regime_prices[-1].timestamp,
                symbol=focus,
                price_momentum=mom_value,
                realized_volatility=vol,
                vix_level=vix_level,
                sentiment=last_sent,
                sentiment_momentum=sent_mom_value,
            )
        )
        if regime_prices
        else None
    )

    backtest = None
    try:
        backtest = backtest_signal(
            aligned_prices,
            daily_sentiment_signal(daily),
            signal_name="daily_sentiment",
            symbol=focus,
            horizon_bars=3,
            entry_threshold=0.05,
        )
    except Exception:
        backtest = None

    context = {
        "sentiment_momentum": momentum[-1].momentum if momentum else None,
        "divergence_score": divergence.divergence_score if divergence else None,
        "divergence_classification": (
            divergence.classification if divergence else None
        ),
        "previous_regime": None,
        "current_regime": regime.regime if regime else None,
        "vix_level": None,
    }
    engine = AlertEngine(channels=[JsonlFileChannel(OUT_DIR / "alerts.jsonl")])
    fired = engine.process(context)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "focus_symbol": focus,
        "engine_model": default_engine().model_name,
        "sentiment_by_asset": {
            t: {
                "n": agg.n_observations,
                "score": round(agg.weighted_score or agg.mean_score or 0, 4),
                "share_positive": round(agg.share_positive or 0, 3),
                "share_negative": round(agg.share_negative or 0, 3),
                "share_neutral": round(agg.share_neutral or 0, 3),
            }
            for t, agg in sorted(by_ticker.items())
        },
        "momentum_recent": [
            {
                "as_of": m.current_window_end.date().isoformat(),
                "previous": round(m.previous_score, 4),
                "current": round(m.current_score, 4),
                "momentum": round(m.momentum, 4),
                "classification": m.classification,
                "accelerating": m.accelerating,
                "reversal": m.reversal,
            }
            for m in momentum[-7:]
        ],
        "divergence": (
            {
                "as_of": divergence.as_of.date().isoformat(),
                "window_days": divergence.window.days,
                "price_return": round(divergence.price_return, 4),
                "sentiment_change": round(divergence.sentiment_change, 4),
                "score": round(divergence.divergence_score, 3),
                "classification": divergence.classification,
                "confidence": round(divergence.confidence, 3),
            }
            if divergence
            else None
        ),
        "regime": (
            {
                "regime": regime.regime,
                "confidence": regime.confidence,
                "risk_appetite": regime.risk_appetite,
                "stress": regime.stress_classification,
                "components": regime.components,
            }
            if regime
            else None
        ),
        "backtest": (
            {
                "hit_rate": round(backtest.hit_rate, 3),
                "random_hit_rate": round(backtest.random_hit_rate, 3),
                "n_evaluated": backtest.n_evaluated,
                "mean_forward_return": round(backtest.mean_forward_return, 5),
                "buy_hold": round(backtest.buy_hold_return, 4),
                "max_drawdown": round(backtest.max_drawdown, 4),
                "note": backtest.metadata["note"],
            }
            if backtest
            else None
        ),
        "alerts_fired_this_run": [a.to_dict() for a in fired],
    }


def write_snapshot(max_rows: int = 400) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = build_snapshot(max_rows)
    target = OUT_DIR / "marketpulse_summary.json"
    target.write_text(json.dumps(snapshot, indent=2))
    print(
        f"wrote {target} ({snapshot['focus_symbol']}, "
        f"{len(snapshot['sentiment_by_asset'])} assets)"
    )
    return target


def main() -> None:
    max_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    write_snapshot(max_rows)


if __name__ == "__main__":
    main()
