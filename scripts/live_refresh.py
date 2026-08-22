"""One-command live refresh: fresh market data -> snapshot -> demo site -> push.

Usage:
    python scripts/live_refresh.py            # refresh local data + pages
    python scripts/live_refresh.py --push     # also publish to GitHub Pages
"""

from __future__ import annotations

import csv
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LIVE_DIR = ROOT / "data" / "raw" / "live"

SYMBOLS = {
    # canonical -> (provider file stem)
    "NIFTY50": "nifty",
    "BANKNIFTY": "banknifty",
    "INDIAVIX": "indiavix",
    "BTC": "btc",
}


def fetch_live_prices(days: int = 90) -> None:
    from src.market import YFinanceProvider

    provider = YFinanceProvider()
    end = date.today()
    start = end - timedelta(days=days)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    for symbol, stem in SYMBOLS.items():
        rows = provider.get_history(symbol, start, end)
        out = LIVE_DIR / f"{stem}_prices.csv"
        with open(out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["date", "open", "high", "low", "close", "volume"]
            )
            written = 0
            for r in rows:
                if r.vix is not None and r.open == r.high == r.low == r.close:
                    # vol-index bars: keep close as all OHLC fields
                    pass
                writer.writerow(
                    [
                        r.timestamp.date().isoformat(),
                        f"{r.open:.4f}",
                        f"{r.high:.4f}",
                        f"{r.low:.4f}",
                        f"{r.close:.4f}",
                        int(r.volume or 0),
                    ]
                )
                written += 1
        print(f"[live] {symbol}: {written} bars -> {out.relative_to(ROOT)}")


def rebuild_snapshot() -> None:
    import build_demo_site as bds
    import build_pulse_summary as bps

    bps.write_snapshot(400)
    bds.main()


def publish() -> None:
    cmds = [
        ["git", "add", "data/processed/marketpulse_summary.json", "docs/index.html"],
        ["git", "commit", "-m", "chore(live): refresh intelligence snapshot"],
        ["git", "push", "origin", "main"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0 and "nothing to commit" not in output:
            print(f"[publish] {' '.join(cmd)} failed:\n{output}")
            sys.exit(1)
        elif output:
            print(f"[publish] {output.splitlines()[0]}")
    print("[publish] pushed — GitHub Pages updates in ~60s")


def main() -> None:
    push = "--push" in sys.argv
    print("── MarketPulse live refresh ─────────────────────────")
    fetch_live_prices()
    rebuild_snapshot()
    if push:
        publish()
    print("─────────────────────────────────────────────────────")
    print("done. dashboard: streamlit run dashboard/app.py")
    if not push:
        print("(add --push to republish the public demo page)")


if __name__ == "__main__":
    main()
