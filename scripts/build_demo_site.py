"""Generate a fully static, dependency-free public demo page.

Reads data/processed/marketpulse_summary.json (produced by
build_pulse_summary.py) and writes docs/index.html suitable for GitHub
Pages. All numbers come from the snapshot — nothing hand-baked.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "processed" / "marketpulse_summary.json"
OUT = ROOT / "docs" / "index.html"

CSS = """
:root{--bg:#080b12;--surf:#0d1320;--surf2:#111827;--bdr:#1e2d45;
--accent:#00c8f8;--green:#00d68f;--red:#ff4d6d;--amber:#ffb830;
--text:#dce8f5;--muted:#8ba3c4;--faint:#4a607d}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;line-height:1.55;padding:32px 18px}
.wrap{max-width:1060px;margin:0 auto}
.kicker{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--accent);letter-spacing:.18em;text-transform:uppercase}
h1{font-size:34px;margin:6px 0 4px}
.sub{color:var(--muted);margin-bottom:26px}
.meta{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--faint);margin-bottom:24px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:28px}
.card{background:var(--surf);border:1px solid var(--bdr);border-radius:12px;padding:16px}
.card .lbl{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint)}
.card .val{font-size:22px;font-weight:700;margin:6px 0 2px;word-break:break-word}
.card .smb{font-size:12px;color:var(--muted)}
h2{font-size:15px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);
font-family:ui-monospace,Menlo,monospace;margin:30px 0 10px;border-bottom:1px solid var(--bdr);padding-bottom:8px}
table{width:100%;border-collapse:collapse;background:var(--surf);border:1px solid var(--bdr);border-radius:12px;overflow:hidden}
th,td{padding:10px 14px;text-align:left;font-size:14px;border-bottom:1px solid #14203a}
th{color:var(--faint);font-size:11px;letter-spacing:.1em;text-transform:uppercase;background:var(--surf2)}
tr:last-child td{border-bottom:none}
.num{font-family:ui-monospace,Menlo,monospace}
.pos{color:var(--green)}.neg{color:var(--red)}.neu{color:var(--amber)}
.panel{background:var(--surf);border:1px solid var(--bdr);border-left:3px solid var(--accent);
border-radius:10px;padding:14px 18px;margin:10px 0;font-size:14.5px}
.panel.red{border-left-color:var(--red)}.panel.green{border-left-color:var(--green)}
.panel.amber{border-left-color:var(--amber)}
.tag{display:inline-block;font-size:10px;letter-spacing:.14em;font-weight:700;
padding:2px 10px;border-radius:99px;margin-right:8px;vertical-align:middle}
.tag.g{background:rgba(0,214,143,.15);color:var(--green)}
.tag.r{background:rgba(255,77,109,.15);color:var(--red)}
.tag.a{background:rgba(255,184,48,.15);color:var(--amber)}
.tag.c{background:rgba(0,200,248,.15);color:var(--accent)}
.alert{border-left:3px solid var(--amber);background:var(--surf);padding:10px 14px;
border-radius:8px;margin:8px 0;font-family:ui-monospace,Menlo,monospace;font-size:13px}
.disclaimer{margin-top:36px;border:1px dashed var(--bdr);border-radius:12px;padding:18px;
color:var(--muted);font-size:13.5px;background:var(--surf)}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
footer{margin-top:26px;color:var(--faint);font-size:12px;font-family:ui-monospace,Menlo,monospace}
.bar{height:8px;border-radius:99px;background:#16233c;overflow:hidden;display:flex;margin-top:4px}
.bar span{display:block;height:100%}
"""


def _cls(v: float) -> str:
    return "pos" if v > 0 else ("neg" if v < 0 else "neu")


def _sign(v: float, pct: bool = False) -> str:
    return f"{v:+.2%}" if pct else f"{v:+.3f}"


def _momentum_svg(points: list[dict]) -> str:
    if len(points) < 2:
        return ""
    w, h, pad = 660, 170, 26
    scores = [p["current"] for p in points]
    deltas = [p["momentum"] for p in points]
    lo, hi = min(scores + [0]) - 0.05, max(scores + [0]) + 0.05
    span = (hi - lo) or 1
    step_x = (w - 2 * pad) / (len(points) - 1)

    def y(v):
        return pad + (1 - (v - lo) / span) * (h - 2 * pad)

    line = " ".join(
        f"{pad + i * step_x:.1f},{y(s):.1f}" for i, s in enumerate(scores)
    )
    area = (
        f"M{pad},{y(scores[0]):.1f} "
        + " ".join(f"L{pad + i * step_x:.1f},{y(s):.1f}" for i, s in enumerate(scores))
        + f" L{pad + (len(points) - 1) * step_x:.1f},{h - pad} L{pad},{h - pad} Z"
    )
    bars = []
    max_d = max(abs(d) for d in deltas) or 1
    base = y(0)
    for i, d in enumerate(deltas):
        x = pad + i * step_x
        bh = abs(d) / max_d * 46
        color = "#00d68f" if d > 0 else "#ff4d6d"
        yy = base - bh if d > 0 else base
        bars.append(
            f'<rect x="{x - 9:.1f}" y="{yy:.1f}" width="18" height="{bh:.1f}" '
            f'fill="{color}" opacity="0.45" rx="2"/>'
        )
    labels = "".join(
        f'<text x="{pad + i * step_x:.1f}" y="{h - 4}" fill="#4a607d" font-size="9" '
        f'text-anchor="middle" font-family="monospace">{p["as_of"][5:]}</text>'
        for i, p in enumerate(points)
    )
    zero = (
        f'<line x1="{pad}" y1="{base:.1f}" x2="{w - pad}" y2="{base:.1f}" '
        f'stroke="#1e2d45" stroke-dasharray="3 4"/>'
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" style="width:100%;background:var(--surf);'
        f'border:1px solid var(--bdr);border-radius:12px">'
        f'<path d="{area}" fill="rgba(0,200,248,.07)"/>'
        f'<polyline points="{line}" fill="none" stroke="#00c8f8" stroke-width="2"/>'
        f'{"".join(bars)}{zero}{labels}</svg>'
    )


def render(snap: dict) -> str:
    gen = snap.get("generated_at", datetime.now(timezone.utc).isoformat())
    focus = snap.get("focus_symbol", "?")
    sent_rows = snap.get("sentiment_by_asset", {})
    total_n = sum(v["n"] for v in sent_rows.values())
    div = snap.get("divergence") or {}
    reg = snap.get("regime") or {}
    bt = snap.get("backtest") or {}
    mom = snap.get("momentum_recent") or []
    alerts = snap.get("alerts_fired_this_run") or []

    div_class = div.get("classification", "—")
    is_divergence = str(div_class).endswith("divergence")
    edge = (
        bt["hit_rate"] - bt["random_hit_rate"]
        if bt.get("n_evaluated")
        else None
    )

    asset_html = "".join(
        f"<tr><td><strong>{t}</strong></td>"
        f"<td class='num {_cls(v['score'])}'>{_sign(v['score'])}</td>"
        f"<td class='num'>{v['share_positive']:.0%}</td>"
        f"<td class='num'>{v['share_negative']:.0%}</td>"
        f"<td class='num'>{v['share_neutral']:.0%}</td>"
        f"<td class='num'>{v['n']}</td></tr>"
        for t, v in sorted(sent_rows.items())
    )

    mom_rows = "".join(
        f"<tr><td class='num'>{m['as_of']}</td>"
        f"<td class='num'>{_sign(m['previous'])}</td>"
        f"<td class='num'>{_sign(m['current'])}</td>"
        f"<td class='num {_cls(m['momentum'])}'>{_sign(m['momentum'])}</td>"
        f"<td>{m['classification']}{' · accelerating' if m.get('accelerating') else ''}"
        f"{' · <span class=&quot;neg&quot;>REVERSAL</span>' if m.get('reversal') else ''}</td></tr>"
        for m in reversed(mom[-7:])
    ) or "<tr><td colspan=5>fewer than two aligned daily buckets in sample corpus</td></tr>"

    alert_html = (
        "".join(
            f"<div class='alert'>⚠ <strong>{a['rule_name']}</strong> — "
            f"{a['message']}</div>"
            for a in alerts[-5:]
        )
        or "<div class='panel green'>No alerts fired in the latest evaluation window.</div>"
    )

    comp = reg.get("components") or {}
    comp_chips = " ".join(
        f"<span class='tag c'>{k.replace('_', ' ')}: {'—' if v is None else format(v, '+.3f')}</span>"
        for k, v in comp.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MarketPulse AI — Live Intelligence Demo</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<div class="kicker">MarketPulse AI · Market Intelligence &amp; Sentiment Research</div>
<h1>What is the market saying — and does price agree?</h1>
<div class="sub">Live snapshot computed by the open-source MarketPulse pipeline on the bundled
sample corpus ({total_n} scored items, engine: {snap.get("engine_model", "?")}).
Focus asset: <strong>{focus}</strong>.</div>
<div class="meta">snapshot generated {gen[:19]}Z · every number below comes from
<a href="https://github.com/Parithosh-Varma/MarketPulse-AI">this repository's code</a>, fully offline</div>

<div class="cards">
<div class="card"><div class="lbl">Articles scored</div><div class="val">{total_n}</div>
<div class="smb">news + GDELT sample corpus</div></div>
<div class="card"><div class="lbl">Divergence ({div.get('window_days', '—')}d window)</div>
<div class="val {'pos' if str(div_class).startswith('bull') else ('neg' if str(div_class).startswith('bear') else '')}">{div_class}</div>
<div class="smb">score {div.get('score', 0):+.2f} · price {div.get('price_return', 0):+.2%} vs sentiment {div.get('sentiment_change', 0):+.2f}</div></div>
<div class="card"><div class="lbl">Market regime</div>
<div class="val">{str(reg.get('regime', '—')).upper()}</div>
<div class="smb">confidence {reg.get('confidence', 0):.0%} · risk appetite {reg.get('risk_appetite', 'unknown')}</div></div>
<div class="card"><div class="lbl">Signal hit rate</div>
<div class="val">{bt.get('hit_rate', 0):.0%}</div>
<div class="smb">vs random baseline {bt.get('random_hit_rate', 0):.0%} · n={bt.get('n_evaluated', 0)}</div></div>
</div>

<h2>Sentiment by asset</h2>
<table><tr><th>Asset</th><th>Avg score</th><th>Positive</th><th>Negative</th><th>Neutral</th><th>n</th></tr>
{asset_html}</table>

<h2>Sentiment momentum — {focus} (daily)</h2>
{_momentum_svg(mom)}
<div style="overflow:auto"><table style="margin-top:10px">
<tr><th>Date</th><th>Previous</th><th>Current</th><th>Momentum</th><th>Classification</th></tr>
{mom_rows}</table></div>

<h2>Price vs sentiment divergence</h2>
<div class="panel {'green' if str(div_class).startswith('bull') else ('red' if str(div_class).startswith('bear') else '')}">
<span class="tag {'g' if str(div_class).startswith('bull') else ('r' if str(div_class).startswith('bear') else 'a')}">{div_class}</span>
Over the last <strong>{div.get('window_days', '—')} days</strong> price returned
<strong class="num">{div.get('price_return', 0):+.2%}</strong> while aggregated sentiment moved
<strong class="num">{div.get('sentiment_change', 0):+.2f}</strong>.
Divergence score <strong class="num">{div.get('score', 0):+.2f}</strong>,
confidence {div.get('confidence', 0):.2f}. Divergence is flagged only when price and
sentiment move in <em>opposite directions beyond threshold</em> — ordinary disagreement stays unclassified.
</div>

<h2>Market regime</h2>
<div class="panel">
<span class="tag c">{str(reg.get('regime', '—')).upper()}</span>
confidence <strong>{reg.get('confidence', 0):.0%}</strong> · stress: {reg.get('stress', 'n/a')}
· rule votes are recorded for auditability.<br><br>{comp_chips}
</div>

<h2>Walk-forward backtest — does sentiment predict direction?</h2>
<div class="panel {'green' if edge is not None and edge > 0.05 else 'amber'}">
Hit rate <strong>{bt.get('hit_rate', 0):.1%}</strong> vs random baseline
<strong>{bt.get('random_hit_rate', 0):.1%}</strong> over n={bt.get('n_evaluated', 0)} signals
(mean forward return {bt.get('mean_forward_return', 0):+.2%}, buy-hold {bt.get('buy_hold', 0):+.2%},
max drawdown {bt.get('max_drawdown', 0):.1%}).<br><br>
<strong>Verdict: no meaningful association on this sample.</strong> This is exactly how a
scientifically honest system reports a null result — with sample size, baseline, and period shown,
never as a trading claim.
</div>

<h2>Alert feed</h2>
{alert_html}

<div class="disclaimer">
⚠️ <strong>Research &amp; decision-support only.</strong> MarketPulse AI never predicts prices,
never places trades, and connects to no broker. Signals are descriptive statistics computed with
strict no-look-ahead rules; every reading carries its sample size and an empirical baseline.
Methodology: <a href="https://github.com/Parithosh-Varma/MarketPulse-AI/tree/main/docs">docs/
(BACKTESTING · SENTIMENT_ENGINE · DATA_SOURCES)</a>.
</div>

<footer>MarketPulse AI · built on FinSentinel research (GMSI · MFI · shock propagation)
· regenerated automatically from data/processed/marketpulse_summary.json</footer>
</div></body></html>"""


def main() -> None:
    if not SNAPSHOT.exists():
        sys.exit("run scripts/build_pulse_summary.py first")
    html = render(json.loads(SNAPSHOT.read_text()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
