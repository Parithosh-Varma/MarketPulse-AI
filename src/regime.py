"""Market regime engine (Phase 7) — transparent rule ladder, no ML.

Two clearly separated layers:

1. **FinSentinel legacy** (unchanged methodology): GMSI quantile stress
   regimes from scripts/regime_analysis.py (Low/Medium/High stress at the
   20/80 percentiles). Exposed here as ``classify_gmsi_stress`` for
   compatibility; the research code remains authoritative.

2. **MarketPulse regime classifier**: deterministic rules over observable
   components (price momentum, realized volatility, VIX level, sentiment,
   sentiment momentum, GMSI/MFI levels). Rules are evaluated top-down and
   the first match wins — every output records its inputs in
   ``components`` so classifications are auditable and reproducible.

Rule ladder (thresholds configurable via RegimeConfig):

    high_volatility : vix >= vix_high OR realized_vol >= vol_high
    risk_off        : gmsi >= gmsi_high OR (vix >= vix_elevated AND
                                          sentiment <= -sent_elevated)
    risk_on         : vix <= vix_low AND momentum > 0 AND sentiment > 0
    bullish         : momentum >= mom_threshold AND sentiment > 0
    bearish         : momentum <= -mom_threshold AND sentiment < 0
    transition      : |momentum| < mom_quiet AND |sentiment| < sent_quiet
    neutral         : otherwise

Confidence = share of available component votes agreeing with the chosen
regime (components absent from the rule path do not count against it).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from src.models.regime import (
    REGIME_BEARISH,
    REGIME_BULLISH,
    REGIME_HIGH_VOL,
    REGIME_NEUTRAL,
    REGIME_RISK_OFF,
    REGIME_RISK_ON,
    REGIME_TRANSITION,
    STRESS_HIGH,
    STRESS_LOW,
    STRESS_MEDIUM,
    MarketRegime,
    MarketStress,
)


@dataclass(frozen=True)
class RegimeConfig:
    vix_low: float = 15.0
    vix_elevated: float = 18.0
    vix_high: float = 25.0
    vol_high: float = 0.020  # daily realized std proxy
    mom_threshold: float = 0.02  # ±2% trend window
    mom_quiet: float = 0.005
    sent_threshold: float = 0.10
    sent_quiet: float = 0.05
    gmsi_high: float = 0.60


def classify_gmsi_stress(gmsi_value: float, p20: float, p80: float) -> str:
    """FinSentinel quantile stress buckets (scripts/regime_analysis.py)."""
    if gmsi_value <= p20:
        return STRESS_LOW
    if gmsi_value <= p80:
        return STRESS_MEDIUM
    return STRESS_HIGH


def make_stress(
    gmsi_value: float,
    as_of: datetime,
    *,
    p20: float,
    p80: float,
    percentile: Optional[float] = None,
    source: str = "fin_sentinel_gmsi",
) -> MarketStress:
    classification = classify_gmsi_stress(gmsi_value, p20, p80)
    return MarketStress(
        as_of=as_of,
        source=source,
        level=gmsi_value,
        classification=classification,
        percentile=percentile,
        metadata={"p20": p20, "p80": p80},
    )


@dataclass(frozen=True)
class RegimeInputs:
    as_of: datetime
    symbol: Optional[str] = None
    price_momentum: Optional[float] = None  # simple return over trend window
    realized_volatility: Optional[float] = None  # daily std of returns
    vix_level: Optional[float] = None  # India VIX or CBOE VIX close
    sentiment: Optional[float] = None  # aggregate score in [-1, 1]
    sentiment_momentum: Optional[float] = None
    gmsi: Optional[float] = None
    mfi: Optional[float] = None


def _vote(votes: Dict[str, bool]) -> float:
    decided = [ok for ok in votes.values() if ok is not None]
    if not decided:
        return 0.0
    return sum(1 for ok in decided if ok) / len(decided)


def _cmp(value: Optional[float], predicate) -> Optional[bool]:
    """Compare only when data exists; None otherwise (never False-by-default)."""
    if value is None:
        return None
    return predicate(value)


_DEFAULT_CONFIG = RegimeConfig()


def classify_regime(
    inputs: RegimeInputs,
    config: RegimeConfig = _DEFAULT_CONFIG,
) -> MarketRegime:
    """Run the rule ladder over ``inputs`` and produce an auditable regime."""
    c = config
    mom = inputs.price_momentum
    vol = inputs.realized_volatility
    vix = inputs.vix_level
    sent = inputs.sentiment
    sent_mom = inputs.sentiment_momentum
    gmsi = inputs.gmsi

    votes: Dict[str, Optional[bool]] = {}
    regime = REGIME_NEUTRAL

    # 1 — volatility shock dominates everything else
    votes["vix_high"] = _cmp(vix, lambda x: x >= c.vix_high)
    votes["vol_high"] = _cmp(vol, lambda x: x >= c.vol_high)
    if any(v is True for v in votes.values()):
        regime = REGIME_HIGH_VOL
        confidence = _vote({k: v for k, v in votes.items() if v is not None})
        return _build(inputs, regime, confidence, votes, c)

    # 2 — stress-driven risk-off
    riskoff_votes: Dict[str, Optional[bool]] = {
        "gmsi_high": _cmp(gmsi, lambda x: x >= c.gmsi_high),
        "vix_sent": (
            (vix >= c.vix_elevated and sent <= -c.sent_threshold)
            if vix is not None and sent is not None
            else None
        ),
    }
    if any(v is True for v in riskoff_votes.values()):
        regime = REGIME_RISK_OFF
        confidence = _vote({k: v for k, v in riskoff_votes.items() if v is not None})
        return _build(inputs, regime, max(confidence, 0.5), votes | riskoff_votes, c)

    # 3 — quiet drift-less market is a transition state regardless of VIX
    quiet_votes: Dict[str, Optional[bool]] = {
        "momentum_quiet": _cmp(mom, lambda x: abs(x) < c.mom_quiet),
        "sentiment_quiet": _cmp(sent, lambda x: abs(x) < c.sent_quiet),
    }
    active_quiet = {k: v for k, v in quiet_votes.items() if v is not None}
    if active_quiet and all(active_quiet.values()) and (
        sent_mom is None or abs(sent_mom) < c.sent_threshold
    ):
        regime = REGIME_TRANSITION
        return _build(inputs, regime, _vote(active_quiet), votes | quiet_votes, c)

    # 4 — calm-and-positive risk-on (requires calm-VIX evidence by definition)
    riskon_votes: Dict[str, Optional[bool]] = {
        "vix_low": _cmp(vix, lambda x: x <= c.vix_low),
        "momentum_up": _cmp(mom, lambda x: x > c.mom_quiet),
        "sentiment_up": _cmp(sent, lambda x: x > 0),
    }
    if all(v is True for v in riskon_votes.values()):
        regime = REGIME_RISK_ON
        return _build(
            inputs,
            regime,
            _vote({k: v for k, v in riskon_votes.items() if v is not None}),
            votes | riskon_votes,
            c,
        )

    # 5/6 — directional regimes need both momentum and sentiment agreement
    directional_votes: Dict[str, Optional[bool]] = {
        "momentum_positive": _cmp(mom, lambda x: x >= c.mom_threshold),
        "sentiment_positive": _cmp(sent, lambda x: x > c.sent_threshold),
    }
    if mom is not None and sent is not None and all(directional_votes.values()):
        regime = REGIME_BULLISH
        return _build(inputs, regime, _vote(directional_votes), votes | directional_votes, c)

    bear_votes: Dict[str, Optional[bool]] = {
        "momentum_negative": _cmp(mom, lambda x: x <= -c.mom_threshold),
        "sentiment_negative": _cmp(sent, lambda x: x < -c.sent_threshold),
    }
    if mom is not None and sent is not None and all(bear_votes.values()):
        regime = REGIME_BEARISH
        return _build(inputs, regime, _vote(bear_votes), votes | bear_votes, c)

    confidence = _vote(
        {k: v for k, v in list(votes.items()) + list(quiet_votes.items())
         if v is not None}
    )
    return _build(inputs, REGIME_NEUTRAL, max(confidence, 0.3), votes | quiet_votes, c)


def _build(
    inputs: RegimeInputs,
    regime: str,
    confidence: float,
    votes: Dict[str, Optional[bool]],
    config: RegimeConfig,
) -> MarketRegime:
    components: Dict[str, Optional[float]] = {
        "price_momentum": inputs.price_momentum,
        "realized_volatility": inputs.realized_volatility,
        "vix_level": inputs.vix_level,
        "sentiment": inputs.sentiment,
        "sentiment_momentum": inputs.sentiment_momentum,
        "gmsi": inputs.gmsi,
        "mfi": inputs.mfi,
    }

    stress_classification = None
    if inputs.gmsi is not None:
        stress_classification = classify_gmsi_stress(
            inputs.gmsi, p20=config.gmsi_high - 0.4, p80=config.gmsi_high
        )

    risk_appetite = None
    if regime == REGIME_RISK_ON:
        risk_appetite = "risk_on"
    elif regime == REGIME_RISK_OFF:
        risk_appetite = "risk_off"
    else:
        risk_appetite = "unknown"

    return MarketRegime(
        as_of=inputs.as_of,
        symbol=inputs.symbol,
        regime=regime,
        confidence=min(round(confidence, 4), 1.0),
        components=components,
        stress_classification=stress_classification,
        risk_appetite=risk_appetite,
        metadata={"rule_votes": {k: v for k, v in votes.items()}},
    )


__all__ = [
    "RegimeConfig",
    "RegimeInputs",
    "classify_regime",
    "classify_gmsi_stress",
    "make_stress",
    "MarketRegime",
    "MarketStress",
]
