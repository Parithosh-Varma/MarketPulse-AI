"""Alerting engine (Phase 10) — deterministic rules, pluggable channels.

Rules are evaluated against a plain ``context`` mapping produced by the
analytics layer (momentum, divergence, regime change, VIX). Every fired
alert is returned and dispatched to configured channels. Default channel
logs; a file channel persists JSONL for the dashboard. External delivery
(email/Telegram/webhook) is intentionally NOT implemented until credentials
are provided by the operator.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"


@dataclass(frozen=True)
class Alert:
    rule_name: str
    severity: str
    message: str
    fired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["fired_at"] = self.fired_at.isoformat()
        return data


class AlertChannel(ABC):
    @abstractmethod
    def deliver(self, alert: Alert) -> None:
        """Transport one alert somewhere."""


class LogChannel(AlertChannel):
    def deliver(self, alert: Alert) -> None:
        log = logger.warning if alert.severity != SEVERITY_HIGH else logger.error
        log("ALERT [%s] %s: %s", alert.severity, alert.rule_name, alert.message)


class JsonlFileChannel(AlertChannel):
    """Append-only JSONL sink the dashboard can tail."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def deliver(self, alert: Alert) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(alert.to_dict()) + "\n")


# ── rules ────────────────────────────────────────────────────────────────────

def check_sentiment_drop(context: Dict[str, Any], *, threshold: float = -0.30) -> Optional[Alert]:
    momentum = context.get("sentiment_momentum")
    if momentum is not None and momentum <= threshold:
        return Alert(
            rule_name="sentiment_drop",
            severity=SEVERITY_MEDIUM,
            message=f"Sentiment moved {momentum:+.2f} within its window (<= {threshold})",
            context={"sentiment_momentum": momentum, "threshold": threshold},
        )
    return None


def check_extreme_divergence(
    context: Dict[str, Any], *, min_abs_score: float = 0.60
) -> Optional[Alert]:
    score = context.get("divergence_score")
    classification = context.get("divergence_classification")
    if (
        score is not None
        and abs(score) >= min_abs_score
        and classification in ("bearish_divergence", "bullish_divergence")
    ):
        return Alert(
            rule_name="extreme_divergence",
            severity=SEVERITY_MEDIUM,
            message=f"Extreme {classification} detected (score {score:+.2f})",
            context={
                "divergence_score": score,
                "classification": classification,
                "threshold": min_abs_score,
            },
        )
    return None


def check_regime_change(context: Dict[str, Any]) -> Optional[Alert]:
    previous = context.get("previous_regime")
    current = context.get("current_regime")
    if previous and current and previous != current:
        return Alert(
            rule_name="regime_change",
            severity=SEVERITY_HIGH,
            message=f"Market regime changed: {previous} -> {current}",
            context={"previous_regime": previous, "current_regime": current},
        )
    return None


def check_vix_spike(
    context: Dict[str, Any],
    *,
    baseline: float = 15.0,
    multiplier: float = 1.4,
) -> Optional[Alert]:
    vix = context.get("vix_level")
    base = context.get("vix_baseline", baseline)
    if vix is not None and base and base > 0 and vix >= base * multiplier:
        return Alert(
            rule_name="vix_spike",
            severity=SEVERITY_HIGH,
            message=f"VIX at {vix:.1f} vs baseline {base:.1f} (>= x{multiplier})",
            context={"vix_level": vix, "baseline": base, "multiplier": multiplier},
        )
    return None


DEFAULT_RULES = {
    "sentiment_drop": check_sentiment_drop,
    "extreme_divergence": check_extreme_divergence,
    "regime_change": check_regime_change,
    "vix_spike": check_vix_spike,
}


class AlertEngine:
    """Evaluate enabled rules against a context and fan out to channels."""

    def __init__(
        self,
        channels: Optional[List[AlertChannel]] = None,
        rules: Optional[List[str]] = None,
        cooldown_seconds: int = 3600,
        clock: Any = datetime.now,
    ) -> None:
        self.channels = channels if channels is not None else [LogChannel()]
        self.rules = {k: DEFAULT_RULES[k] for k in (rules or list(DEFAULT_RULES))}
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._last_fired: Dict[str, datetime] = {}

    def process(self, context: Dict[str, Any]) -> List[Alert]:
        now = self._now()
        fired: List[Alert] = []
        for name, rule in self.rules.items():
            try:
                alert = rule(context)
            except Exception:  # noqa: BLE001 - one bad rule must not stop others
                logger.exception("alert rule %s crashed", name)
                continue
            if alert is None:
                continue
            last = self._last_fired.get(name)
            if last is not None and (now - last).total_seconds() < self.cooldown_seconds:
                continue
            self._last_fired[name] = now
            fired.append(alert)
        for alert in fired:
            for channel in self.channels:
                channel.deliver(alert)
        return fired

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value


__all__ = [
    "Alert",
    "AlertChannel",
    "AlertEngine",
    "JsonlFileChannel",
    "LogChannel",
    "SEVERITY_HIGH",
    "SEVERITY_LOW",
    "SEVERITY_MEDIUM",
]
