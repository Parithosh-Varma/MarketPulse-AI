"""Reusable data-quality checks for observations and series (§16).

Every check returns structured :class:`QualityIssue` records instead of
raising, so pipelines can aggregate a report, log it, and decide policy
(discard / keep / fail) explicitly. Nothing is ever silently dropped here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.models import AggregatedSentiment, MarketObservation, SentimentObservation

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"


@dataclass(frozen=True)
class QualityIssue:
    check: str
    severity: str
    message: str
    count: int = 1
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "count": self.count,
            **self.context,
        }


@dataclass(frozen=True)
class QualityReport:
    source: str
    n_records: int
    issues: List[QualityIssue]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def ok(self) -> bool:
        return not any(i.severity == SEVERITY_ERROR for i in self.issues)

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        errors = sum(1 for i in self.issues if i.severity == SEVERITY_ERROR)
        warnings = sum(1 for i in self.issues if i.severity == SEVERITY_WARNING)
        return (
            f"[{status}] {self.source}: {self.n_records} records, "
            f"{errors} error(s), {warnings} warning(s)"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "n_records": self.n_records,
            "ok": self.ok,
            "generated_at": self.generated_at.isoformat(),
            "issues": [i.to_dict() for i in self.issues],
        }


def _dupes(iterable: Iterable[Any]) -> int:
    seen: set = set()
    dupes = 0
    for item in iterable:
        if item in seen:
            dupes += 1
        else:
            seen.add(item)
    return dupes


def check_sentiment_observations(
    rows: Sequence[SentimentObservation],
    *,
    max_future_skew: timedelta = timedelta(minutes=5),
) -> List[QualityIssue]:
    issues: List[QualityIssue] = []
    now = datetime.now(timezone.utc)

    future = [r for r in rows if r.timestamp > now + max_future_skew]
    if future:
        issues.append(
            QualityIssue(
                check="future_timestamp",
                severity=SEVERITY_ERROR,
                message="observations stamped in the future",
                count=len(future),
                context={"latest": max(r.timestamp for r in future).isoformat()},
            )
        )

    tz_naive_source_keys = [
        r.source for r in rows if "(naive)" in (r.metadata.get("tz_note") or "")
    ]
    if tz_naive_source_keys:
        issues.append(
            QualityIssue(
                check="timezone_assumed_utc",
                severity=SEVERITY_WARNING,
                message="some sources supplied naive timestamps (assumed UTC)",
                count=len(tz_naive_source_keys),
            )
        )

    key = lambda r: (r.source, r.text, r.timestamp)  # noqa: E731
    duplicate_count = _dupes(map(key, rows))
    if duplicate_count:
        issues.append(
            QualityIssue(
                check="duplicate_observations",
                severity=SEVERITY_WARNING,
                message="duplicate (source, text, timestamp) rows",
                count=duplicate_count,
            )
        )

    missing_text = [r for r in rows if not (r.text or "").strip()]
    if missing_text:
        issues.append(
            QualityIssue(
                check="missing_text",
                severity=SEVERITY_ERROR,
                message="scored rows must carry their text payload",
                count=len(missing_text),
            )
        )

    no_probs = [r for r in rows if r.prob_positive is None and r.model_name != "vader"]
    if no_probs:
        issues.append(
            QualityIssue(
                check="missing_probabilities",
                severity=SEVERITY_INFO,
                message="non-VADER rows without class probabilities",
                count=len(no_probs),
            )
        )
    return issues


def check_market_observations(
    rows: Sequence[MarketObservation],
    *,
    expected_symbol: Optional[str] = None,
    staleness_limit: Optional[timedelta] = None,
    outlier_z: float = 10.0,
) -> List[QualityIssue]:
    issues: List[QualityIssue] = []
    if not rows:
        return [
            QualityIssue(
                check="empty_series",
                severity=SEVERITY_ERROR,
                message="no market observations provided",
            )
        ]

    if expected_symbol:
        foreign = {r.symbol for r in rows} - {expected_symbol}
        if foreign:
            issues.append(
                QualityIssue(
                    check="unexpected_symbol",
                    severity=SEVERITY_ERROR,
                    message=f"rows carry symbols outside {expected_symbol!r}",
                    count=len(foreign),
                    context={"symbols": sorted(foreign)},
                )
            )

    timestamps = [r.timestamp for r in rows]
    duplicate_count = _dupes(timestamps)
    if duplicate_count:
        issues.append(
            QualityIssue(
                check="duplicate_timestamps",
                severity=SEVERITY_ERROR,
                message="multiple bars share one timestamp",
                count=duplicate_count,
            )
        )

    gaps = [
        (prev_ts, next_ts)
        for prev_ts, next_ts in zip(sorted(timestamps), sorted(timestamps)[1:])
        if (next_ts - prev_ts) > timedelta(days=5)
    ]
    if gaps:
        issues.append(
            QualityIssue(
                check="timestamp_gaps",
                severity=SEVERITY_WARNING,
                message="calendar gaps larger than 5 days",
                count=len(gaps),
                context={"first_gap": f"{gaps[0][0].date()}..{gaps[0][1].date()}"},
            )
        )

    closes = [float(r.close) for r in rows]
    mean_close = sum(closes) / len(closes)
    var = sum((c - mean_close) ** 2 for c in closes) / len(closes)
    std = var ** 0.5
    if std > 0:
        outliers = [
            c for c in closes if abs((c - mean_close) / std) > outlier_z
        ]
        if outliers:
            issues.append(
                QualityIssue(
                    check="price_outliers",
                    severity=SEVERITY_WARNING,
                    message=f"closes beyond {outlier_z} sigma of series mean",
                    count=len(outliers),
                )
            )

    zero_or_negative = [c for c in closes if c <= 0]
    if zero_or_negative:
        issues.append(
            QualityIssue(
                check="invalid_prices",
                severity=SEVERITY_ERROR,
                message="non-positive closes present (schema should block these)",
                count=len(zero_or_negative),
            )
        )

    if staleness_limit is not None:
        newest = max(timestamps)
        age = datetime.now(timezone.utc) - newest
        if age > staleness_limit:
            issues.append(
                QualityIssue(
                    check="stale_data",
                    severity=SEVERITY_WARNING,
                    message=f"newest bar older than {staleness_limit}",
                    count=1,
                    context={"age_days": round(age.total_seconds() / 86400, 1)},
                )
            )
    return issues


def check_aggregate_continuity(
    aggregates: Sequence[AggregatedSentiment],
    *,
    max_gap_multiplier: float = 3.0,
) -> List[QualityIssue]:
    """Flag broken cadence and score discontinuities in an aggregate series."""
    issues: List[QualityIssue] = []
    if len(aggregates) < 2:
        return issues

    ordered = sorted(aggregates, key=lambda a: a.window_start)
    window = ordered[0].window
    gaps = [
        (prev.window_end, curr.window_start)
        for prev, curr in zip(ordered, ordered[1:])
        if curr.window_start - prev.window_end > window * max_gap_multiplier
    ]
    if gaps:
        issues.append(
            QualityIssue(
                check="aggregate_gaps",
                severity=SEVERITY_WARNING,
                message="aggregate windows missing between points",
                count=len(gaps),
            )
        )

    jumps = [
        (a.window_end, b.window_start)
        for a, b in zip(ordered, ordered[1:])
        if abs(
            (b.mean_score if b.mean_score is not None else 0.0)
            - (a.mean_score if a.mean_score is not None else 0.0)
        )
        > 0.9
    ]
    if jumps:
        issues.append(
            QualityIssue(
                check="sentiment_discontinuity",
                severity=SEVERITY_WARNING,
                message="mean score jumped by >0.9 between adjacent windows",
                count=len(jumps),
            )
        )
    return issues


def build_report(source: str, n_records: int, issues: List[QualityIssue]) -> QualityReport:
    """Convenience wrapper so pipelines log one object per dataset."""
    report = QualityReport(source=source, n_records=n_records, issues=issues)
    if not report.ok:
        # errors surface loudly; warnings stay visible without failing runs
        import logging

        logging.getLogger(__name__).warning(report.summary())
    return report


__all__ = [
    "QualityIssue",
    "QualityReport",
    "SEVERITY_ERROR",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "build_report",
    "check_aggregate_continuity",
    "check_market_observations",
    "check_sentiment_observations",
]
