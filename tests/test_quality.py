from datetime import datetime, timedelta, timezone

from src.models import AggregatedSentiment, MarketObservation, SentimentObservation
from src.quality import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    build_report,
    check_aggregate_continuity,
    check_market_observations,
    check_sentiment_observations,
)

NOW = datetime.now(timezone.utc)


def sent_obs(ts=NOW, text="headline", source="news"):
    return SentimentObservation(timestamp=ts, source=source, score=0.1, text=text)


def market_obs(day_offset, close=100.0, symbol="NIFTY50"):
    close = float(close)
    return MarketObservation(
        timestamp=NOW - timedelta(days=day_offset),
        symbol=symbol,
        open=close * 0.999,
        high=close * 1.001,
        low=close * 0.998,
        close=close,
        volume=10,
    )


def agg(start_day, score, window_days=1):
    start = NOW - timedelta(days=start_day)
    end = start + timedelta(days=window_days)
    return AggregatedSentiment(
        window_start=start,
        window_end=end,
        window=end - start,
        n_observations=5,
        mean_score=score,
    )


class TestSentimentChecks:
    def test_clean_series_passes(self):
        issues = check_sentiment_observations([sent_obs(), sent_obs(text="other")])
        assert not any(i.severity == SEVERITY_ERROR for i in issues)

    def test_future_timestamps_flagged(self):
        issues = check_sentiment_observations(
            [sent_obs(ts=NOW + timedelta(hours=6))]
        )
        assert any(i.check == "future_timestamp" and i.severity == SEVERITY_ERROR for i in issues)

    def test_duplicates_flagged(self):
        same = sent_obs()
        issues = check_sentiment_observations([same, same])
        dup = next(i for i in issues if i.check == "duplicate_observations")
        assert dup.count == 1 and dup.severity == SEVERITY_WARNING

    def test_missing_text_flagged(self):
        issues = check_sentiment_observations([sent_obs(text="   ")])
        assert any(i.check == "missing_text" and i.severity == SEVERITY_ERROR for i in issues)

    def test_missing_probabilities_informational(self):
        issues = check_sentiment_observations([sent_obs()])
        missing = [i for i in issues if i.check == "missing_probabilities"]
        assert len(missing) == 1  # vader-style rows legitimately lack probs


class TestMarketChecks:
    def test_empty_series_is_error(self):
        issues = check_market_observations([])
        assert issues[0].severity == SEVERITY_ERROR

    def test_wrong_symbol_flagged(self):
        issues = check_market_observations([market_obs(1)], expected_symbol="BANKNIFTY")
        assert any(i.check == "unexpected_symbol" for i in issues)

    def test_duplicate_timestamps_flagged(self):
        row = market_obs(1)
        issues = check_market_observations([row, row])
        assert any(i.check == "duplicate_timestamps" for i in issues)

    def test_calendar_gap_flagged(self):
        issues = check_market_observations([market_obs(30), market_obs(1)])
        gap = next(i for i in issues if i.check == "timestamp_gaps")
        assert gap.severity == SEVERITY_WARNING

    def test_outliers_flagged_but_normal_vol_not(self):
        normal = [market_obs(d, 100 + (d % 3)) for d in range(20)]
        assert not any(i.check == "price_outliers" for i in check_market_observations(normal))
        spiky = normal + [market_obs(40, 5000)]
        assert any(i.check == "price_outliers" for i in check_market_observations(spiky))

    def test_stale_data_flagged_with_limit(self):
        issues = check_market_observations(
            [market_obs(d) for d in range(1, 5)], staleness_limit=timedelta(hours=12)
        )
        stale = next(i for i in issues if i.check == "stale_data")
        assert stale.context["age_days"] > 0.5


class TestAggregateContinuity:
    def test_gap_between_windows_flagged(self):
        series = [agg(10, 0.2), agg(1, 0.2)]  # 9-day hole between them
        issues = check_aggregate_continuity(series)
        assert any(i.check == "aggregate_gaps" for i in issues)

    def test_discontinuity_flagged(self):
        series = [agg(2, 0.4), agg(1, -0.6)]
        issues = check_aggregate_continuity(series)
        assert any(i.check == "sentiment_discontinuity" for i in issues)

    def test_smooth_series_quiet(self):
        series = [agg(3, 0.10), agg(2, 0.12), agg(1, 0.11)]
        assert check_aggregate_continuity(series) == []


class TestReport:
    def test_ok_property_and_summary(self):
        report = build_report("unit", 5, [])
        assert report.ok and "PASS" in report.summary()

    def test_error_fails_report(self):
        from src.quality import QualityIssue

        report = build_report(
            "unit", 5, [QualityIssue("c", SEVERITY_ERROR, "bad")]
        )
        assert not report.ok and "FAIL" in report.summary()
        assert report.to_dict()["ok"] is False
