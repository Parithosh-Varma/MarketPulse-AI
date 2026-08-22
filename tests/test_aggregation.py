from datetime import datetime, timedelta, timezone

import pytest

from src.aggregation import (
    AggregationError,
    aggregate,
    aggregate_by_source,
    aggregate_by_ticker,
    aggregate_by_time_window,
    classify,
    floor_to_window,
)
from src.models import SentimentObservation


def obs(score, source="news", ticker="NIFTY", minutes_after_start=0, confidence=None):
    return SentimentObservation(
        timestamp=datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc)
        + timedelta(minutes=minutes_after_start),
        source=source,
        score=score,
        ticker=ticker,
        confidence=confidence,
        model_name="mock",
    )


class TestAggregate:
    def test_basic_stats(self):
        agg = aggregate([obs(1.0), obs(-0.5), obs(0.5)])
        assert agg.n_observations == 3
        assert agg.mean_score == pytest.approx(1 / 3)
        assert agg.median_score == pytest.approx(0.5)

    def test_window_bounds_from_data(self):
        agg = aggregate([obs(0.2, minutes_after_start=0), obs(-0.2, minutes_after_start=30)])
        assert agg.window == timedelta(minutes=30)
        assert agg.window_start < agg.window_end

    def test_class_shares(self):
        agg = aggregate([obs(0.8), obs(0.4), obs(-0.6), obs(0.0)])
        assert agg.share_positive == pytest.approx(0.5)
        assert agg.share_negative == pytest.approx(0.25)
        assert agg.share_neutral == pytest.approx(0.25)
        assert agg.share_positive + agg.share_negative + agg.share_neutral == pytest.approx(1.0)

    def test_weighted_vs_unweighted(self):
        observations = [obs(0.9, source="reuters"), obs(-0.6, source="random_blog")]
        plain = aggregate(observations)
        weighted = aggregate(
            observations,
            source_weights={"reuters": 10.0, "random_blog": 0.1},
            default_source_weight=1.0,
        )
        assert weighted.weighted_score > plain.mean_score
        # heavy reuters weight pulls the weighted score near reuters' own
        assert weighted.weighted_score > 0.7

    def test_unknown_sources_use_default_weight(self):
        agg = aggregate(
            [obs(0.5, source="mystery")],
            source_weights={"reuters": 5.0},
        )
        assert agg.weighted_score == pytest.approx(0.5)

    def test_invalid_weight_rejected(self):
        with pytest.raises(ValueError, match="weight"):
            aggregate([obs(0.0)], source_weights={"x": 0})

    def test_confidence_mean_ignores_none(self):
        agg = aggregate([obs(0.5, confidence=0.8), obs(0.5), obs(-0.2, confidence=1.0)])
        assert agg.mean_confidence == pytest.approx(0.9)

    def test_no_confidences_at_all(self):
        assert aggregate([obs(0.5)]).mean_confidence is None

    def test_empty_input_rejected(self):
        with pytest.raises(AggregationError):
            aggregate([])

    def test_single_observation(self):
        agg = aggregate([obs(0.25)])
        assert agg.n_observations == 1 and agg.mean_score == pytest.approx(0.25)


class TestGrouping:
    def test_by_ticker(self):
        out = aggregate_by_ticker(
            [
                obs(0.5, ticker="NIFTY"),
                obs(-0.5, ticker="NIFTY"),
                obs(1.0, ticker="BANKNIFTY"),
            ]
        )
        assert set(out) == {"NIFTY", "BANKNIFTY"}
        assert out["NIFTY"].n_observations == 2
        assert out["BANKNIFTY"].ticker == "BANKNIFTY"

    def test_by_ticker_untagged_bucket(self):
        untagged = SentimentObservation(
            timestamp=datetime(2026, 8, 20, tzinfo=timezone.utc),
            source="s",
            score=0.1,
        )
        out = aggregate_by_ticker([untagged])
        assert list(out) == ["_untagged"]
        assert out["_untagged"].ticker is None

    def test_by_source(self):
        out = aggregate_by_source([obs(0.3, source="a"), obs(0.7, source="b")])
        assert set(out) == {"a", "b"}
        assert out["b"].source == "b"

    def test_grouping_empty_rejected(self):
        with pytest.raises(AggregationError):
            aggregate_by_ticker([])


class TestTimeWindows:
    def test_floor_to_hour_grid(self):
        ts = datetime(2026, 8, 20, 9, 47, 13, tzinfo=timezone.utc)
        assert floor_to_window(ts, timedelta(hours=1)) == datetime(
            2026, 8, 20, 9, 0, tzinfo=timezone.utc
        )

    def test_floor_handles_naive_ts_as_utc(self):
        ts = datetime(2026, 8, 20, 9, 47)
        assert floor_to_window(ts, timedelta(hours=1)).tzinfo == timezone.utc

    def test_floor_rejects_nonpositive_window(self):
        with pytest.raises(ValueError):
            floor_to_window(datetime.now(timezone.utc), timedelta(0))

    def test_one_day_buckets(self):
        day1 = obs(0.5, minutes_after_start=0)      # 2026-08-20 09:00 UTC
        day2 = obs(-0.5, minutes_after_start=60 * 24)  # 2026-08-21 09:00 UTC
        out = aggregate_by_time_window([day1, day2], timedelta(days=1))
        assert len(out) == 2
        assert out[0].window_start.date() != out[1].window_start.date()
        assert out[0].window == timedelta(days=1)
        assert out[0].window_end - out[0].window_start == timedelta(days=1)

    def test_fifteen_minute_buckets(self):
        observations = [obs(0.2, minutes_after_start=m) for m in (0, 5, 16, 31)]
        out = aggregate_by_time_window(observations, timedelta(minutes=15))
        assert len(out) == 3
        assert out[0].n_observations == 2
        assert out[1].n_observations == 1

    def test_min_observations_filters(self):
        observations = [obs(0.2, minutes_after_start=m) for m in (0, 5, 16)]
        out = aggregate_by_time_window(
            observations, timedelta(minutes=15), min_observations=2
        )
        assert len(out) == 1
        assert out[0].n_observations == 2

    def test_all_filtered_returns_empty_list(self):
        out = aggregate_by_time_window([obs(0.0)], timedelta(hours=1), min_observations=5)
        assert out == []


class TestSerialization:
    def test_round_trip(self):
        from src.models import AggregatedSentiment

        agg = aggregate([obs(0.5, confidence=0.9)])
        restored = AggregatedSentiment.from_json(agg.to_json())
        assert restored == agg

    def test_validation_bad_shares(self):
        from src.models import AggregatedSentiment

        with pytest.raises(ValueError, match="shares"):
            AggregatedSentiment(
                window_start=datetime(2026, 1, 1),
                window_end=datetime(2026, 1, 2),
                window=timedelta(days=1),
                n_observations=3,
                mean_score=0.1,
                share_positive=0.5,
                share_negative=0.5,
                share_neutral=0.5,
            )


class TestClassify:
    @pytest.mark.parametrize(
        "score,expected",
        [(0.001, "positive"), (-0.001, "negative"), (0.0, "neutral"), (1e-12, "neutral")],
    )
    def test_classification(self, score, expected):
        assert classify(score) == expected
