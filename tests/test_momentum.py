from datetime import datetime, timedelta, timezone

import pytest

from src.aggregation import aggregate
from src.models import AggregatedSentiment, SentimentObservation
from src.models.momentum import (
    CLASS_DETERIORATING,
    CLASS_IMPROVING,
    CLASS_STABLE,
)
from src.momentum import MomentumError, momentum, momentum_series

T0 = datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc)


def agg_at(hour, score, ticker="NIFTY", weighted=None):
    start = T0 + timedelta(hours=hour - 1)
    end = T0 + timedelta(hours=hour)
    return AggregatedSentiment(
        window_start=start,
        window_end=end,
        window=timedelta(hours=1),
        n_observations=5,
        mean_score=score,
        weighted_score=weighted,
        mean_confidence=0.8,
        share_positive=0.5,
        share_negative=0.25,
        share_neutral=0.25,
        ticker=ticker,
    )


class TestMomentumBasics:
    def test_simple_improvement(self):
        m = momentum([agg_at(1, 0.10), agg_at(2, 0.30)])
        assert m.momentum == pytest.approx(0.20)
        assert m.classification == CLASS_IMPROVING
        assert m.rate_per_hour == pytest.approx(0.20)

    def test_deterioration(self):
        m = momentum([agg_at(1, 0.40), agg_at(2, 0.05)])
        assert m.momentum == pytest.approx(-0.35)
        assert m.classification == CLASS_DETERIORATING

    def test_stable_within_threshold(self):
        m = momentum([agg_at(1, 0.20), agg_at(2, 0.25)])
        assert abs(m.momentum) < m.threshold
        assert m.classification == CLASS_STABLE

    def test_threshold_is_configurable(self):
        series = [agg_at(1, 0.00), agg_at(2, 0.08)]
        default = momentum(series)
        strict = momentum(series, threshold=0.05)
        assert default.classification == CLASS_STABLE
        assert strict.classification == CLASS_IMPROVING

    def test_weighted_preferred_over_mean(self):
        m = momentum([agg_at(1, 0.0, weighted=0.10), agg_at(2, 0.0, weighted=0.50)])
        assert m.previous_score == pytest.approx(0.10)
        assert m.current_score == pytest.approx(0.50)
        assert m.metadata["score_basis"] == "weighted_score"

    def test_falls_back_to_mean(self):
        a = agg_at(1, 0.10)
        b = agg_at(2, 0.30)
        plain_a = AggregatedSentiment(**{**a.to_dict(), "weighted_score": None})
        plain_b = AggregatedSentiment(**{**b.to_dict(), "weighted_score": None})
        m = momentum([plain_a, plain_b])
        assert m.metadata["score_basis"] == "mean_score"
        assert m.momentum == pytest.approx(0.20)

    def test_ticker_inherited(self):
        assert momentum([agg_at(1, 0.1), agg_at(2, 0.3)]).ticker == "NIFTY"

    def test_explicit_ticker_fills_missing(self):
        a = AggregatedSentiment(**{**agg_at(1, 0.1).to_dict(), "ticker": None})
        b = AggregatedSentiment(**{**agg_at(2, 0.3).to_dict(), "ticker": None})
        assert momentum([a, b], ticker="BTC").ticker == "BTC"

    def test_needs_two_points(self):
        with pytest.raises(MomentumError):
            momentum([agg_at(1, 0.5)])


class TestSeriesFlags:
    def test_acceleration_detected(self):
        # rising, and rising faster each step
        out = momentum_series(
            [agg_at(1, 0.00), agg_at(2, 0.15), agg_at(3, 0.45)]
        )
        assert all(m.classification == CLASS_IMPROVING for m in out)
        assert out[1].accelerating is True
        assert out[0].accelerating is None

    def test_no_acceleration_when_slowing(self):
        out = momentum_series([agg_at(1, 0.00), agg_at(2, 0.30), agg_at(3, 0.40)])
        assert out[1].accelerating is False

    def test_reversal_detected(self):
        # sharp improvement then sharp deterioration
        out = momentum_series([agg_at(1, 0.10), agg_at(2, 0.50), agg_at(3, 0.05)])
        last = out[-1]
        assert last.reversal is True
        assert last.classification == CLASS_DETERIORATING

    def test_small_wiggle_is_not_reversal(self):
        out = momentum_series([agg_at(1, 0.10), agg_at(2, 0.13), agg_at(3, 0.09)])
        assert all(m.reversal is False or m.reversal is None for m in out)

    def test_series_length(self):
        out = momentum_series([agg_at(h, 0.1) for h in (1, 2, 3, 4)])
        assert len(out) == 3


class TestFromRawObservations:
    def test_pipeline_aggregate_then_momentum(self):
        def make(score, hour_offset):
            return SentimentObservation(
                timestamp=T0 + timedelta(hours=hour_offset),
                source="news",
                score=score,
                ticker="NIFTY",
            )

        day1 = aggregate([make(s, 0) for s in (0.1, 0.2, 0.15)])
        day2 = aggregate([make(s, 24) for s in (0.4, 0.5, 0.45)])
        m = momentum([day1, day2], threshold=0.05)
        assert m.classification == CLASS_IMPROVING
        assert m.elapsed >= timedelta(hours=23)


class TestSchemaValidation:
    def test_rejects_backwards_time(self):
        from src.models import SentimentMomentum

        with pytest.raises(ValueError, match="precede"):
            SentimentMomentum(
                ticker="X",
                previous_window_end=T0,
                current_window_end=T0 - timedelta(hours=1),
                window=timedelta(hours=1),
                previous_score=0.0,
                current_score=0.1,
                classification=CLASS_IMPROVING,
                threshold=0.05,
            )

    def test_rejects_bad_classification(self):
        from src.models import SentimentMomentum

        with pytest.raises(ValueError, match="classification"):
            SentimentMomentum(
                ticker="X",
                previous_window_end=T0,
                current_window_end=T0 + timedelta(hours=1),
                window=timedelta(hours=1),
                previous_score=0.0,
                current_score=0.1,
                classification="moonshot",
                threshold=0.05,
            )

    def test_round_trip(self):
        from src.models import SentimentMomentum

        m = momentum_series([agg_at(1, 0.0), agg_at(2, 0.3)])[0]
        restored = SentimentMomentum.from_json(m.to_json())
        assert restored == m
        assert restored.momentum == pytest.approx(m.momentum)
