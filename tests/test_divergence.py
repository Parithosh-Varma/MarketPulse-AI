from datetime import datetime, timedelta, timezone

import pytest

from src.divergence import (
    CLASS_ALIGNED_BEARISH,
    CLASS_ALIGNED_BULLISH,
    CLASS_BEARISH_DIVERGENCE,
    CLASS_BULLISH_DIVERGENCE,
    CLASS_NEUTRAL,
    compute_divergence,
    detect_divergence,
    price_return_over,
    simple_return,
)
from src.models import AggregatedSentiment, DivergenceObservation, MarketObservation

T0 = datetime(2026, 8, 20, 15, 30, tzinfo=timezone.utc)
DAY = timedelta(days=1)


def kwargs(**overrides):
    base = dict(
        symbol="NIFTY50",
        as_of=T0,
        window=timedelta(days=5),
        n_observations=50,
    )
    base.update(overrides)
    return base


class TestClassification:
    def test_textbook_bearish_divergence(self):
        out = compute_divergence(
            price_return=0.04, sentiment_change=-0.22, **kwargs()
        )
        assert out.classification == CLASS_BEARISH_DIVERGENCE
        assert out.is_divergence and out.divergence_score < 0

    def test_textbook_bullish_divergence(self):
        out = compute_divergence(
            price_return=-0.03, sentiment_change=0.19, **kwargs()
        )
        assert out.classification == CLASS_BULLISH_DIVERGENCE
        assert out.is_divergence and out.divergence_score > 0

    def test_small_disagreement_is_not_a_signal(self):
        out = compute_divergence(price_return=0.004, sentiment_change=-0.01, **kwargs())
        assert out.classification == CLASS_NEUTRAL
        assert not out.is_divergence

    def test_aligned_bullish(self):
        out = compute_divergence(price_return=0.02, sentiment_change=0.10, **kwargs())
        assert out.classification == CLASS_ALIGNED_BULLISH

    def test_aligned_bearish(self):
        out = compute_divergence(price_return=-0.02, sentiment_change=-0.10, **kwargs())
        assert out.classification == CLASS_ALIGNED_BEARISH

    def test_flat_everything_neutral(self):
        out = compute_divergence(price_return=0.0, sentiment_change=0.0, **kwargs())
        assert out.classification == CLASS_NEUTRAL

    def test_threshold_gates_classification(self):
        args = dict(price_return=0.03, sentiment_change=-0.06, **kwargs())
        loose = compute_divergence(threshold=0.1, **args)
        strict = compute_divergence(threshold=0.9, **args)
        assert loose.classification == CLASS_BEARISH_DIVERGENCE
        assert strict.classification == CLASS_NEUTRAL


class TestScoringAndConfidence:
    def test_score_bounded(self):
        out = compute_divergence(price_return=0.9, sentiment_change=-0.9, **kwargs())
        assert -1.0 <= out.divergence_score <= 1.0

    def test_confidence_grows_with_sample_size(self):
        few = compute_divergence(price_return=0.04, sentiment_change=-0.22, **kwargs(n_observations=2))
        many = compute_divergence(price_return=0.04, sentiment_change=-0.22, **kwargs(n_observations=500))
        assert many.confidence > few.confidence >= 0

    def test_extreme_volatility_dampens_confidence(self):
        calm = compute_divergence(price_return=0.04, sentiment_change=-0.22, **kwargs())
        wild = compute_divergence(
            price_return=0.04, sentiment_change=-0.22, volatility_context=0.08, **kwargs()
        )
        assert wild.confidence < calm.confidence

    def test_example_json_shape(self):
        out = compute_divergence(price_return=0.031, sentiment_change=-0.18, **kwargs())
        payload = out.to_dict()
        for key in (
            "symbol", "price_return", "sentiment_change", "divergence_score",
            "classification", "confidence", "n_observations", "window", "as_of",
        ):
            assert key in payload


class TestValidation:
    @pytest.mark.parametrize("bad_threshold", [-0.1, 1.5])
    def test_bad_threshold_rejected(self, bad_threshold):
        with pytest.raises(ValueError, match="threshold"):
            compute_divergence(price_return=0.01, sentiment_change=0.0, threshold=bad_threshold, **kwargs())

    def test_bad_scale_rejected(self):
        with pytest.raises(ValueError, match="scale"):
            compute_divergence(price_return=0.01, sentiment_change=0.0, price_scale=0, **kwargs())

    def test_bad_classification_in_schema(self):
        with pytest.raises(ValueError, match="classification"):
            DivergenceObservation(
                symbol="X", as_of=T0, window=DAY, price_return=0.0,
                sentiment_change=0.0, divergence_score=0.0,
                classification="moonshot", confidence=0.0, n_observations=1,
            )

    def test_round_trip(self):
        out = compute_divergence(price_return=0.04, sentiment_change=-0.22, volatility_context=0.01, **kwargs())
        assert DivergenceObservation.from_json(out.to_json()) == out


def make_price(close, day_offset):
    return MarketObservation(
        timestamp=T0 + DAY * (day_offset - 10),
        symbol="NIFTY50",
        open=close * 0.999,
        high=close * 1.001,
        low=close * 0.998,
        close=close,
        volume=1000,
    )


def make_agg(day_offset, score, weighted=None):
    end = T0 + DAY * (day_offset - 10) + DAY
    return AggregatedSentiment(
        window_start=end - DAY,
        window_end=end,
        window=DAY,
        n_observations=20,
        mean_score=score,
        weighted_score=weighted,
        ticker="NIFTY50",
    )


class TestHelpers:
    def test_simple_return(self):
        assert simple_return(100.0, 103.1) == pytest.approx(0.031)

    def test_simple_return_rejects_zero_base(self):
        with pytest.raises(ValueError):
            simple_return(0.0, 10.0)

    def test_price_return_over(self):
        prices = [make_price(100.0 + i, i) for i in range(6)]
        ret, as_of = price_return_over(prices, lookback_bars=3)
        assert ret == pytest.approx(prices[-1].close / prices[-4].close - 1)
        assert as_of == prices[-1].timestamp

    def test_price_return_over_insufficient_history(self):
        with pytest.raises(ValueError):
            price_return_over([make_price(100, 0)], lookback_bars=3)


class TestDetectDivergence:
    def build(self, closes=(100, 101, 102, 103, 104), scores=(0.0, 0.05)):
        prices = [make_price(c, i) for i, c in enumerate(closes)]
        aggs = [make_agg(i, s, weighted=s) for i, s in enumerate(scores)]
        return prices, aggs

    def test_bearish_case_end_to_end(self):
        # price grinds up, sentiment collapses over the same span
        prices, aggs = self.build(closes=(100, 102, 104, 106, 108), scores=(0.30, -0.20))
        out = detect_divergence(prices, aggs, symbol="NIFTY50")
        assert out is not None and out.is_divergence
        assert out.classification == CLASS_BEARISH_DIVERGENCE
        assert out.price_return > 0 and out.sentiment_change < 0

    def test_returns_none_without_enough_data(self):
        prices, aggs = self.build()
        assert detect_divergence(prices[:1], aggs, symbol="NIFTY50") is None
        assert detect_divergence(prices, aggs[:1], symbol="NIFTY50") is None

    def test_only_uses_aggregates_before_as_of(self):
        prices = [make_price(c, i) for i, c in enumerate((100, 102, 104, 106, 108))]
        aggs = [
            make_agg(0, 0.30, weighted=0.30),
            make_agg(1, -0.20, weighted=-0.20),
            make_agg(99, 0.99, weighted=0.99),
        ]
        out = detect_divergence(prices, aggs, symbol="NIFTY50")
        assert out.sentiment_change == pytest.approx(-0.50)
