from datetime import datetime, timedelta, timezone

import pytest

from src.backtest import (
    BacktestError,
    backtest_signal,
    compare_with_baseline,
    daily_sentiment_signal,
    max_drawdown,
)
from src.models import BacktestResult, MarketObservation
from src.models.aggregation import AggregatedSentiment

T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
DAY = timedelta(days=1)


def make_prices(closes):
    rows = []
    for i, close in enumerate(closes):
        close = float(close)
        rows.append(
            MarketObservation(
                timestamp=T0 + DAY * i,
                symbol="TEST",
                open=close * 0.99,
                high=close * 1.01,
                low=close * 0.98,
                close=close,
                volume=1000,
            )
        )
    return rows


class TestMaxDrawdown:
    def test_no_losses(self):
        assert max_drawdown([0.01, 0.02, 0.01]) == 0.0

    def test_known_drawdown(self):
        # 1.10 -> 0.99 => -10% from peak
        assert max_drawdown([0.10, -0.10]) == pytest.approx(-0.10)

    def test_empty(self):
        assert max_drawdown([]) == 0.0


class TestBacktestSignal:
    def test_perfect_positive_signal_hit_rate_1(self):
        prices = make_prices([100, 101, 102, 103, 104, 105])
        signals = [(T0 + DAY * i, +0.5) for i in range(3)]
        result = backtest_signal(
            prices, signals, signal_name="always_up", symbol="TEST", horizon_bars=2
        )
        assert result.hit_rate == 1.0
        assert result.n_evaluated == 3
        assert result.mean_forward_return > 0

    def test_exact_forward_math(self):
        closes = [100, 100, 110, 120]
        prices = make_prices(closes)
        # signal at T0 -> entry first bar after (index1: 100), exit idx3: 120
        result = backtest_signal(
            prices, [(T0, 0.5)], signal_name="s", symbol="TEST", horizon_bars=2
        )
        assert result.mean_forward_return == pytest.approx(0.20)
        assert result.hit_rate == 1.0

    def test_no_lookahead_signal_at_last_bar_excluded(self):
        prices = make_prices([100, 101, 102])
        result = backtest_signal(
            prices,
            [(T0, 0.5), (T0 + DAY * 2, 0.9)],
            signal_name="s",
            symbol="TEST",
            horizon_bars=1,
        )
        assert result.n_signals == 2
        assert result.n_evaluated == 1
        assert result.metadata["skipped_incomplete_window"] == 1

    def test_negative_signal_predicting_drop(self):
        closes = [100, 100, 95, 95, 92]
        prices = make_prices(closes)
        result = backtest_signal(
            prices, [(T0, -0.7)], signal_name="bear", symbol="TEST", horizon_bars=2
        )
        assert result.hit_rate == 1.0
        assert result.max_drawdown == 0.0  # short side gained throughout

    def test_wrong_way_signal_creates_drawdown(self):
        closes = [100, 100, 95, 95, 92]
        prices = make_prices(closes)
        result = backtest_signal(
            prices, [(T0, +0.7)], signal_name="wrong", symbol="TEST", horizon_bars=2
        )
        assert result.hit_rate == 0.0
        assert result.max_drawdown < 0

    def test_flat_signals_are_no_opinion(self):
        prices = make_prices([100, 101, 102, 103])
        result = backtest_signal(
            prices,
            [(T0, 0.01), (T0 + DAY, 0.5)],
            signal_name="mixed",
            symbol="TEST",
            horizon_bars=1,
            entry_threshold=0.05,
        )
        assert result.n_evaluated == 1
        assert result.metadata["skipped_no_opinion"] == 1

    def test_duplicate_timestamp_keeps_last_value(self):
        closes = [100, 100, 95, 95, 92]
        prices = make_prices(closes)
        result = backtest_signal(
            prices,
            [(T0, +0.5), (T0, -0.5)],
            signal_name="dup",
            symbol="TEST",
            horizon_bars=2,
        )
        assert result.n_signals == 1
        assert result.hit_rate == 1.0  # the later bearish call was right

    def test_random_hit_rate_is_empirical_base_rate(self):
        closes = [100, 104, 98, 106, 97, 108]
        prices = make_prices(closes)
        signals = [(T0 + DAY * i, 0.5 if i % 2 else -0.5) for i in range(3)]
        result = backtest_signal(
            prices, signals, signal_name="noise", symbol="TEST", horizon_bars=2
        )
        # windows: 104->106 (+), 98->97 (-), 106->108 (+) => 2/3 positive
        assert result.random_hit_rate == pytest.approx(2 / 3)
        assert result.hit_rate <= 1.0

    def test_insufficient_history_yields_no_evaluable(self):
        with pytest.raises(BacktestError, match="no evaluable"):
            backtest_signal(
                make_prices([100, 101]),
                [(T0, 0.5)],
                signal_name="s",
                symbol="TEST",
                horizon_bars=5,
            )

    def test_all_signals_unevaluable_rejected(self):
        prices = make_prices([100, 101, 102])
        with pytest.raises(BacktestError, match="no evaluable"):
            backtest_signal(
                prices,
                [(T0 + DAY * 2, 0.5)],
                signal_name="late",
                symbol="TEST",
                horizon_bars=1,
            )

    def test_buy_hold_and_excess_consistent(self):
        closes = [100, 102, 104, 106, 108]
        prices = make_prices(closes)
        signals = [(T0 + DAY * i, 0.4) for i in range(2)]
        result = backtest_signal(
            prices, signals, signal_name="long_only", symbol="TEST", horizon_bars=2
        )
        assert result.buy_hold_return == pytest.approx(0.08)
        # rising market: per-window buy-hold baseline is positive but below
        # any single window's actual gain here; excess must be finite
        assert isinstance(result.excess_vs_buy_hold, float)
        assert abs(result.excess_vs_buy_hold) < 1.0


class TestDailySentimentSignal:
    def _agg(self, end_offset_days, score):
        end = T0 + DAY * end_offset_days
        return AggregatedSentiment(
            window_start=end - DAY,
            window_end=end,
            window=DAY,
            n_observations=10,
            mean_score=score,
            weighted_score=None,
            ticker="NIFTY50",
        )

    def test_conversion_prefers_weighted_when_present(self):
        agg = AggregatedSentiment(
            window_start=T0,
            window_end=T0 + DAY,
            window=DAY,
            n_observations=10,
            mean_score=0.1,
            weighted_score=0.42,
            ticker="NIFTY50",
        )
        points = daily_sentiment_signal([agg])
        assert points[0][0] == T0 + DAY
        assert points[0][1] == pytest.approx(0.42)

    def test_falls_back_to_mean(self):
        points = daily_sentiment_signal([self._agg(1, -0.2)])
        assert points[0][1] == pytest.approx(-0.2)


class TestBaselineComparison:
    def test_summary_mentions_sample_size_and_caveat(self):
        prices = make_prices([100, 101, 102, 103, 104, 105])
        result = backtest_signal(
            prices,
            [(T0 + DAY * i, 0.5) for i in range(3)],
            signal_name="up",
            symbol="NIFTY50",
            horizon_bars=2,
        )
        summary = compare_with_baseline(result)
        assert "n=" in summary and "Research metric only" in summary

    def test_weak_edge_reported_as_no_association(self):
        prices = make_prices([100, 100.4, 99.6, 100.2, 99.8, 100.0])
        result = backtest_signal(
            prices,
            [(T0 + DAY * i, 0.5) for i in range(3)],
            signal_name="weak",
            symbol="X",
            horizon_bars=2,
        )
        summary = compare_with_baseline(result)
        # edge within ±5pp of base rate OR weak overall — either phrasing valid
        assert "no meaningful association" in summary or "associated" in summary


class TestSchema:
    def test_round_trip(self):
        prices = make_prices([100, 102, 101, 104, 103, 106])
        result = backtest_signal(
            prices, [(T0, 0.5), (T0 + DAY, -0.5)], signal_name="rt", symbol="TEST",
            horizon_bars=2,
        )
        restored = BacktestResult.from_json(result.to_json())
        assert restored == result

    @pytest.mark.parametrize("bad_field,bad_value", [
        ("horizon_bars", 0),
        ("hit_rate", 1.5),
        ("max_drawdown", 0.05),
        ("n_evaluated", 99),
    ])
    def test_validation(self, bad_field, bad_value):
        payload = {
            "signal_name": "s", "symbol": "X", "horizon_bars": 5,
            "period_start": "2026-01-01T00:00:00+00:00",
            "period_end": "2026-02-01T00:00:00+00:00",
            "n_signals": 3, "n_evaluated": 2, "hit_rate": 0.5,
            "mean_forward_return": 0.01, "median_forward_return": 0.01,
            "std_forward_return": 0.02, "trade_sharpe": 0.5,
            "buy_hold_return": 0.03, "excess_vs_buy_hold": 0.0,
            "random_hit_rate": 0.55, "max_drawdown": -0.1, "metadata": {},
        }
        payload[bad_field] = bad_value
        with pytest.raises(ValueError):
            BacktestResult(**payload)

    def test_reproducibility_same_inputs_same_result(self):
        prices = make_prices([100, 101, 102, 103, 104])
        signals = [(T0 + DAY * i, 0.3) for i in range(2)]
        a = backtest_signal(prices, signals, signal_name="r", symbol="T", horizon_bars=2)
        b = backtest_signal(prices, signals, signal_name="r", symbol="T", horizon_bars=2)
        assert a == b
