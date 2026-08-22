from datetime import datetime, timezone

import pytest

from src.models import MarketObservation


def make_valid(**overrides):
    base = dict(
        timestamp="2026-08-20T09:15:00+00:00",
        symbol="^NSEI",
        open=24_500.0,
        high=24_700.0,
        low=24_450.0,
        close=24_650.0,
        volume=250_000_000,
        volatility=0.0081,
        vix=None,
        metadata={"interval": "1d", "vol_window_days": 7},
    )
    base.update(overrides)
    return MarketObservation(**base)


class TestValidObservation:
    def test_minimal_valid(self):
        obs = MarketObservation(
            timestamp=datetime(2026, 8, 20, tzinfo=timezone.utc),
            symbol="BTC-USD",
            open=1.0,
            high=2.0,
            low=0.5,
            close=1.5,
        )
        assert obs.volume is None
        assert obs.vix is None

    def test_full_valid(self):
        obs = make_valid()
        assert obs.symbol == "^NSEI"
        assert obs.volatility == 0.0081

    def test_zero_values_allowed(self):
        obs = make_valid(open=0, high=0, low=0, close=0, volume=0)
        assert obs.close == 0


class TestInvalidFields:
    @pytest.mark.parametrize("name", ["open", "high", "low", "close"])
    def test_negative_price_rejected(self, name):
        with pytest.raises(ValueError, match=name):
            make_valid(**{name: -10.0})

    def test_non_numeric_price_rejected(self):
        with pytest.raises(ValueError):
            make_valid(close="expensive")

    def test_high_below_close_rejected(self):
        with pytest.raises(ValueError, match="high"):
            make_valid(high=24_000.0, close=24_650.0)

    def test_low_above_open_rejected(self):
        with pytest.raises(ValueError, match="low"):
            make_valid(low=24_600.0, open=24_500.0)

    def test_high_below_low_rejected(self):
        with pytest.raises(ValueError, match="high"):
            make_valid(high=100.0, low=200.0)

    def test_negative_volume_rejected(self):
        with pytest.raises(ValueError, match="volume"):
            make_valid(volume=-1)

    def test_negative_volatility_rejected(self):
        with pytest.raises(ValueError, match="volatility"):
            make_valid(volatility=-0.01)

    def test_negative_vix_rejected(self):
        with pytest.raises(ValueError, match="vix"):
            make_valid(vix=-5)

    def test_empty_symbol_rejected(self):
        with pytest.raises(ValueError):
            make_valid(symbol="")

    def test_bad_timestamp_rejected(self):
        with pytest.raises(ValueError):
            make_valid(timestamp="20/08/2026")


class TestTimestampHandling:
    def test_naive_assumed_utc(self):
        obs = make_valid(timestamp=datetime(2026, 8, 20, 9, 15))
        assert obs.timestamp.tzinfo == timezone.utc

    def test_z_suffix_parsed_as_utc(self):
        obs = make_valid(timestamp="2026-08-20T09:15:00Z")
        assert obs.timestamp.utcoffset().total_seconds() == 0


class TestSerialization:
    def test_dict_round_trip(self):
        obs = make_valid(vix=13.5)
        restored = MarketObservation.from_dict(obs.to_dict())
        assert restored == obs

    def test_json_round_trip(self):
        obs = make_valid()
        assert MarketObservation.from_json(obs.to_json()) == obs

    def test_extra_keys_ignored_on_load(self):
        data = make_valid().to_dict()
        data["return"] = 0.006
        assert MarketObservation.from_dict(data) == make_valid()

    def test_none_fields_survive_round_trip(self):
        obs = MarketObservation(
            timestamp="2026-08-20T09:15:00Z",
            symbol="X",
            open=1,
            high=1,
            low=1,
            close=1,
        )
        restored = MarketObservation.from_json(obs.to_json())
        assert restored.volume is None and restored.vix is None
