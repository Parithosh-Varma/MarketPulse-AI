import importlib
from datetime import date, datetime, timedelta, timezone

import pytest

from src.market import (
    FiiDiiError,
    FREDProvider,
    ManualCsvFiiDiiProvider,
    MarketDataError,
    YFinanceProvider,
    resolve_ticker,
    with_retries,
)
from src.models import MarketObservation
from src.timeutils import ist_market_date, to_ist


def make_fake_yf(rows_by_ticker, fail_times=0):
    calls = {"count": 0}

    class FakeYF:
        @staticmethod
        def download(ticker, start=None, end=None, progress=False, auto_adjust=False):
            nonlocal calls
            if calls["count"] < fail_times:
                calls["count"] += 1
                raise ConnectionError("transient")
            import pandas as pd

            frame = rows_by_ticker.get(ticker)
            if not frame:
                return pd.DataFrame()
            df = pd.DataFrame(frame)
            df.index = pd.bdate_range("2026-08-10", periods=len(df))
            return df

    return FakeYF(), calls


class TestResolveTicker:
    def test_canonical_names(self):
        assert resolve_ticker("NIFTY50") == "^NSEI"
        assert resolve_ticker("BANKNIFTY") == "^NSEBANK"
        assert resolve_ticker("INDIAVIX") == "^INDIAVIX"
        assert resolve_ticker("BTC") == "BTC-USD"

    def test_passthrough_unknown(self):
        assert resolve_ticker("RELIANCE.NS") == "RELIANCE.NS"

    def test_custom_mapping_overrides(self):
        assert resolve_ticker("NIFTY50", {"NIFTY50": "MY.TICK"}) == "MY.TICK"


class TestWithRetries:
    def test_success_first_try(self):
        assert with_retries(lambda: 42) == 42

    def test_retries_then_succeeds(self):
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise ConnectionError()
            return "ok"

        sleeps = []
        result = with_retries(flaky, attempts=3, sleep=sleeps.append)
        assert result == "ok" and len(sleeps) == 2

    def test_exhaustion_reraises_last_error(self):
        def always_fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            with_retries(always_fail, attempts=2, sleep=lambda s: None)

    def test_invalid_attempts(self):
        with pytest.raises(ValueError):
            with_retries(lambda: 1, attempts=0)


class TestYFinanceProvider:
    def make_provider(self, rows_by_ticker, fail_times=0):
        fake_yf, calls = make_fake_yf(rows_by_ticker, fail_times)
        return YFinanceProvider(yf_module=fake_yf, retries=3), calls

    def test_history_conversion(self):
        rows = {
            "^NSEI": [
                {"Open": 24000.0, "High": 24700.0, "Low": 23900.0, "Close": 24650.5,
                 "Volume": 250000000},
                {"Open": 24660.0, "High": 24800.0, "Low": 24500.0, "Close": 24710.0,
                 "Volume": 230000000},
            ]
        }
        provider, _ = self.make_provider(rows)
        history = provider.get_history("NIFTY50", date(2026, 8, 19), date(2026, 8, 20))
        assert len(history) == 2
        first = history[0]
        assert isinstance(first, MarketObservation)
        assert first.symbol == "NIFTY50"
        assert first.close == pytest.approx(24650.5)
        assert first.volume == pytest.approx(250_000_000)
        assert first.metadata["ticker"] == "^NSEI"

    def test_vix_row_populates_vix_field(self):
        rows = {
            "^INDIAVIX": [
                {"Open": 13.2, "High": 14.0, "Low": 13.0, "Close": 13.55, "Volume": None},
            ]
        }
        provider, _ = self.make_provider(rows)
        (row,) = provider.get_history("INDIAVIX", date(2026, 8, 20), date(2026, 8, 20))
        assert row.vix == pytest.approx(13.55)

    def test_retry_on_transient_failure(self):
        provider, calls = self.make_provider(
            {"^NSEI": [{"Open": 1, "High": 1, "Low": 1, "Close": 1, "Volume": 0}]},
            fail_times=2,
        )
        history = provider.get_history("NIFTY50", date(2026, 8, 20), date(2026, 8, 20))
        assert len(history) == 1
        assert calls["count"] == 2

    def test_empty_result_returns_list(self):
        provider, _ = self.make_provider({})
        assert provider.get_history("NIFTY50", date(2026, 8, 20), date(2026, 8, 21)) == []

    def test_in_memory_cache_hits(self):
        provider, _ = self.make_provider(
            {"^NSEI": [{"Open": 1, "High": 1, "Low": 1, "Close": 1, "Volume": 0}]}
        )
        a = provider.get_history("NIFTY50", date(2026, 8, 20), date(2026, 8, 20))
        b = provider.get_history("NIFTY50", date(2026, 8, 20), date(2026, 8, 20))
        assert a is b

    def test_get_latest(self):
        rows = {"^NSEI": []}
        for day in range(1, 6):
            close = float(day)
            rows["^NSEI"].append(
                {"Open": close - 0.25, "High": close + 0.5, "Low": close - 0.5,
                 "Close": close, "Volume": 0}
            )
        provider, _ = self.make_provider(rows)
        latest = provider.get_latest("NIFTY50", lookback_days=30)
        assert len(latest) == 1 and latest[0].close == pytest.approx(5.0)


class TestFREDProvider:
    class FakeResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def make_provider(self, payload, status=200):
        captured = {}

        def fake_get(url, params=None, timeout=None):
            captured.update({"url": url, "params": params})
            resp = object.__new__(TestFREDProvider.FakeResponse)
            TestFREDProvider.FakeResponse.status_code = status
            resp._payload = payload
            return resp

        provider = FREDProvider(api_key="test-key", http_get=fake_get)
        return provider, captured

    def test_parses_series_and_skips_missing(self):
        payload = {
            "observations": [
                {"date": "2026-08-01", "value": "5.33"},
                {"date": "2026-08-02", "value": "."},
                {"date": "2026-08-03", "value": "5.31"},
            ]
        }
        provider, captured = self.make_provider(payload)
        rows = provider.get_history("FEDFUNDS", date(2026, 8, 1), date(2026, 8, 3))
        assert [r.close for r in rows] == [pytest.approx(5.33), pytest.approx(5.31)]
        assert captured["params"]["series_id"] == "FEDFUNDS"
        assert captured["url"].startswith("https://api.stlouisfed.org")

    def test_http_error_raises(self):
        provider, _ = self.make_provider({}, status=400)
        with pytest.raises(MarketDataError, match="400"):
            provider.get_history("GDP", date(2026, 1, 1), date(2026, 1, 2))

    def test_missing_key_raises_clearly(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        provider = FREDProvider(http_get=lambda *a, **k: None)
        with pytest.raises(MarketDataError, match="FRED_API_KEY"):
            provider.get_history("GDP", date(2026, 1, 1), date(2026, 1, 2))


class TestFiiDii:
    def test_manual_csv_provider(self, tmp_path):
        csv_file = tmp_path / "flows.csv"
        csv_file.write_text(
            "date,institution,buy_value_cr,sell_value_cr,net_value_cr\n"
            "2026-08-18,FII,12000.5,11500.0,500.5\n"
            "2026-08-18,DII,9000.0,8500.25,499.75\n"
            "2026-08-19,FII,11000.0,12500.0,-1500.0\n"
        )
        provider = ManualCsvFiiDiiProvider(csv_file)
        flows = provider.get_flows(date(2026, 8, 18), date(2026, 8, 19))
        assert len(flows) == 3
        fii_only = provider.get_flows(date(2026, 8, 18), date(2026, 8, 19), institution="fii")
        assert all(f.institution == "FII" for f in fii_only) and len(fii_only) == 2

    def test_missing_csv_is_explicit(self, tmp_path):
        provider = ManualCsvFiiDiiProvider(tmp_path / "nope.csv")
        with pytest.raises(FiiDiiError, match="DATA_SOURCES"):
            provider.get_flows(date(2026, 1, 1), date(2026, 1, 2))

    def test_bad_institution_rejected(self):
        from src.market.fiidii import FiiDiiObservation

        with pytest.raises(ValueError, match="institution"):
            FiiDiiObservation(date(2026, 8, 18), "HEDGE", 1, 1, 0)


class TestTimezoneStandard:
    def test_to_ist(self):
        utc_ts = datetime(2026, 8, 20, 4, 30, tzinfo=timezone.utc)
        ist = to_ist(utc_ts)
        assert ist.hour == 10  # UTC+5:30
        assert str(ist.tzinfo) == "Asia/Kolkata"

    def test_naive_treated_as_utc(self):
        naive = datetime(2026, 8, 20, 4, 30)
        assert to_ist(naive).hour == 10

    def test_ist_market_date(self):
        # 20:30 UTC on Aug 20 = 02:00 IST on Aug 21
        assert ist_market_date(datetime(2026, 8, 20, 20, 30, tzinfo=timezone.utc)) == "2026-08-21"


@pytest.mark.parametrize("module_name", ["src.market", "src.timeutils"])
def test_modules_import_clean(module_name):
    assert importlib.import_module(module_name) is not None


def test_timedelta_helper_import():
    _ = timedelta(days=1)
