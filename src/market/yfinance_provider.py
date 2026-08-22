"""Yahoo Finance-backed provider (lazy yfinance import, retry + cache)."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, List, Optional

from src.market.base import (
    CachedProviderMixin,
    MarketDataError,
    MarketDataProvider,
    resolve_ticker,
    with_retries,
)
from src.models import MarketObservation

logger = logging.getLogger(__name__)


class YFinanceProvider(CachedProviderMixin, MarketDataProvider):
    """Daily OHLCV via yfinance. ``vix`` is populated for INDIAVIX/^VIX rows.

    ``yf_module`` is injectable for offline tests (defaults to the real
    yfinance module).
    """

    def __init__(
        self,
        symbol_mapping: Optional[dict] = None,
        retries: int = 3,
        yf_module: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self._mapping = symbol_mapping or {}
        self._retries = retries
        self._yf = yf_module

    def _get_yf(self) -> Any:
        if self._yf is None:
            try:
                import yfinance as yf
            except ImportError as exc:
                raise MarketDataError(
                    "yfinance is not installed; pip install yfinance"
                ) from exc
            self._yf = yf
        return self._yf

    @staticmethod
    def _is_vol_index(ticker: str) -> bool:
        return ticker.upper() in ("^INDIAVIX", "^VIX", "^VXN")

    def get_history(self, symbol: str, start: date, end: date) -> List[MarketObservation]:
        key = (symbol, start, end)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        yf = self._get_yf()
        ticker = resolve_ticker(symbol, self._mapping)
        # yfinance's ``end`` is exclusive — pad one day for inclusive semantics
        end_exclusive = date.fromordinal(end.toordinal() + 1)

        frame = with_retries(
            lambda: yf.download(
                ticker,
                start=start.isoformat(),
                end=end_exclusive.isoformat(),
                progress=False,
                auto_adjust=False,
            ),
            attempts=self._retries,
        )
        if frame is None or len(frame) == 0:
            logger.warning("no data returned for %s (%s)", symbol, ticker)
            return []

        if hasattr(frame.columns, "get_level_values") and getattr(frame.columns, "nlevels", 1) > 1:
            frame.columns = frame.columns.get_level_values(0)

        rows: List[MarketObservation] = []
        for idx, row in frame.iterrows():
            close = row.get("Close")
            if close is None or close != close:  # NaN guard
                continue
            raw_volume = row.get("Volume")
            volume = (
                None
                if raw_volume is None or raw_volume != raw_volume
                else float(raw_volume)
            )
            vix_value = float(close) if self._is_vol_index(ticker) else None
            rows.append(
                MarketObservation(
                    timestamp=idx.to_pydatetime(),
                    symbol=symbol,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(close),
                    volume=volume,
                    vix=vix_value,
                    metadata={"provider": "yfinance", "ticker": ticker},
                )
            )
        self._cache_put(key, rows)
        return rows
