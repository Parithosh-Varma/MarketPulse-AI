"""FRED macro provider — plain REST client, key from environment only."""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any, List, Optional

from src.market.base import (
    CachedProviderMixin,
    MarketDataProvider,
    MarketDataError,
    with_retries,
)
from src.models import MarketObservation

logger = logging.getLogger(__name__)

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

FREQUENT_SERIES: dict = {
    "FEDFUNDS": "Federal Funds Effective Rate",
    "CPIAUCSL": "CPI All Urban Consumers",
    "UNRATE": "Civilian Unemployment Rate",
    "DGS10": "10-Year Treasury Constant Maturity",
    "DGS2": "2-Year Treasury Constant Maturity",
    "T10Y2Y": "10y-2y Term Spread",
    "GDP": "Gross Domestic Product",
}


class FREDProvider(CachedProviderMixin, MarketDataProvider):
    """Fetch FRED series as close-only MarketObservation rows.

    Credentials: reads ``FRED_API_KEY`` from the environment at call time
    (never hard-coded). ``http_get`` is injectable for offline tests.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        retries: int = 3,
        http_get: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._retries = retries
        self._http_get = http_get

    def _get_key(self) -> str:
        key = self._api_key or os.environ.get("FRED_API_KEY", "")
        if not key:
            raise MarketDataError(
                "FRED_API_KEY is not set; export it or pass api_key="
            )
        return key

    def _get_http(self) -> Any:
        if self._http_get is None:
            try:
                import requests
            except ImportError as exc:
                raise MarketDataError("requests is required for FREDProvider") from exc
            self._http_get = requests.get
        return self._http_get

    def get_history(self, symbol: str, start: date, end: date) -> List[MarketObservation]:
        cache_key = (symbol, start, end)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        http_get = self._get_http()
        response = with_retries(
            lambda: http_get(
                FRED_OBSERVATIONS_URL,
                params={
                    "series_id": symbol,
                    "api_key": self._get_key(),
                    "file_type": "json",
                    "observation_start": start.isoformat(),
                    "observation_end": end.isoformat(),
                },
                timeout=30,
            ),
            attempts=self._retries,
        )
        if getattr(response, "status_code", 0) != 200:
            raise MarketDataError(
                f"FRED returned HTTP {getattr(response, 'status_code', '?')} "
                f"for {symbol}"
            )

        rows: List[MarketObservation] = []
        for point in response.json().get("observations", []):
            if point.get("value") in (".", None):  # FRED missing-data marker
                continue
            rows.append(
                MarketObservation(
                    timestamp=point["date"],
                    symbol=symbol,
                    open=float(point["value"]),
                    high=float(point["value"]),
                    low=float(point["value"]),
                    close=float(point["value"]),
                    metadata={
                        "provider": "fred",
                        "description": FREQUENT_SERIES.get(symbol, ""),
                    },
                )
            )
        self._cache_put(cache_key, rows)
        return rows
