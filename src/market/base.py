"""Market data provider abstraction (Phase 5).

Canonical symbol names are provider-independent:

==================  ===========  ==================
Canonical           Yahoo        Notes
==================  ===========  ==================
NIFTY50             ^NSEI        NIFTY 50 index
BANKNIFTY           ^NSEBANK     BANK NIFTY index
INDIAVIX            ^INDIAVIX    India VIX
BTC                 BTC-USD      Bitcoin
RELIANCE           RELIANCE.NS  example equity
NIFTY_IT            ^CNXIT       sector index
==================  ===========  ==================

Providers return typed :class:`MarketObservation` rows with UTC timestamps.
Licensing note (docs/DATA_SOURCES.md): Yahoo Finance data is fine for
personal/research use but not for redistribution; FRED is free with an API
key under FRED terms.
"""

from __future__ import annotations

import logging
import time as _time
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional

from src.models import MarketObservation

logger = logging.getLogger(__name__)

CANONICAL_SYMBOLS: Dict[str, str] = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "INDIAVIX": "^INDIAVIX",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_FMCG": "^CNXFMCG",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "BTC": "BTC-USD",
}


class MarketDataError(RuntimeError):
    pass


def resolve_ticker(canonical: str, mapping: Optional[Dict[str, str]] = None) -> str:
    """Canonical name -> provider ticker (identity if already a ticker)."""
    lookup = {**CANONICAL_SYMBOLS, **(mapping or {})}
    return lookup.get(canonical, canonical)


def with_retries(
    fn: Callable[[], object],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    sleep: Callable[[float], None] = _time.sleep,
) -> object:
    """Run ``fn`` with exponential-backoff retries on any exception.

    ``sleep`` is injectable so tests run without real delays. The last
    exception is re-raised once all attempts fail.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            logger.warning("attempt %d/%d failed (%s) — retrying in %.1fs", attempt + 1, attempts, exc, delay)
            sleep(delay)
    raise MarketDataError("unreachable")


class MarketDataProvider(ABC):
    """A source of OHLCV history convertible to MarketObservation rows."""

    @abstractmethod
    def get_history(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> List[MarketObservation]:
        """Inclusive [start, end] daily history for one canonical symbol."""

    def get_latest(self, symbol: str, lookback_days: int = 7) -> List[MarketObservation]:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=lookback_days)
        history = self.get_history(symbol, start, end)
        return history[-1:] if history else []


class CachedProviderMixin:
    """Tiny in-memory cache keyed by (symbol, start, end)."""

    def __init__(self) -> None:
        self._cache: Dict[tuple, List[MarketObservation]] = {}

    def _cache_get(self, key: tuple) -> Optional[List[MarketObservation]]:
        return self._cache.get(key)

    def _cache_put(self, key: tuple, rows: List[MarketObservation]) -> None:
        self._cache[key] = rows


def filter_observations(
    rows: Iterable[MarketObservation],
    start: date,
    end: date,
) -> List[MarketObservation]:
    """Keep rows whose UTC date falls within [start, end]."""
    return [
        row
        for row in rows
        if start <= row.timestamp.date() <= end
    ]
