"""FII/DII flow data — provider abstraction without an unstable scraper.

There is no reliable free public API for Indian FII/DII daily flows. The
authoritative sources are NSE daily filing pages and NSDL/CSDL depository
reports (HTML/PDF). Rather than scraping fragile pages, this module defines:

- ``FiiDiiProvider`` — the interface analytics code depends on
- ``ManualCsvFiiDiiProvider`` — loads a CSV the researcher maintains by
  hand from the official pages (documented in docs/DATA_SOURCES.md)

Expected CSV columns: date, institution (FII|DII), buy_value_cr,
sell_value_cr, net_value_cr.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


class FiiDiiError(RuntimeError):
    pass


class FiiDiiObservation:
    __slots__ = ("date", "institution", "buy_value_cr", "sell_value_cr", "net_value_cr")

    def __init__(
        self,
        date: date,
        institution: str,
        buy_value_cr: float,
        sell_value_cr: float,
        net_value_cr: float,
    ) -> None:
        if institution not in ("FII", "DII"):
            raise ValueError(f"institution must be FII or DII, got {institution!r}")
        self.date = date
        self.institution = institution
        self.buy_value_cr = buy_value_cr
        self.sell_value_cr = sell_value_cr
        self.net_value_cr = net_value_cr

    def to_dict(self) -> Dict[str, object]:
        return {
            "date": self.date.isoformat(),
            "institution": self.institution,
            "buy_value_cr": self.buy_value_cr,
            "sell_value_cr": self.sell_value_cr,
            "net_value_cr": self.net_value_cr,
        }


class FiiDiiProvider(ABC):
    @abstractmethod
    def get_flows(
        self,
        start: date,
        end: date,
        institution: Optional[str] = None,
    ) -> List[FiiDiiObservation]:
        """Flow observations within [start, end], optionally per institution."""


class ManualCsvFiiDiiProvider(FiiDiiProvider):
    def __init__(self, csv_path: Path | str) -> None:
        self.csv_path = Path(csv_path)

    def _load_all(self) -> List[FiiDiiObservation]:
        if not self.csv_path.exists():
            raise FiiDiiError(
                f"FII/DII CSV not found at {self.csv_path}; see "
                "docs/DATA_SOURCES.md for how to maintain it"
            )
        out: List[FiiDiiObservation] = []
        with open(self.csv_path, newline="", encoding="utf-8") as handle:
            for line, row in enumerate(csv.DictReader(handle), start=2):
                try:
                    out.append(
                        FiiDiiObservation(
                            date=datetime.strptime(row["date"], "%Y-%m-%d")
                            .replace(tzinfo=timezone.utc)
                            .date(),
                            institution=row["institution"].strip().upper(),
                            buy_value_cr=float(row["buy_value_cr"]),
                            sell_value_cr=float(row["sell_value_cr"]),
                            net_value_cr=float(row["net_value_cr"]),
                        )
                    )
                except (KeyError, ValueError) as exc:
                    raise FiiDiiError(f"{self.csv_path}:{line}: {exc}") from exc
        return sorted(out, key=lambda o: o.date)

    def get_flows(
        self,
        start: date,
        end: date,
        institution: Optional[str] = None,
    ) -> List[FiiDiiObservation]:
        return [
            obs
            for obs in self._load_all()
            if start <= obs.date <= end
            and (institution is None or obs.institution == institution.upper())
        ]
