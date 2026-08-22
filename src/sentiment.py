"""Unified financial sentiment engine (Phase 2).

Provides one canonical interface over FinBERT (primary) and VADER
(fallback), returning standardized :class:`SentimentObservation` objects.

Conventions (inherited from the original notebook methodology):
    sentiment_score = P(positive) - P(negative)
    confidence      = max(P(positive), P(neutral), P(negative))

FinBERT label order is resolved at runtime from ``model.config.id2label``
instead of assuming positional order, which makes scoring robust to any
checkpoint layout.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union

from src.models import SentimentObservation

logger = logging.getLogger(__name__)

FINBERT_MODEL_ID = "ProsusAI/finbert"
MAX_SEQUENCE_LENGTH = 512

_BATCH_ITEM_FIELDS = ("text", "source", "ticker", "author_id", "timestamp", "metadata")


@dataclass(frozen=True)
class ScoreResult:
    """Raw model output for one text, before observation wrapping."""

    score: float
    confidence: Optional[float] = None
    prob_positive: Optional[float] = None
    prob_negative: Optional[float] = None
    prob_neutral: Optional[float] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


def score_from_probs(
    prob_positive: float,
    prob_negative: float,
    prob_neutral: float,
) -> tuple[float, float]:
    """Derive ``(score, confidence)`` from class probabilities."""
    probs = (prob_positive, prob_negative, prob_neutral)
    for name, value in zip(("positive", "negative", "neutral"), probs, strict=False):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"prob_{name} out of range [0, 1]: {value}")
    total = sum(float(p) for p in probs)
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"probabilities must sum to ~1.0, got {total}")
    return float(prob_positive) - float(prob_negative), float(max(probs))


def _validate_text(text: Any, context: str = "text") -> str:
    if not isinstance(text, str):
        raise TypeError(f"{context} must be a string, got {type(text).__name__}")
    if not text.strip():
        raise ValueError(f"{context} must be non-empty")
    return text


class BaseSentimentEngine(ABC):
    """Common analyze/batch logic; subclasses implement :meth:`score_batch`."""

    model_name: str = "base"

    @abstractmethod
    def score_batch(self, texts: Sequence[str]) -> List[ScoreResult]:
        """Score a non-empty sequence of validated texts."""

    def _build_observation(
        self,
        text: str,
        result: ScoreResult,
        *,
        source: str,
        ticker: Optional[str],
        author_id: Optional[str],
        timestamp: Optional[datetime],
        metadata: Optional[Dict[str, Any]],
    ) -> SentimentObservation:
        merged_meta: Dict[str, Any] = {
            "text_chars": len(text),
            **(metadata or {}),
            **result.extra_metadata,
        }
        return SentimentObservation(
            timestamp=timestamp or datetime.now(timezone.utc),
            source=source,
            score=result.score,
            ticker=ticker,
            author_id=author_id,
            text=text,
            prob_positive=result.prob_positive,
            prob_negative=result.prob_negative,
            prob_neutral=result.prob_neutral,
            model_name=self.model_name,
            confidence=result.confidence,
            metadata=merged_meta,
        )

    def analyze(
        self,
        text: str,
        *,
        source: str = "manual",
        ticker: Optional[str] = None,
        author_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SentimentObservation:
        """Score one text and wrap it in a :class:`SentimentObservation`."""
        text = _validate_text(text)
        result = self.score_batch([text])[0]
        return self._build_observation(
            text,
            result,
            source=source,
            ticker=ticker,
            author_id=author_id,
            timestamp=timestamp,
            metadata=metadata,
        )

    def analyze_batch(
        self,
        items: Sequence[Union[str, Dict[str, Any]]],
    ) -> List[SentimentObservation]:
        """Score many texts.

        Each item is either a plain string or a dict with keys ``text`` plus
        any of ``source``, ``ticker``, ``author_id``, ``timestamp``,
        ``metadata``.
        """
        if not items:
            return []
        normalized: List[Dict[str, Any]] = []
        for i, item in enumerate(items):
            if isinstance(item, str):
                normalized.append({"text": item})
            elif isinstance(item, dict):
                unknown = set(item) - set(_BATCH_ITEM_FIELDS)
                if unknown:
                    raise ValueError(f"item {i}: unknown keys {sorted(unknown)}")
                if "text" not in item:
                    raise ValueError(f"item {i}: missing 'text'")
                normalized.append(dict(item))
            else:
                raise TypeError(f"item {i} must be str or dict")

            try:
                _validate_text(normalized[-1]["text"], context=f"item {i} text")
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"item {i}: {exc}") from exc

        results = self.score_batch([it["text"] for it in normalized])
        return [
            self._build_observation(
                it["text"],
                result,
                source=it.get("source", "manual"),
                ticker=it.get("ticker"),
                author_id=it.get("author_id"),
                timestamp=it.get("timestamp"),
                metadata=it.get("metadata"),
            )
            for it, result in zip(normalized, results, strict=False)
        ]


class VADEREngine(BaseSentimentEngine):
    """Lexicon fallback using vaderSentiment's compound score.

    Probabilities are unavailable from a lexicon method, so only ``score``
    and ``confidence`` (= |compound|, clipped to [0, 1]) are populated.
    """

    model_name = "vader"

    def __init__(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as exc:  # pragma: no cover - trivial guard
            raise RuntimeError(
                "vaderSentiment is not installed; pip install vaderSentiment"
            ) from exc
        self._analyzer = SentimentIntensityAnalyzer()

    def score_batch(self, texts: Sequence[str]) -> List[ScoreResult]:
        results = []
        for text in texts:
            compound = self._analyzer.polarity_scores(text)["compound"]
            results.append(
                ScoreResult(score=compound, confidence=min(abs(compound), 1.0))
            )
        return results


class FinBERTEngine(BaseSentimentEngine):
    """Transformer-based engine around ProsusAI/finbert.

    Loads tokenizer/model once on first use. Device priority: cuda > mps >
    cpu. The heavy dependencies are imported lazily so this module stays
    importable without torch installed.
    """

    model_name = FINBERT_MODEL_ID

    def __init__(
        self,
        model_id: str = FINBERT_MODEL_ID,
        device: Optional[str] = None,
        max_length: int = MAX_SEQUENCE_LENGTH,
    ) -> None:
        self.model_id = model_id
        self._device_request = device
        self.max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None
        self._label_map: Dict[int, str] = {}
        self._device: str = ""

    # ── loading ────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers/torch are required for FinBERTEngine; "
                "pip install transformers torch"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        model.eval()

        self._device = self._resolve_device(torch)
        model.to(self._device)
        self._model = model
        self._torch = torch
        self._label_map = self._build_label_map(model.config.id2label)

    @staticmethod
    def _resolve_device(torch: Any) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _build_label_map(id2label: Dict[int, str]) -> Dict[int, str]:
        """Map row index -> canonical class name using config labels."""
        mapping: Dict[int, str] = {}
        for idx, raw in id2label.items():
            label = str(raw).strip().lower()
            if "positive" in label or label == "bullish":
                mapping[int(idx)] = "positive"
            elif "negative" in label or label == "bearish":
                mapping[int(idx)] = "negative"
            elif "neutral" in label or label == "none":
                mapping[int(idx)] = "neutral"
        if set(mapping.values()) != {"positive", "negative", "neutral"}:
            raise ValueError(
                f"could not resolve FinBERT labels from id2label={id2label!r}"
            )
        return mapping

    # ── inference ──────────────────────────────────────────────────────
    def score_batch(self, texts: Sequence[str]) -> List[ScoreResult]:
        self._ensure_loaded()
        torch = self._torch
        inputs = self._tokenizer(
            list(texts),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).tolist()
        return [self._probs_to_result(row) for row in probs]

    def _probs_to_result(self, probs_row: Sequence[float]) -> ScoreResult:
        by_class = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for idx, p in enumerate(probs_row):
            cls = self._label_map[idx]
            by_class[cls] += p
        pos, neg, neu = by_class["positive"], by_class["negative"], by_class["neutral"]
        score, confidence = score_from_probs(pos, neg, neu)
        return ScoreResult(
            score=score,
            confidence=confidence,
            prob_positive=pos,
            prob_negative=neg,
            prob_neutral=neu,
            extra_metadata={"device": self._device},
        )


_ENGINE_SINGLETON: Optional[BaseSentimentEngine] = None


def default_engine() -> BaseSentimentEngine:
    """Process-wide shared engine: FinBERT when available, else VADER."""
    global _ENGINE_SINGLETON
    if _ENGINE_SINGLETON is None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            logger.info("Using FinBERT sentiment engine")
            _ENGINE_SINGLETON = FinBERTEngine()
        except ImportError:
            logger.warning("transformers/torch unavailable — falling back to VADER")
            _ENGINE_SINGLETON = VADEREngine()
    return _ENGINE_SINGLETON
