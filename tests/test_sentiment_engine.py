from datetime import datetime, timezone

import pytest

from src.models import SentimentObservation
from src.sentiment import (
    BaseSentimentEngine,
    FinBERTEngine,
    ScoreResult,
    VADEREngine,
    score_from_probs,
)


class DeterministicEngine(BaseSentimentEngine):
    model_name = "mock"

    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.calls = []

    def score_batch(self, texts):
        self.calls.append(list(texts))
        results = []
        for text in texts:
            if text in self.mapping:
                p, n, neu = self.mapping[text]
            else:
                p, n, neu = 0.6, 0.2, 0.2
            score, conf = score_from_probs(p, n, neu)
            results.append(
                ScoreResult(
                    score=score,
                    confidence=conf,
                    prob_positive=p,
                    prob_negative=n,
                    prob_neutral=neu,
                    extra_metadata={"engine": "mock"},
                )
            )
        return results


class TestScoreFromProbs:
    def test_bullish(self):
        score, confidence = score_from_probs(0.7, 0.1, 0.2)
        assert score == pytest.approx(0.6)
        assert confidence == pytest.approx(0.7)

    def test_bearish(self):
        score, _ = score_from_probs(0.05, 0.85, 0.10)
        assert score == pytest.approx(-0.80)

    def test_perfectly_neutral(self):
        score, confidence = score_from_probs(0.0, 0.0, 1.0)
        assert score == 0.0
        assert confidence == pytest.approx(1.0)

    def test_probability_out_of_range(self):
        with pytest.raises(ValueError):
            score_from_probs(1.5, -0.5, 0.0)

    def test_probabilities_do_not_sum_to_one(self):
        with pytest.raises(ValueError):
            score_from_probs(0.5, 0.5, 0.5)


class TestBaseEngineAnalyze:
    def setup_method(self):
        self.engine = DeterministicEngine()

    def test_returns_observation(self):
        obs = self.engine.analyze("Record profits", source="news", ticker="RELIANCE")
        assert isinstance(obs, SentimentObservation)
        assert obs.source == "news"
        assert obs.ticker == "RELIANCE"
        assert obs.model_name == "mock"
        assert obs.score == pytest.approx(0.4)
        assert obs.confidence == pytest.approx(0.6)
        assert obs.prob_positive == pytest.approx(0.6)
        assert obs.text == "Record profits"

    def test_default_source_and_timestamp(self):
        obs = self.engine.analyze("hello")
        assert obs.source == "manual"
        assert obs.timestamp.tzinfo == timezone.utc

    def test_explicit_timestamp_respected(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert self.engine.analyze("hello", timestamp=ts).timestamp == ts

    def test_metadata_merged(self):
        obs = self.engine.analyze("hello", metadata={"lang": "en"})
        assert obs.metadata["text_chars"] == 5
        assert obs.metadata["lang"] == "en"
        assert obs.metadata["engine"] == "mock"

    @pytest.mark.parametrize("bad", ["", "   ", None, 42])
    def test_invalid_text_rejected(self, bad):
        with pytest.raises((ValueError, TypeError)):
            self.engine.analyze(bad)


class TestBatch:
    def setup_method(self):
        self.engine = DeterministicEngine({"Bull run": (0.9, 0.05, 0.05)})

    def test_strings(self):
        out = self.engine.analyze_batch(["Bull run", "Mild concerns"])
        assert len(out) == 2
        assert out[0].score == pytest.approx(0.85)
        assert out[1].score == pytest.approx(0.4)
        assert self.engine.calls == [["Bull run", "Mild concerns"]]

    def test_dicts_with_fields(self):
        items = [
            {"text": "Bull run", "source": "reddit", "ticker": "NIFTY"},
            {"text": "Mild concerns", "source": "news"},
        ]
        out = self.engine.analyze_batch(items)
        assert out[0].source == "reddit" and out[0].ticker == "NIFTY"
        assert out[1].source == "news"

    def test_empty_input(self):
        assert self.engine.analyze_batch([]) == []
        assert self.engine.calls == []

    def test_item_missing_text(self):
        with pytest.raises(ValueError, match="missing 'text'"):
            self.engine.analyze_batch([{"source": "x"}])

    def test_unknown_item_key(self):
        with pytest.raises(ValueError, match="unknown keys"):
            self.engine.analyze_batch([{"text": "a", "bogus": 1}])

    def test_bad_item_reports_index(self):
        with pytest.raises(ValueError, match="item 1"):
            self.engine.analyze_batch(["ok", "   "])


class TestFinBERTLabelResolution:
    def make_engine(self, id2label):
        engine = FinBERTEngine.__new__(FinBERTEngine)
        engine._tokenizer = None
        engine._model = object()
        engine._label_map = FinBERTEngine._build_label_map(id2label)
        engine._device = "cpu"
        engine._torch = None
        return engine

    def test_notebook_order_neg_neu_pos(self):
        engine = self.make_engine({0: "negative", 1: "neutral", 2: "positive"})
        result = engine._probs_to_result([0.1, 0.2, 0.7])
        assert result.prob_positive == pytest.approx(0.7)
        assert result.score == pytest.approx(0.6)

    def test_hf_card_order_pos_neg_neu(self):
        engine = self.make_engine({0: "positive", 1: "negative", 2: "neutral"})
        result = engine._probs_to_result([0.8, 0.15, 0.05])
        assert result.prob_positive == pytest.approx(0.8)
        assert result.score == pytest.approx(0.65)

    def test_case_insensitive_labels(self):
        engine = self.make_engine({0: "POSITIVE", 1: "NEGATIVE", 2: "NEUTRAL"})
        result = engine._probs_to_result([0.5, 0.25, 0.25])
        assert result.score == pytest.approx(0.25)

    def test_unresolvable_labels_rejected(self):
        with pytest.raises(ValueError, match="id2label"):
            FinBERTEngine._build_label_map({0: "positive", 1: "negative"})


class TestVADEREngine:
    def setup_method(self):
        self.engine = VADEREngine()

    def test_positive_text(self):
        result = self.engine.score_batch(["Record quarterly profits surge"])[0]
        assert result.score > 0.2
        assert 0 <= result.confidence <= 1
        assert result.prob_positive is None

    def test_negative_text(self):
        result = self.engine.score_batch(["Massive losses and bankruptcy"])[0]
        assert result.score < -0.2

    def test_analyze_wraps_observation(self):
        obs = self.engine.analyze("Bankruptcy fears grow", source="news")
        assert obs.model_name == "vader"
        assert obs.score < 0
        assert obs.confidence == min(abs(obs.score), 1.0)


class TestDefaultEngineSelection:
    def test_returns_working_engine(self):
        from src.sentiment import default_engine

        engine = default_engine()
        obs = engine.analyze("Steady quarter, no surprises")
        assert -1.0 <= obs.score <= 1.0
