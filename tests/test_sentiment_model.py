import json
from datetime import datetime, timezone

import pytest

from src.models import SentimentObservation


def make_valid(**overrides):
    base = dict(
        timestamp="2026-08-20T09:30:00+00:00",
        source="news_gdelt",
        score=0.42,
        ticker="NIFTY",
        author_id=None,
        text="Momentum funds see 4x rise in assets",
        prob_positive=0.7,
        prob_negative=0.1,
        prob_neutral=0.2,
        model_name="finbert",
        confidence=0.88,
        metadata={"channel": "news"},
    )
    base.update(overrides)
    return SentimentObservation(**base)


class TestValidObservation:
    def test_minimal_valid(self):
        obs = SentimentObservation(
            timestamp=datetime(2026, 8, 20, tzinfo=timezone.utc),
            source="reddit",
            score=-0.5,
        )
        assert obs.score == -0.5
        assert obs.ticker is None
        assert obs.metadata == {}

    def test_full_valid(self):
        obs = make_valid()
        assert obs.model_name == "finbert"
        assert obs.prob_positive == 0.7

    def test_score_boundaries_allowed(self):
        assert SentimentObservation(
            timestamp="2026-08-20T00:00:00Z", source="x", score=-1.0
        ).score == -1.0
        assert SentimentObservation(
            timestamp="2026-08-20T00:00:00Z", source="x", score=1.0
        ).score == 1.0


class TestInvalidFields:
    @pytest.mark.parametrize("bad", [-1.01, 1.5, "abc", None])
    def test_invalid_score(self, bad):
        with pytest.raises(ValueError):
            make_valid(score=bad)

    @pytest.mark.parametrize("name", ["prob_positive", "prob_negative", "prob_neutral"])
    @pytest.mark.parametrize("bad", [-0.1, 1.2])
    def test_probability_out_of_range(self, name, bad):
        kwargs = {
            "prob_positive": 0.34,
            "prob_negative": 0.33,
            "prob_neutral": 0.33,
        }
        kwargs[name] = bad
        with pytest.raises(ValueError):
            make_valid(**kwargs)

    def test_probabilities_do_not_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            make_valid(prob_positive=0.5, prob_negative=0.5, prob_neutral=0.5)

    def test_confidence_out_of_range(self):
        with pytest.raises(ValueError):
            make_valid(confidence=1.5)

    def test_empty_source_rejected(self):
        with pytest.raises(ValueError):
            make_valid(source="   ")

    def test_bad_timestamp_string(self):
        with pytest.raises(ValueError):
            make_valid(timestamp="not-a-date")

    def test_bad_timestamp_type(self):
        with pytest.raises(TypeError):
            make_valid(timestamp=12345)


class TestTimestampHandling:
    def test_iso_z_suffix_parsed_as_utc(self):
        obs = make_valid(timestamp="2026-08-20T09:30:00Z")
        assert obs.timestamp.tzinfo == timezone.utc

    def test_naive_datetime_assumed_utc(self):
        naive = datetime(2026, 8, 20, 9, 30)
        obs = make_valid(timestamp=naive)
        assert obs.timestamp.tzinfo == timezone.utc
        assert obs.timestamp.hour == 9

    def test_non_utc_normalized(self):
        obs = make_valid(timestamp="+05:30".join(["2026-08-20T15:00:00", ""]))
        assert obs.timestamp.utcoffset().total_seconds() == 0
        assert obs.timestamp.hour == 9

    def test_datetime_preserved_exactly(self):
        dt = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        assert make_valid(timestamp=dt).timestamp == dt


class TestSerialization:
    def test_dict_round_trip(self):
        obs = make_valid()
        restored = SentimentObservation.from_dict(obs.to_dict())
        assert restored == obs

    def test_json_round_trip(self):
        obs = make_valid()
        restored = SentimentObservation.from_json(obs.to_json())
        assert restored == obs

    def test_to_dict_timestamp_is_iso_string(self):
        data = make_valid().to_dict()
        assert isinstance(data["timestamp"], str)
        datetime.fromisoformat(data["timestamp"])

    def test_extra_keys_ignored_on_load(self):
        data = make_valid().to_dict()
        data["unknown_future_field"] = "ok"
        assert SentimentObservation.from_dict(data) == make_valid()

    def test_json_payload_is_valid_json(self):
        payload = make_valid(metadata={"k": [1, 2]}).to_json()
        assert json.loads(payload)["metadata"] == {"k": [1, 2]}
