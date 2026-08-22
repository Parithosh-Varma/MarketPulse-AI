import json
from datetime import datetime, timedelta, timezone

from src.alerts import (
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    Alert,
    AlertEngine,
    JsonlFileChannel,
    check_extreme_divergence,
    check_regime_change,
    check_sentiment_drop,
    check_vix_spike,
)


class RecordingChannel:
    def __init__(self):
        self.received = []

    def deliver(self, alert):
        self.received.append(alert)


def ctx(**overrides):
    base = {
        "sentiment_momentum": -0.45,
        "divergence_score": -0.72,
        "divergence_classification": "bearish_divergence",
        "previous_regime": "bullish",
        "current_regime": "risk_off",
        "vix_level": 22.0,
        "vix_baseline": 14.0,
    }
    base.update(overrides)
    return base


class TestRules:
    def test_sentiment_drop_fires(self):
        alert = check_sentiment_drop(ctx())
        assert alert is not None and alert.severity == SEVERITY_MEDIUM

    def test_mild_drop_stays_quiet(self):
        assert check_sentiment_drop(ctx(sentiment_momentum=-0.10)) is None

    def test_missing_data_never_fires(self):
        assert check_sentiment_drop({}) is None
        assert check_extreme_divergence({}) is None
        assert check_vix_spike({}) is None
        assert check_regime_change({}) is None

    def test_divergence_requires_divergent_class(self):
        ok = check_extreme_divergence(ctx())
        assert ok is not None
        aligned = check_extreme_divergence(
            ctx(divergence_score=-0.9, divergence_classification="aligned_bearish")
        )
        assert aligned is None

    def test_regime_change(self):
        alert = check_regime_change(ctx())
        assert alert is not None and alert.severity == SEVERITY_HIGH
        assert check_regime_change(ctx(current_regime="bullish")) is None

    def test_vix_spike_multiplier(self):
        fired = check_vix_spike(ctx())  # 22 >= 14 * 1.4 = 19.6
        assert fired is not None
        calm = check_vix_spike(ctx(vix_level=16.0))
        assert calm is None


class TestEngine:
    def test_dispatches_to_channels(self):
        channel = RecordingChannel()
        engine = AlertEngine(channels=[channel], cooldown_seconds=0)
        fired = engine.process(ctx())
        names = {a.rule_name for a in fired}
        assert {"sentiment_drop", "extreme_divergence", "regime_change", "vix_spike"} <= names
        assert len(channel.received) == len(fired)

    def test_cooldown_suppresses_repeat(self):
        channel = RecordingChannel()
        fake_now = [datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)]
        engine = AlertEngine(
            channels=[channel],
            cooldown_seconds=3600,
            clock=lambda: fake_now[0],
        )
        first = engine.process(ctx())
        fake_now[0] = fake_now[0] + timedelta(minutes=10)
        second = engine.process(ctx())
        assert first and second == []

    def test_cooldown_expires(self):
        channel = RecordingChannel()
        fake_now = [datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)]
        engine = AlertEngine(
            channels=[channel],
            cooldown_seconds=3600,
            clock=lambda: fake_now[0],
        )
        engine.process(ctx())
        fake_now[0] = fake_now[0] + timedelta(hours=2)
        again = engine.process(ctx())
        assert again

    def test_subset_of_rules_selectable(self):
        channel = RecordingChannel()
        engine = AlertEngine(channels=[channel], rules=["regime_change"], cooldown_seconds=0)
        fired = engine.process(ctx())
        assert [a.rule_name for a in fired] == ["regime_change"]

    def test_crashing_rule_does_not_block_others(self, monkeypatch):
        from src.alerts import DEFAULT_RULES

        def boom(context):
            raise RuntimeError("boom")

        monkeypatch.setitem(DEFAULT_RULES, "sentiment_drop", boom)
        channel = RecordingChannel()
        engine = AlertEngine(channels=[channel], rules=["sentiment_drop", "regime_change"],
                             cooldown_seconds=0)
        fired = engine.process(ctx())
        assert [a.rule_name for a in fired] == ["regime_change"]


class TestJsonlChannel:
    def test_writes_valid_jsonl(self, tmp_path):
        target = tmp_path / "alerts" / "feed.jsonl"
        channel = JsonlFileChannel(target)
        channel.deliver(Alert(rule_name="r", severity="low", message="m"))
        channel.deliver(Alert(rule_name="r2", severity="high", message="m2"))
        lines = target.read_text().strip().splitlines()
        assert len(lines) == 2
        parsed = [json.loads(line) for line in lines]
        assert parsed[1]["rule_name"] == "r2"
        datetime.fromisoformat(parsed[0]["fired_at"])
