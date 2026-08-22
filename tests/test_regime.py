from datetime import datetime, timezone

import pytest

from src.models.regime import MarketRegime, MarketStress
from src.regime import (
    REGIME_BEARISH,
    REGIME_BULLISH,
    REGIME_HIGH_VOL,
    REGIME_NEUTRAL,
    REGIME_RISK_OFF,
    REGIME_RISK_ON,
    REGIME_TRANSITION,
    STRESS_HIGH,
    STRESS_LOW,
    STRESS_MEDIUM,
    RegimeConfig,
    RegimeInputs,
    classify_gmsi_stress,
    classify_regime,
    make_stress,
)

T0 = datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)


def inputs(**overrides):
    base = dict(as_of=T0, symbol="NIFTY50")
    base.update(overrides)
    return RegimeInputs(**base)


class TestLegacyGMSIStress:
    def test_quantile_buckets_match_research_script(self):
        # scripts/regime_analysis.py: <=p20 low, <=p80 medium, else high
        assert classify_gmsi_stress(-1.5, p20=-0.5, p80=0.5) == STRESS_LOW
        assert classify_gmsi_stress(0.0, p20=-0.5, p80=0.5) == STRESS_MEDIUM
        assert classify_gmsi_stress(2.0, p20=-0.5, p80=0.5) == STRESS_HIGH

    def test_make_stress_wraps_schema(self):
        stress = make_stress(0.9, T0, p20=-0.5, p80=0.5)
        assert isinstance(stress, MarketStress)
        assert stress.classification == STRESS_HIGH
        assert stress.source == "fin_sentinel_gmsi"
        assert MarketStress.from_dict(stress.to_dict()) == stress


class TestRuleLadder:
    def test_volatility_shock_wins(self):
        out = classify_regime(
            inputs(vix_level=30.0, price_momentum=0.05, sentiment=0.8)
        )
        assert out.regime == REGIME_HIGH_VOL

    def test_realized_vol_also_triggers_high_vol(self):
        out = classify_regime(inputs(realized_volatility=0.03))
        assert out.regime == REGIME_HIGH_VOL

    def test_risk_off_via_elevated_vix_and_negative_sentiment(self):
        out = classify_regime(inputs(vix_level=19.0, sentiment=-0.30, gmsi=0.1))
        assert out.regime == REGIME_RISK_OFF

    def test_risk_off_via_gmsi(self):
        out = classify_regime(inputs(gmsi=0.9))
        assert out.regime == REGIME_RISK_OFF

    def test_risk_on_requires_all_three(self):
        calm = classify_regime(inputs(vix_level=12.0, price_momentum=0.01, sentiment=0.2))
        assert calm.regime == REGIME_RISK_ON
        missing_sent = classify_regime(inputs(vix_level=12.0, price_momentum=0.01))
        assert missing_sent.regime != REGIME_RISK_ON

    def test_bullish_trend_with_sentiment(self):
        out = classify_regime(inputs(vix_level=17.0, price_momentum=0.04, sentiment=0.35))
        assert out.regime == REGIME_BULLISH

    def test_bearish_trend_with_sentiment(self):
        out = classify_regime(inputs(price_momentum=-0.05, sentiment=-0.40))
        assert out.regime == REGIME_BEARISH

    def test_quiet_market_is_transition(self):
        out = classify_regime(inputs(price_momentum=0.001, sentiment=0.02, vix_level=13.0))
        assert out.regime == REGIME_TRANSITION

    def test_conflicting_signals_fall_to_neutral(self):
        out = classify_regime(inputs(price_momentum=-0.06, sentiment=0.60))
        assert out.regime in (REGIME_NEUTRAL,)

    def test_no_data_defaults_neutral(self):
        out = classify_regime(inputs())
        assert out.regime == REGIME_NEUTRAL


class TestAuditability:
    def test_components_recorded(self):
        out = classify_regime(
            inputs(
                price_momentum=0.03,
                realized_volatility=0.006,
                vix_level=14.0,
                sentiment=0.25,
                sentiment_momentum=0.05,
                gmsi=0.2,
                mfi=0.4,
            )
        )
        assert out.components["price_momentum"] == pytest.approx(0.03)
        assert out.components["mfi"] == pytest.approx(0.4)
        assert out.metadata["rule_votes"]

    def test_risk_appetite_mapping(self):
        on = classify_regime(inputs(vix_level=12.0, price_momentum=0.01, sentiment=0.3))
        off = classify_regime(inputs(gmsi=0.95))
        other = classify_regime(inputs(price_momentum=0.05, sentiment=0.5))
        assert on.risk_appetite == "risk_on"
        assert off.risk_appetite == "risk_off"
        assert other.risk_appetite == "unknown"

    def test_stress_classification_attached_when_gmsi_present(self):
        out = classify_regime(inputs(gmsi=0.9))
        assert out.stress_classification == STRESS_HIGH

    def test_custom_config_thresholds(self):
        strict = RegimeConfig(mom_threshold=0.10, sent_threshold=0.5)
        out = classify_regime(inputs(price_momentum=0.04, sentiment=0.35), config=strict)
        assert out.regime != REGIME_BULLISH  # below the stricter gates


class TestSchema:
    def test_invalid_regime_rejected(self):
        with pytest.raises(ValueError, match="regime"):
            MarketRegime(
                as_of=T0, symbol=None, regime="moonshot", confidence=0.5,
                components={},
            )

    def test_confidence_bounds_enforced(self):
        with pytest.raises(ValueError, match="confidence"):
            MarketRegime(
                as_of=T0, symbol=None, regime=REGIME_BULLISH, confidence=1.5,
                components={},
            )

    def test_round_trip(self):
        out = classify_regime(inputs(vix_level=12.0, price_momentum=0.01, sentiment=0.3))
        restored = MarketRegime.from_dict(out.to_dict())
        assert restored == out
        assert MarketRegime.from_json(out.to_json()) == out
