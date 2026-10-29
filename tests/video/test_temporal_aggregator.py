"""Decision-boundary tests for DeepQShield temporal aggregation."""

from models.temporal_aggregator import TemporalAggregator


def test_calibrated_webcam_score_is_real():
    result = TemporalAggregator().aggregate([0.5928] * 5)

    assert result["video_fake_probability"] == 0.5928
    assert result["label"] == "REAL"
    assert result["confidence"] == 0.5928


def test_narrow_calibration_gap_abstains():
    result = TemporalAggregator().aggregate([0.625] * 5)

    assert result["label"] == "INDETERMINATE"


def test_borderline_webcam_score_requires_retry_not_deepfake_verdict():
    """A score below the validated fake region is inconclusive, not synthetic."""
    result = TemporalAggregator().aggregate([0.6569] * 19)

    assert result["label"] == "INDETERMINATE"
    assert result["decision_thresholds"]["deepfake_at_or_above"] == 0.70


def test_high_score_remains_a_deepfake_verdict():
    result = TemporalAggregator().aggregate([0.82] * 5)

    assert result["label"] == "DEEPFAKE"
    assert result["confidence"] == 0.82


def test_low_score_remains_a_real_verdict():
    result = TemporalAggregator().aggregate([0.14] * 5)

    assert result["label"] == "REAL"
    assert result["confidence"] == 0.86
