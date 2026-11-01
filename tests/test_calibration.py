import numpy as np
import pytest
from src.calibration import fit_platt_calibrator, model_output_to_score


def test_probability_output_is_not_sigmoided_again():
    assert model_output_to_score(0.8, output_type="probability") == pytest.approx(0.8)


def test_raw_margin_is_sigmoided():
    assert model_output_to_score(0.0, output_type="raw_margin") == pytest.approx(0.5)


def test_platt_calibrator_maps_monotonically():
    scores = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    calibrator = fit_platt_calibrator(scores, labels)
    assert model_output_to_score(-2.0, calibrator=calibrator) < model_output_to_score(2.0, calibrator=calibrator)


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf")])
def test_probability_output_rejects_invalid_values(score):
    with pytest.raises(ValueError):
        model_output_to_score(score, output_type="probability")
