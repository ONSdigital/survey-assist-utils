"""Tests for evaluation metrics calculations."""

import pandas as pd
import pytest

from survey_assist_utils.evaluation.metrics import (
    calc_accuracy_metrics,
    calc_ambiguity_metrics,
    calc_codability_metrics,
    calc_simple_metrics,
)


def test_calc_ambiguity_metrics_basic():
    """Basic test for ambiguity metrics calculation."""
    df = pd.DataFrame(
        {
            "initial_ambiguous": [True, False, True, False],
            "clerical_ambiguous": [True, True, False, False],
        }
    )
    ambiguity_metrics = calc_ambiguity_metrics(df)
    assert ambiguity_metrics.TP == 1
    assert ambiguity_metrics.FP == 1
    assert ambiguity_metrics.FN == 1
    assert ambiguity_metrics.TN == 1
    assert ambiguity_metrics.precision == pytest.approx(0.5)
    assert ambiguity_metrics.recall == pytest.approx(0.5)
    assert ambiguity_metrics.f1 == pytest.approx(0.5)


def test_calc_codability_metrics_basic():
    """Basic test for codability metrics calculation."""
    df = pd.DataFrame(
        {
            "initial_ambiguous": [True, False, True, False],
            "final_ambiguous": [True, False, False, False],
        }
    )
    codability_metrics = calc_codability_metrics(df)
    assert (
        codability_metrics.initial_codable_count
        < codability_metrics.final_codable_count
    )
    assert codability_metrics.initial_codable_prop == pytest.approx(
        0.5
    ), "Initial codable proportion mismatch"
    assert codability_metrics.final_codable_prop == pytest.approx(
        0.75
    ), "Final codable proportion mismatch"
    assert codability_metrics.codability_improvement_prop == pytest.approx(
        0.25
    ), "Codability improvement proportion mismatch"


def test_calc_accuracy_metrics_basic():
    """Basic test for accuracy metrics calculation."""
    df = pd.DataFrame(
        {
            "sa_initial_codes": [{"A"}, {"B", "C"}, {"C"}, {"D"}],
            "clerical_codes": [{"A"}, {"B"}, {"C"}, {"E"}],
        }
    )
    accuracy_metrics = calc_accuracy_metrics(df)
    assert accuracy_metrics.total_records == df.shape[0]
    assert accuracy_metrics.matches_oo == df.shape[0] / 2
    assert accuracy_metrics.matches_mm == df.shape[0] - 1
    assert accuracy_metrics.accuracy_oo_total == pytest.approx(0.5)
    assert accuracy_metrics.accuracy_mm_total == pytest.approx(0.75)


def test_calc_simple_metrics_basic():
    """Basic test for simple metrics calculation."""
    df = pd.DataFrame(
        {
            "clerical_codes": ["A", "B", -9, "E"],
            "sa_initial_codes": ["A", ["B", "C"], None, {"D"}],
            "sa_final_codes": [["A"], ["B"], ["C"], ["D"]],
        }
    )
    simple_metrics = calc_simple_metrics(df)

    assert simple_metrics.ambiguity_metrics.f1 == pytest.approx(2 / 3)
    assert (
        simple_metrics.codability_metrics.codability_improvement_prop
        == pytest.approx(0.5)
    )
    assert (
        simple_metrics.initial_accuracy_metrics.accuracy_mo_unambiguous
        == pytest.approx(0.5)
    )
    assert (
        simple_metrics.final_accuracy_metrics.accuracy_om_unambiguous
        == pytest.approx(2 / 3)
    )
