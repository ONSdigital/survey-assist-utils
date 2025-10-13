"""Tests for data preparation functions."""

# ignore pylint warnings about missing function docstrings and redefined outer names (fixtures)
# pylint: disable=C0116,W0621

import pandas as pd
import pytest

from survey_assist_utils.data_cleaning.prep_data import (
    prep_clerical_codes,
    prep_model_codes,
)


@pytest.fixture
def sample_cc_df():
    return pd.DataFrame(
        [
            {
                "unique_id": "A1",
                "sic_ind_occ1": "86101",
                "sic_ind_occ2": "1420",
                "sic_ind_occ3": "86210",
            },
            {
                "unique_id": "A2",
                "sic_ind_occ1": "86210",
                "sic_ind_occ2": "663xx",
                "sic_ind_occ3": None,
            },
            {
                "unique_id": "A3",
                "sic_ind_occ1": "-9",
                "sic_ind_occ2": "nan",
                "sic_ind_occ3": "NAN",
            },
            {
                "unique_id": "A4",
                "sic_ind_occ1": "4+",
                "sic_ind_occ2": None,
                "sic_ind_occ3": None,
            },
        ]
    )


@pytest.fixture
def sample_cc_four_plus():
    return pd.DataFrame(
        {
            "unique_id": ["A4"],
            "sic_ind_occ": ["66210;66220;66290;663xx"],
        }
    )


def test_prep_clerical_codes_basic(sample_cc_df):
    result = prep_clerical_codes(sample_cc_df, digits=5)
    assert "clerical_codes" in result.columns, "Output column missing"
    assert len(result) == len(
        sample_cc_df
    ), "Unexpected number of rows after processing"
    assert (
        result["clerical_codes"].apply(lambda x: isinstance(x, set)).all()
    )  # All output codes should be sets
    assert result.loc[result["unique_id"] == "A1", "clerical_codes"].iloc[0] == {
        "86101",
        "01420",
        "86210",
    }, "Incorrect codes for ID A1"
    assert (
        result.loc[result["unique_id"] == "A3", "clerical_codes"].iloc[0] == set()
    ), "Incorrect codes for ID A3"
    assert (
        result.loc[result["unique_id"] == "A4", "clerical_codes"].iloc[0] == set()
    ), "Incorrect codes for ID A4"


def test_prep_clerical_codes_with_four_plus(sample_cc_df, sample_cc_four_plus):
    result = prep_clerical_codes(sample_cc_df, sample_cc_four_plus, digits=3)
    # Entries with four_plus should be replaced
    assert (
        result["clerical_codes"].apply(lambda x: isinstance(x, set)).all()
    )  # All output codes should be sets
    assert result.loc[result["unique_id"] == "A2", "clerical_codes"].iloc[0] == {
        "862",
        "663",
    }, "Incorrect codes for ID A2"
    assert result.loc[result["unique_id"] == "A4", "clerical_codes"].iloc[0] == {
        "662",
        "663",
    }, "Incorrect codes for ID A4"


def test_prep_clerical_codes_empty_df():
    df = pd.DataFrame(
        columns=["unique_id", "sic_ind_occ1", "sic_ind_occ2", "sic_ind_occ3"]
    )
    result = prep_clerical_codes(df)
    assert result.empty


def test_prep_model_codes_initial_only():
    df = pd.DataFrame(
        {
            "unique_id": ["A1", "A2"],
            "initial_code": ["12345", "23456"],
        }
    )
    result = prep_model_codes(df)
    assert "model_codes" in result.columns
    assert result["model_codes"].apply(lambda x: isinstance(x, set)).all()


def test_prep_model_codes_alt_only():
    df = pd.DataFrame(
        {
            "unique_id": ["A1", "A2"],
            "alt_sic_candidates": [
                [{"code": "86101", "likelihood": 0.9}],
                [{"code": "86210", "likelihood": 0.8}],
            ],
        }
    )
    result = prep_model_codes(df, codes_col=None, alt_codes_col="alt_sic_candidates")
    assert result["model_codes"].apply(lambda x: isinstance(x, set)).all()
    assert result["model_codes"].all()


def test_prep_model_codes_missing_id():
    df = pd.DataFrame(
        {
            "initial_code": ["12345"],
        }
    )
    with pytest.raises(ValueError):
        prep_model_codes(df)


def test_prep_model_codes_missing_cols():
    df = pd.DataFrame(
        {
            "unique_id": ["A1"],
        }
    )
    with pytest.raises(ValueError):
        prep_model_codes(df)


def test_prep_model_codes_threshold():
    df = pd.DataFrame(
        {
            "unique_id": ["A1", "A2"],
            "initial_code": ["", "-9"],
            "alt_sic_candidates": [
                [
                    {"code": "86101", "likelihood": 0.8},
                    {"code": "86210", "likelihood": 0.5},
                    {"code": "01420", "likelihood": 0.4},
                ],
                [
                    {"code": "86101", "likelihood": 0.8},
                    {"code": "86210", "likelihood": 0.7},
                    {"code": "01420", "likelihood": 0.4},
                ],
            ],
        }
    )
    result = prep_model_codes(
        df, codes_col=None, alt_codes_col="alt_sic_candidates", threshold=0.7
    )
    # Only codes with likelihood >= 0.7 should be present
    assert result.loc[result["unique_id"] == "A1", "model_codes"].iloc[0] == {"86101"}
    assert result.loc[result["unique_id"] == "A2", "model_codes"].iloc[0] == {
        "86210",
        "86101",
        "01420",
    }
