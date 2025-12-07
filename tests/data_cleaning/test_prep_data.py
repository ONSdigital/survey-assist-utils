"""Tests for data preparation functions."""

# ignore pylint warnings about missing function docstrings and redefined outer names (fixtures)
# pylint: disable=C0116,W0621

import pandas as pd
import pytest

from survey_assist_utils.data_cleaning.prep_data import (
    prep_clerical_codes,
    prep_model_codes,
)

# Set some constants for the returned invalid column names:
CLERICAL_COL = "clerical_codes"
MODEL_COL = "model_codes"
INVALID_CLERICAL_COL = f"{CLERICAL_COL}_invalid"
INVALID_MODEL_COL = f"{MODEL_COL}_invalid"


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
    assert CLERICAL_COL in result.columns, "Output column missing"
    assert INVALID_CLERICAL_COL in result.columns
    assert len(result) == len(
        sample_cc_df
    ), "Unexpected number of rows after processing"
    assert (
        result[CLERICAL_COL].apply(lambda x: isinstance(x, set)).all()
    )  # All output codes should be sets
    assert result.loc[result["unique_id"] == "A1", CLERICAL_COL].iloc[0] == {
        "86101",
        "01420",
        "86210",
    }, "Incorrect codes for ID A1"
    # Expect no incorrect codes for A1:
    assert (
        result.loc[result["unique_id"] == "A1", INVALID_CLERICAL_COL].iloc[0] == set()
    )
    assert (
        result.loc[result["unique_id"] == "A3", CLERICAL_COL].iloc[0] == set()
    ), "Incorrect codes for ID A3"
    assert (
        result.loc[result["unique_id"] == "A4", CLERICAL_COL].iloc[0] == set()
    ), "Incorrect codes for ID A4"


def test_prep_clerical_with_invalid():
    df = pd.DataFrame(
        {
            "unique_id": ["B1"],
            "sic_ind_occ1": "98765",
            "sic_ind_occ2": "86101",
            "sic_ind_occ3": "23456",
        }
    )
    result = prep_clerical_codes(df)

    row = result.loc[result["unique_id"] == "B1"].iloc[0]
    assert "86101" in row[CLERICAL_COL]
    assert "98765" in row[INVALID_CLERICAL_COL]
    assert "23456" in row[INVALID_CLERICAL_COL]


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


def test_prep_model_codes_with_invalid():
    df = pd.DataFrame(
        {
            "unique_id": ["C1"],
            "initial_code": ["98765"],
            # Add alt candidates to ensure they don't overwrite the invalid column
            "alt_sic_candidates": [[{"code": "86101", "likelihood": 0.9}]],
        }
    )

    result = prep_model_codes(df, alt_codes_col="alt_sic_candidates")

    row = result.loc[result["unique_id"] == "C1"].iloc[0]

    # Ensure invalid code was captured
    assert "98765" in row[INVALID_MODEL_COL]

    # Ensure valid code from alternatives was still populated
    assert "86101" in row[MODEL_COL]


def test_prep_model_codes_initial_only():
    df = pd.DataFrame(
        {
            "unique_id": ["A1", "A2"],
            "initial_code": ["12345", "23456"],
        }
    )
    result = prep_model_codes(df)
    row1 = result.loc[result["unique_id"] == "A1"].iloc[0]
    row2 = result.loc[result["unique_id"] == "A2"].iloc[0]
    assert MODEL_COL in result.columns
    assert result[MODEL_COL].apply(lambda x: isinstance(x, set)).all()
    assert "12345" in row1[INVALID_MODEL_COL]
    assert "23456" in row2[INVALID_MODEL_COL]


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
    assert result[MODEL_COL].apply(lambda x: isinstance(x, set)).all()
    assert result[MODEL_COL].all()


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
    assert result.loc[result["unique_id"] == "A1", MODEL_COL].iloc[0] == {"86101"}
    assert result.loc[result["unique_id"] == "A2", MODEL_COL].iloc[0] == {
        "86210",
        "86101",
        "01420",  # this shouldn't be here.
    }
