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
        "01420",
    }


def test_prep_model_codes_comprehensive_scenarios():
    """Test all permutations of Primary and Alternative codes to ensure
    priority and invalid handling work as expected.
    """
    # Define the 8 scenarios
    data = [
        # 1. Valid Primary, No Alt -> Keep Primary
        {"unique_id": "Case1", "initial": "86101", "alts": None},
        # 2. Invalid Primary, No Alt -> Invalid Primary
        {"unique_id": "Case2", "initial": "12345", "alts": None},
        # 3. Valid Primary, Valid Alt -> Keep Primary (Ignore Alt)
        {
            "unique_id": "Case3",
            "initial": "86101",
            "alts": [{"code": "01110", "likelihood": 1.0}],
        },
        # 4. Invalid Primary, Valid Alt -> Use Alt (Keep Primary in Invalid)
        {
            "unique_id": "Case4",
            "initial": "12345",
            "alts": [{"code": "01110", "likelihood": 1.0}],
        },
        # 5. Invalid Primary, Invalid Alt -> Both Invalid
        {
            "unique_id": "Case5",
            "initial": "12345",
            "alts": [{"code": "98765", "likelihood": 1.0}],
        },
        # 6. Missing Primary, Valid Alt -> Use Alt
        {
            "unique_id": "Case6",
            "initial": None,
            "alts": [{"code": "01110", "likelihood": 1.0}],
        },
        # 7. Missing Primary, Invalid Alt -> Invalid Alt
        {
            "unique_id": "Case7",
            "initial": None,
            "alts": [{"code": "98765", "likelihood": 1.0}],
        },
        # 8. Missing Primary, No Alt -> Empty
        {"unique_id": "Case8", "initial": None, "alts": None},
    ]

    df = pd.DataFrame(data).rename(
        columns={"initial": "initial_code", "alts": "alt_sic_candidates"}
    )

    # Run the function
    result = prep_model_codes(df, alt_codes_col="alt_sic_candidates")

    # Helper to get a row by ID
    def get_row(uid):
        return result.loc[result["unique_id"] == uid].iloc[0]

    # --- ASSERTIONS ---

    # Case 1: Standard Success
    r1 = get_row("Case1")
    assert r1[MODEL_COL] == {"86101"}
    assert r1[INVALID_MODEL_COL] == set()

    # Case 2: Standard Failure
    r2 = get_row("Case2")
    assert r2[MODEL_COL] == set()
    assert r2[INVALID_MODEL_COL] == {"12345"}

    # Case 3: Priority Check (Primary wins)
    r3 = get_row("Case3")
    assert r3[MODEL_COL] == {"86101"}
    assert "01110" not in r3[MODEL_COL]  # Alt ignored
    assert r3[INVALID_MODEL_COL] == set()

    # Case 4: Fallback Success (Primary invalid captured)
    r4 = get_row("Case4")
    assert r4[MODEL_COL] == {"01110"}  # Recovered via Alt
    assert r4[INVALID_MODEL_COL] == {"12345"}  # Primary still marked invalid

    # Case 5: Total Failure (Accumulates both errors)
    r5 = get_row("Case5")
    assert r5[MODEL_COL] == set()
    assert "12345" in r5[INVALID_MODEL_COL]
    assert "98765" in r5[INVALID_MODEL_COL]

    # Case 6: Standard Fill
    r6 = get_row("Case6")
    assert r6[MODEL_COL] == {"01110"}
    assert r6[INVALID_MODEL_COL] == set()

    # Case 7: Failed Fill
    r7 = get_row("Case7")
    assert r7[MODEL_COL] == set()
    assert r7[INVALID_MODEL_COL] == {"98765"}

    # Case 8: No Data
    r8 = get_row("Case8")
    assert r8[MODEL_COL] == set()
    assert r8[INVALID_MODEL_COL] == set()
