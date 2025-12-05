"""Test validation and deduplication of data extracted from Firestore."""

# pylint: disable=redefined-outer-name
import json

import pandas as pd
import pytest

from survey_assist_utils.data_cleaning.data_egress_validity_utils import (
    assign_response_unique,
    assign_response_valid,
)

situation_cases = {
    "direct_lookup": [
        "lookup_valid.json",
        "lookup_invalid_1.json",
        "lookup_invalid_2.json",
    ],
    "successful_classify": [
        "successful_classify_valid.json",
        "successful_classify_invalid_1.json",
        "successful_classify_invalid_2.json",
        "successful_classify_invalid_3.json",
        "successful_classify_invalid_4.json",
    ],
    "failed_classify": [
        "failed_classify_valid.json",
        "failed_classify_invalid_1.json",
        "failed_classify_invalid_2.json",
        "failed_classify_invalid_3.json",
        "failed_classify_invalid_4.json",
    ],
    "duplicates": [
        "duplicate_valid.json",
        "duplicate_invalid_1.json",
        "duplicate_invalid_2.json",
    ],
}


@pytest.fixture(scope="module")
def get_relevant_fake_records(request) -> list[pd.DataFrame]:
    """Loads the example situations as Pandas DataFrames."""
    situation = request.param
    dfs = []
    for example in situation_cases.get(situation, []):
        with open(
            f"tests/data_egress_validity_test_cases/{example}",
            encoding="utf8",
        ) as f:
            example_as_dict = json.load(f)
        dfs.append(pd.DataFrame(example_as_dict))
    return dfs


@pytest.mark.parametrize(
    "get_relevant_fake_records",
    ["direct_lookup", "successful_classify", "failed_classify"],
    indirect=True,
)
def test_validity(get_relevant_fake_records: list[pd.DataFrame]):
    """This test will run for each of the three main 'paths' for the survey;
    - direct lookup success,
    - SA classification success,
    - SA classification failure.
    The first record for each case is expected to be valid, the rest invalid.
    """
    list_of_dfs = get_relevant_fake_records
    list_of_dfs[0]["valid_response"] = list_of_dfs[0].apply(
        assign_response_valid, axis=1
    )
    assert list_of_dfs[0].iloc[0][
        "valid_response"
    ], "first example is valid in each situation"
    for example_id, df in enumerate(list_of_dfs[1:]):
        df["valid_response"] = df.apply(assign_response_valid, axis=1)
        print(example_id + 1, df["valid_response"])
        assert not df.iloc[0][
            "valid_response"
        ], "all other examples are invalid in each situation"


@pytest.mark.parametrize("get_relevant_fake_records", ["duplicates"], indirect=True)
def test_duplication(get_relevant_fake_records: list[pd.DataFrame]):
    """Tests that deduplication is performed correctly.
    The first example contains no duplication, the rest have duplication.
    """
    list_of_dfs = get_relevant_fake_records
    for i, _ in enumerate(list_of_dfs):
        list_of_dfs[i]["unique_response"] = [
            assign_response_unique(list_of_dfs[i], row)
            for __, row in list_of_dfs[i].iterrows()
        ]

    assert list_of_dfs[0].iloc[0]["unique_response"], "first example has no duplication"
    assert list_of_dfs[0].iloc[1]["unique_response"], "first example has no duplication"

    assert not list_of_dfs[1].iloc[0][
        "unique_response"
    ], "second example has duplicate, second record is first"
    assert list_of_dfs[1].iloc[1][
        "unique_response"
    ], "second example has duplicate, second record is first"

    assert list_of_dfs[2].iloc[0][
        "unique_response"
    ], "third example has duplicate, first record is first"
    assert not list_of_dfs[2].iloc[1][
        "unique_response"
    ], "third example has duplicate, first record is first"
