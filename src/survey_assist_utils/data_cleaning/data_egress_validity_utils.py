#!/usr/bin/env python
# pylint: disable=pointless-string-statement
"""A collection of helper functions for validating survey response data.

This module provides functions to check for the validity and uniqueness of survey
responses based on a set of predefined rules. The rules are categorized into
conditions that must always be met, and specific cases for different response
scenarios.

Functions:
- assign_response_valid: Determine if a survey response is valid.
- assign_response_unique: Check if a response is the latest one from a specific user.
"""

import pandas as pd

"""
*Always*
- first interaction should exist.
- survey_assist_interactions_0_response_found should not be missing
- survey_assist_interactions_0_type == 'lookup'
- org_description should not be missing
- job_description should not be missing
- job_title should not be missing
- there should be no more than 2 interactions

*Case 1: lookup successful*
- should have no second interaction at all
- survey_assist_interactions_0_response_found == True
- survey_assist_interactions_0_response_code should not be missing

*Case 2: lookup unsuccessful, initial classification successful*
- survey_assist_interactions_0_response_found == False
- survey_assist_interactions_0_response_code should be missing
- survey_assist_interactions_1_type == 'classify'
- survey_assist_interactions_1_response_classified == True
- survey_assist_interactions_1_response_code should not be missing
- survey_assist_interactions_1_response_candidates_0_code should not be missing
- survey_assist_interactions_1_response_follow_up_questions_0_text should be missing
- survey_assist_interactions_1_response_follow_up_questions_0_response should be missing
- survey_assist_interactions_1_response_follow_up_questions_1_text should be missing
- survey_assist_interactions_1_response_follow_up_questions_1_response should be missing

*Case 3: lookup unsuccessful, initial classification unsuccessful*
- survey_assist_interactions_0_response_found == False
- survey_assist_interactions_0_response_code should be missing
- survey_assist_interactions_1_type == 'classify'
- survey_assist_interactions_1_response_classified == False
- survey_assist_interactions_1_response_code should be missing
- survey_assist_interactions_1_response_candidates_0_code should not be missing
- survey_assist_interactions_1_response_candidates_1_code should not be missing
- survey_assist_interactions_1_response_follow_up_questions_0_text should not be missing
- survey_assist_interactions_1_response_follow_up_questions_0_response should not be missing
- survey_assist_interactions_1_response_follow_up_questions_1_text should not be missing
- survey_assist_interactions_1_response_follow_up_questions_1_response should not be missing
- survey_assist_interactions_1_response_follow_up_questions_1_select_options_0 should not be missing
- survey_assist_interactions_1_response_follow_up_questions_1_select_options_1 should not be missing
- survey_assist_interactions_1_response_follow_up_questions_0_id should be f1.1
- survey_assist_interactions_1_response_follow_up_questions_1_id should be f1.2
- survey_assist_interactions_1_response_follow_up_questions_2_id should be missing
"""


def _check_always_required(  # noqa: PLR0911 # pylint: disable=too-many-return-statements
    row: pd.Series,
) -> bool:
    """Check for conditions that must always be met for a response to be valid.

    Args:
        row: A pandas Series representing a single survey response.

    Returns:
        True if the response meets the always-required conditions, False otherwise.
    """
    if not all(
        (
            "survey_assist_interactions_0_type" in row,
            "survey_assist_interactions_0_response_found" in row,
            "survey_assist_interactions_0_input_2_org_description" in row,
            "survey_assist_interactions_0_input_1_job_description" in row,
            "survey_assist_interactions_0_input_0_job_title" in row,
        )
    ):
        return False
    if any(
        (
            row["survey_assist_interactions_0_type"] != "lookup",
            row["survey_assist_interactions_0_response_found"] in ("", None),
            row["survey_assist_interactions_0_input_2_org_description"] in ("", None),
            row["survey_assist_interactions_0_input_1_job_description"] in ("", None),
            row["survey_assist_interactions_0_input_0_job_title"] in ("", None),
        )
    ):
        return False
    if "survey_assist_interactions_2_type" in row and (
        row["survey_assist_interactions_2_type"] not in ("", None)
    ):
        return False
    if "survey_assist_interactions_3_type" in row and (
        row["survey_assist_interactions_3_type"] not in ("", None)
    ):
        return False
    if "survey_assist_interactions_4_type" in row and (
        row["survey_assist_interactions_4_type"] not in ("", None)
    ):
        return False
    if "survey_assist_interactions_5_type" in row and (
        row["survey_assist_interactions_5_type"] not in ("", None)
    ):
        return False
    if "survey_assist_interactions_1_response_follow_up_questions_2_id" in row and (
        row["survey_assist_interactions_1_response_follow_up_questions_2_id"]
        not in ("", None)
    ):
        return False
    if "survey_assist_interactions_1_response_follow_up_questions_3_id" in row and (
        row["survey_assist_interactions_1_response_follow_up_questions_3_id"]
        not in ("", None)
    ):
        return False
    if (  # noqa: SIM103
        "survey_assist_interactions_1_response_follow_up_questions_4_id" in row
        and (
            row["survey_assist_interactions_1_response_follow_up_questions_4_id"]
            not in ("", None)
        )
    ):
        return False
    return True


def _check_case_1(row: pd.Series) -> bool:
    """Check for validity when the initial lookup was successful (Case 1).

    Args:
        row: A pandas Series representing a single survey response.

    Returns:
        True if the response is a valid 'Case 1' response, False otherwise.
    """
    return all(
        (
            row["survey_assist_interactions_1_type"] in ("", None),
            row["survey_assist_interactions_1_response_classified"] in ("", None),
            row["survey_assist_interactions_0_response_found"] is True,
            row["survey_assist_interactions_0_response_code"] not in ("", None),
        )
    )


def _check_case_2(row: pd.Series) -> bool:
    """Check for validity when lookup was unsuccessful but classification was successful (Case 2).

    Args:
        row: A pandas Series representing a single survey response.

    Returns:
        True if the response is a valid 'Case 2' response, False otherwise.
    """
    return all(
        (
            row["survey_assist_interactions_0_response_found"] is False,
            row["survey_assist_interactions_0_response_code"] in ("", None),
            row["survey_assist_interactions_1_type"] == "classify",
            row["survey_assist_interactions_1_response_classified"] is True,
            row["survey_assist_interactions_1_response_code"] not in ("", None),
            row["survey_assist_interactions_1_response_candidates_0_code"]
            not in ("", None),
            row["survey_assist_interactions_1_response_follow_up_questions_0_text"]
            in ("", None),
            row["survey_assist_interactions_1_response_follow_up_questions_1_text"]
            in ("", None),
            row["survey_assist_interactions_1_response_follow_up_questions_0_response"]
            in ("", None),
            row["survey_assist_interactions_1_response_follow_up_questions_1_response"]
            in ("", None),
        )
    )


def _check_case_3(row: pd.Series) -> bool:
    """Check for validity when both lookup and classification were unsuccessful (Case 3).

    Args:
        row: A pandas Series representing a single survey response.

    Returns:
        True if the response is a valid 'Case 3' response, False otherwise.
    """
    return all(
        (
            row["survey_assist_interactions_0_response_found"] is False,
            row["survey_assist_interactions_0_response_code"] in ("", None),
            row["survey_assist_interactions_1_type"] == "classify",
            row["survey_assist_interactions_1_response_classified"] is False,
            row["survey_assist_interactions_1_response_code"] in ("", None),
            row["survey_assist_interactions_1_response_candidates_0_code"]
            not in ("", None),
            row["survey_assist_interactions_1_response_candidates_1_code"]
            not in ("", None),
            row["survey_assist_interactions_1_response_follow_up_questions_0_id"]
            == "f1.1",
            row["survey_assist_interactions_1_response_follow_up_questions_1_id"]
            == "f1.2",
            row["survey_assist_interactions_1_response_follow_up_questions_0_text"]
            not in ("", None),
            row["survey_assist_interactions_1_response_follow_up_questions_1_text"]
            not in ("", None),
            row["survey_assist_interactions_1_response_follow_up_questions_0_response"]
            not in ("", None),
            row["survey_assist_interactions_1_response_follow_up_questions_1_response"]
            not in ("", None),
        )
    )


def assign_response_valid(df_row: pd.Series) -> bool:
    """Determine if a survey response is valid by checking against predefined cases.

    It first checks for universally required fields and then delegates to
    case-specific validation functions.

    Args:
        df_row: A pandas Series representing a single survey response.

    Returns:
        True if the response is valid according to one of the cases, False otherwise.
    """
    # check the 'always required' conditions
    if not _check_always_required(df_row):
        return False

    # check if case 1 is appropriate
    if df_row["survey_assist_interactions_0_response_found"] is True:
        return _check_case_1(df_row)

    # check if case 2 is appropriate
    if (
        df_row["survey_assist_interactions_0_response_found"] is False
        and df_row["survey_assist_interactions_1_response_classified"] is True
    ):
        return _check_case_2(df_row)

    # check if case 3 is appropriate
    if (
        df_row["survey_assist_interactions_0_response_found"] is False
        and df_row["survey_assist_interactions_1_response_classified"] is False
    ):
        return _check_case_3(df_row)
    # those cases should be exhaustive, so anything else indicates an issue
    return False


def assign_response_unique(df: pd.DataFrame, row: pd.Series) -> bool:
    """Check if a response is the latest one from a specific user.

    If multiple responses exist for the same user, this function marks only the
    most recent one (by 'time_start') as unique.

    Args:
        df: The DataFrame containing all survey responses.
        row: The specific response (as a pandas Series) to check for uniqueness.

    Raises:
        ValueError: If the row's user cannot be found in the main DataFrame.

    Returns:
        True if the response is the only one or the latest one for the user.
    """
    unique_responses = df[df["user"] == row["user"]]
    if len(unique_responses) == 1:
        return True
    if len(unique_responses) == 0:
        raise ValueError(
            f"row {row.name}'s user ({row['user']}) could not be found in overall dataframe"
        )
    if len(unique_responses) > 1:
        for duplicate_response in unique_responses.iterrows():
            if duplicate_response[0] == row.name:
                continue
            if duplicate_response[1]["time_start"] > row["time_start"]:
                return False
    return True
