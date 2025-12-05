#!/usr/bin/env python3
"""This utility script reformats survey response data from intermediate parquet files
saved via the `retrieve_survey_responses.py` script into CSV files suitable for use by
clerical coders.

It reads chunked parquet files from a specified directory (local or GCS),
filters responses based on response validity and a timestamp, renames columns,
and then outputs three-four CSV files:
- A 'minimal' version with only the participant information and 3 TLFS fields.
- An 'extra' version with the survey-assist questions and responses included as well.
- An 'evaluation' version with all fields included for analysis purposes.
- An 'invalid' version with responses that were marked as invalid or duplicated.
- A 'not employed' version with responses that were not in employment.
"""
# pylint: disable=line-too-long,C0103,C0121
# ruff: noqa: E712

import json
import os
from argparse import ArgumentParser as AP
from datetime import datetime

import pandas as pd
from google.cloud import storage

from survey_assist_utils.data_cleaning.data_egress_validity_utils import (
    assign_response_unique,
    assign_response_valid,
)
from survey_assist_utils.logging import (
    get_logger,
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")

job_title_col = "job_title"
job_description_col = "job_description"
org_description_col = "org_description"
lookup_classified_col = "direct_lookup_classified"
lookup_code_col = "direct_lookup_assigned_code"
survey_assist_classified_col = "survey_assist_classified"
survey_assist_code_col = "survey_assist_assigned_code"
survey_assist_alt_code_cols = "survey_assist_alt_candidate_code_"
open_question_col = "survey_assist_open_question"
open_question_response_col = "survey_assist_open_question_response"
closed_question_response_col = "survey_assist_closed_question_response"
closed_question_opt_cols = "survey_assist_closed_question_option_"


COLUMN_NAME_MAPPING = {
    "id": "unique_id",
    "survey_assist_interactions_0_response_found": lookup_classified_col,
    "survey_assist_interactions_0_response_code": lookup_code_col,
    "survey_assist_interactions_0_input_2_org_description": org_description_col,
    "survey_assist_interactions_0_input_1_job_description": job_description_col,
    "survey_assist_interactions_0_input_0_job_title": job_title_col,
    "survey_assist_interactions_1_response_classified": survey_assist_classified_col,
    "survey_assist_interactions_1_response_code": survey_assist_code_col,
    "survey_assist_interactions_1_response_candidates_0_code": survey_assist_alt_code_cols
    + "1",
    "survey_assist_interactions_1_response_candidates_1_code": survey_assist_alt_code_cols
    + "2",
    "survey_assist_interactions_1_response_candidates_2_code": survey_assist_alt_code_cols
    + "3",
    "survey_assist_interactions_1_response_candidates_3_code": survey_assist_alt_code_cols
    + "4",
    "survey_assist_interactions_1_response_candidates_4_code": survey_assist_alt_code_cols
    + "5",
    "survey_assist_interactions_1_response_follow_up_questions_0_text": open_question_col,
    "survey_assist_interactions_1_response_follow_up_questions_0_response": open_question_response_col,
    "survey_assist_interactions_1_response_follow_up_questions_1_response": closed_question_response_col,
    "survey_assist_interactions_1_response_follow_up_questions_1_select_options_0": closed_question_opt_cols
    + "1",
    "survey_assist_interactions_1_response_follow_up_questions_1_select_options_1": closed_question_opt_cols
    + "2",
    "survey_assist_interactions_1_response_follow_up_questions_1_select_options_2": closed_question_opt_cols
    + "3",
    "survey_assist_interactions_1_response_follow_up_questions_1_select_options_3": closed_question_opt_cols
    + "4",
    "survey_assist_interactions_1_response_follow_up_questions_1_select_options_4": closed_question_opt_cols
    + "5",
    "survey_assist_interactions_1_response_follow_up_questions_1_select_options_5": closed_question_opt_cols
    + "6",
}

CC_COLUMNS_MINIMAL = [
    "unique_id",
    "user",
    job_title_col,
    job_description_col,
    org_description_col,
]

CC_COLUMNS_EXTRA = [
    *CC_COLUMNS_MINIMAL,
    open_question_col,
    open_question_response_col,
]

EVALUATION_COLUMNS = [
    *CC_COLUMNS_EXTRA,
    lookup_classified_col,
    lookup_code_col,
    survey_assist_classified_col,
    survey_assist_code_col,
    survey_assist_alt_code_cols + "1",
    survey_assist_alt_code_cols + "2",
    survey_assist_alt_code_cols + "3",
    survey_assist_alt_code_cols + "4",
    survey_assist_alt_code_cols + "5",
    closed_question_response_col,
    closed_question_opt_cols + "1",
    closed_question_opt_cols + "2",
    closed_question_opt_cols + "3",
    closed_question_opt_cols + "4",
    closed_question_opt_cols + "5",
    closed_question_opt_cols + "6",
    "response_valid",
    "response_unique",
    "not_in_employment_proxy",
]

FEEDBACK_COLUMNS = [
    "questions_0_response",
    "questions_1_response",
    "questions_2_response",
    "questions_3_response",
    "questions_4_response",
]

FEEDBACK_COLUMN_NAMES = [
    "feedback_age_range",
    "feedback_survey_ease",
    "feedback_survey_relevance",
    "feedback_survey_comfort",
    "feedback_comments",
]


def setup_logger():
    """Set up the logger."""
    logger_tool = get_logger("data_egress", level=LOG_LEVEL.upper())
    return logger_tool


def setup_parser() -> AP:
    """Sets up a CLI parser."""
    parser = AP()
    parser.add_argument(
        "intermediate_responses_path",
        type=str,
        help="path to the folder containing the files output "
        "from the response data egress process.",
    )
    parser.add_argument(
        "output_name_base",
        type=str,
        help="The base of the name of the output CSV files.",
    )
    parser.add_argument(
        "--intermediate_feedback_path",
        type=str,
        default="",
        help="path to the folder containing the files output "
        "from the feedback data egress process.",
    )
    parser.add_argument(
        "--only_after",
        type=str,
        default="2024_01_01__00_00_000000",
        help="Restrict results to those collected after specified timestamp. "
        "Format Y_m_d__H_M_S (e.g. '2024_01_01__00_00_000000').",
    )
    return parser


def grab_chunk(path_to_folder: str, current_chunk_id: int) -> pd.DataFrame:
    """Loads a specific data chunk from a parquet file.

    Args:
        path_to_folder (str): The path to the directory containing the chunk files.
        current_chunk_id (int): The ID of the chunk to load.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the chunk.
    """
    return pd.read_parquet(f"{path_to_folder}/chunk_{current_chunk_id}.parquet")


def load_metadata(path_to_folder: str, use_gcp=False) -> dict:
    """Loads the metadata file from a local or GCS path.

    Args:
        path_to_folder (str): The path to the directory containing the metadata.json file.
            Can be a local path or a GCS URI.
        use_gcp (bool): Flag indicating whether the path is a GCS URI. Defaults to False.

    Returns:
        dict: The loaded metadata as a dictionary.
    """
    if use_gcp:
        client = storage.Client()
        bucket = client.bucket(path_to_folder.removeprefix("gs://").split("/")[0])
        blob = bucket.blob(
            f"{'/'.join(path_to_folder.removeprefix('gs://').split('/')[1:])}/metadata.json"
        )
        return json.loads(blob.download_as_bytes().decode("utf-8"))
    with open(f"{path_to_folder}/metadata.json", encoding="utf8") as f:
        return json.load(f)


def get_feedback(row: pd.Series, f_df: pd.DataFrame) -> dict:
    """Extracts the corresponding feedback for a given row.
    If there is no corresponding feedback, or if there are multiple
    feedback entries, for a given respondent it returns an empty dict.

    Args:
        row (pd.Series): a row in the 'responses' dataframe.
        f_df (pd.DataFrame): a dataframe containing the feedback data.

    Returns:
        dict: the corresponding row in the 'feedback' dataframe,
              converted to a dict, or an empty dict if there is
              no corresponding feedback.
    """
    matches = f_df[f_df["person_id"] == row["person_id"]]
    if len(matches) == 1:
        return matches.iloc[0].to_dict()
    return {}


def make_extract_feedback_field_func(field):
    """Generates a mapping function to be applied to the intermediate feedback
    column to extract a given field.

    Args:
        field (str): The name of the field to extract.

    """
    return lambda row: row.get(field, "")


if __name__ == "__main__":
    logger = setup_logger()
    cli_parser = setup_parser()
    args = cli_parser.parse_args()
    logger.debug("Parsed the CLI arguments")
    folder_is_in_gcp_bucket = args.intermediate_responses_path.startswith("gs://")
    logger.debug("Loading the responses metadata file...")
    responses_metadata = load_metadata(
        args.intermediate_responses_path, folder_is_in_gcp_bucket
    )
    logger.debug("Metadata loaded successfully.")
    start_date, start_time = args.only_after.split("__")
    only_after_timestamp = pd.Timestamp(
        datetime(
            *[int(i) for i in start_date.split("_")],  # type: ignore[arg-type]
            *[int(i) for i in start_time.split("_")],  # type: ignore[arg-type]
        ),
        tz="UTC",
        unit="ns",
    )
    logger.debug(
        f"restricting output to responses entered after {only_after_timestamp}"
    )
    cc_chunks = []
    logger.info(f"Processing {responses_metadata['number_of_chunks']} chunks...")
    for chunk_id in range(responses_metadata["number_of_chunks"]):
        logger.debug(f"Processing chunk {chunk_id}...")
        chunk = grab_chunk(args.intermediate_responses_path, chunk_id)
        chunk = chunk[chunk["time_start"] > only_after_timestamp]
        if len(chunk) == 0:
            logger.debug(f"Chunk {chunk_id} is empty after filtering, skipping.")
            continue
        logger.debug(
            f"Filtering chunk {chunk_id} for responses after {only_after_timestamp}."
        )
        cc_chunks.append(chunk)
    merged_df = pd.concat(cc_chunks)
    merged_df = merged_df.where(merged_df.notnull(), None)
    logger.info(
        f"Completed processing all chunks. Number of responses: {len(merged_df)}"
    )

    # A user pressing 'no' for the 'paid employment' question results in a null value
    # being saved for the interaction 0 response. However it is possible for other
    # issues (associated with the back button issue) to also cause this behaviour.
    # We use it as a proxy measure for if they were in employment, but note that it
    # is an imperfect measure.
    merged_df["not_in_employment_proxy"] = merged_df[
        "survey_assist_interactions_0_response_found"
    ].isna()

    merged_df["response_valid"] = None
    employed_mask = merged_df["not_in_employment_proxy"] == False
    merged_df.loc[employed_mask, "response_valid"] = merged_df[employed_mask].apply(
        assign_response_valid, axis=1
    )

    duplication_status = []
    for _, r in merged_df.iterrows():
        if r["response_valid"]:
            duplication_status.append(
                assign_response_unique(
                    merged_df[merged_df["response_valid"] == True], r
                )
            )
        else:
            duplication_status.append(None)  # type: ignore[arg-type]
    merged_df["response_unique"] = duplication_status

    if args.intermediate_feedback_path != "":
        logger.info("Loading the feedback metadata file...")
        folder_is_in_gcp_bucket = args.intermediate_feedback_path.startswith("gs://")
        feedback_metadata = load_metadata(
            args.intermediate_feedback_path, folder_is_in_gcp_bucket
        )
        logger.info("Metadata loaded successfully.")
        feedback_chunks = []
        logger.info(f"Processing {feedback_metadata['number_of_chunks']} chunks...")
        for chunk_id in range(feedback_metadata["number_of_chunks"]):
            logger.debug(f"Processing chunk {chunk_id}...")
            chunk = grab_chunk(args.intermediate_feedback_path, chunk_id)
            feedback_chunks.append(chunk)
        feedback_df = pd.concat(feedback_chunks)
        logger.info("Merging in the feedback data...")
        merged_df["intermediate_feedback_column"] = merged_df.apply(
            lambda r: get_feedback(r, feedback_df), axis=1
        )
        for fc_name, fc_raw_name in zip(FEEDBACK_COLUMN_NAMES, FEEDBACK_COLUMNS):
            extraction_func = make_extract_feedback_field_func(fc_raw_name)
            merged_df[fc_name] = merged_df["intermediate_feedback_column"].apply(
                extraction_func
            )
        del merged_df["intermediate_feedback_column"]
        logger.debug("Completed merging the feedback data.")
        EVALUATION_COLUMNS.extend(FEEDBACK_COLUMN_NAMES)

    merged_df = merged_df.rename(columns=COLUMN_NAME_MAPPING)

    valid_unique_employed_mask = (
        (merged_df["response_valid"] == True)
        & (merged_df["response_unique"] == True)
        & (merged_df["not_in_employment_proxy"] == False)
    )
    valid_unique_employed_df = merged_df[valid_unique_employed_mask]

    invalid_or_duplicated_employed_mask = (
        merged_df["not_in_employment_proxy"] == False
    ) & (
        (merged_df["response_valid"] == False) | (merged_df["response_unique"] == False)
    )

    invalid_or_duplicate_df = merged_df[invalid_or_duplicated_employed_mask]
    not_employed_df = merged_df[merged_df["not_in_employment_proxy"]]

    logger.info("Saving dataframes to CSV files...")
    valid_unique_employed_df[EVALUATION_COLUMNS].to_csv(
        f"{args.output_name_base}_evaluation.csv", index=False
    )
    valid_unique_employed_df[EVALUATION_COLUMNS].to_parquet(
        f"{args.output_name_base}_evaluation.parquet", index=False
    )
    valid_unique_employed_df[CC_COLUMNS_MINIMAL].to_csv(
        f"{args.output_name_base}_minimal.csv", index=False
    )
    valid_unique_employed_df.loc[
        ~valid_unique_employed_df["survey_assist_open_question"].isna(),
        CC_COLUMNS_EXTRA,
    ].to_csv(f"{args.output_name_base}_extra.csv", index=False)

    invalid_or_duplicate_df.to_csv(f"{args.output_name_base}_invalid.csv", index=False)
    invalid_or_duplicate_df.to_parquet(
        f"{args.output_name_base}_invalid.parquet", index=False
    )
    logger.info(
        f"Saved invalid responses to {args.output_name_base}_invalid<.csv/.parquet>"
    )

    not_employed_df.to_csv(f"{args.output_name_base}_not_employed.csv", index=False)
    not_employed_df.to_parquet(
        f"{args.output_name_base}_not_employed.parquet", index=False
    )
    logger.info(
        f"Saved not-employed responses to {args.output_name_base}_not_employed<.csv/.parquet>"
    )

    logger.info(
        f"Saved dataframes to {args.output_name_base}_extra.csv, "
        f"{args.output_name_base}_minimal.csv and "
        f"{args.output_name_base}_evaluation<.csv/.parquet>"
    )
    logger.info("Survey response reformatting finished.")

    print(
        f"""
EXPORT SUMMARY
--------------------------------
total:                   {len(merged_df)}
--------------------------------
valid, unique, employed: {len(valid_unique_employed_df)}
invalid / duplicate:     {len(invalid_or_duplicate_df)}
invalid:                 {len(invalid_or_duplicate_df[invalid_or_duplicate_df['response_valid'] == False])}
duplicate:               {len(invalid_or_duplicate_df[invalid_or_duplicate_df['response_unique'] == False])}
not employed:            {len(not_employed_df)}

Note: some records may be simultaneously invalid, duplicates,
      and/or not in employment.
    """
    )
