#!/usr/bin/env python3
"""This utility script reformats survey response data from intermediate parquet files
saved via the `retrieve_survey_responses.py` script into CSV files suitable for use by
clerical coders.

It reads chunked parquet files from a specified directory (local or GCS),
filters responses based on response validity and a timestamp, renames columns,
and then outputs two CSV files:
- A 'minimal' version with only the participant information and 3 TLFS fields.
- An 'extra' version with the survey-assist questions and responses included as well.

"""
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


COLUMN_NAME_MAPPING = {
    "survey_assist_interactions_0_response_found": "direct_lookup_successful",
    "survey_assist_interactions_0_response_code": "direct_lookup_code",
    "survey_assist_interactions_0_input_2_org_description": "org_description",
    "survey_assist_interactions_0_input_1_job_description": "job_description",
    "survey_assist_interactions_0_input_0_job_title": "job_title",
    "survey_assist_interactions_1_response_classified": "survey_assist_says_codable",
    "survey_assist_interactions_1_response_code": "survey_assist_assigned_code",
    "survey_assist_interactions_1_response_candidates_0_code": "survey_assist_alt_candidate_1_code",  # pylint: disable=line-too-long
    "survey_assist_interactions_1_response_candidates_1_code": "survey_assist_alt_candidate_2_code",  # pylint: disable=line-too-long
    "survey_assist_interactions_1_response_candidates_2_code": "survey_assist_alt_candidate_3_code",  # pylint: disable=line-too-long
    "survey_assist_interactions_1_response_candidates_3_code": "survey_assist_alt_candidate_4_code",  # pylint: disable=line-too-long
    "survey_assist_interactions_1_response_candidates_4_code": "survey_assist_alt_candidate_5_code",  # pylint: disable=line-too-long
    "survey_assist_interactions_1_response_follow_up_questions_0_text": "survey_assist_open_question",  # pylint: disable=line-too-long
    "survey_assist_interactions_1_response_follow_up_questions_0_response": "survey_assist_open_question_response",  # pylint: disable=line-too-long
    "survey_assist_interactions_1_response_follow_up_questions_1_text": "survey_assist_closed_question",  # pylint: disable=line-too-long
    "survey_assist_interactions_1_response_follow_up_questions_1_response": "survey_assist_closed_question_response",  # pylint: disable=line-too-long
    "survey_assist_interactions_1_response_follow_up_questions_1_select_options_0": "survey_assist_closed_question_opt_1",  # pylint: disable=line-too-long
    "survey_assist_interactions_1_response_follow_up_questions_1_select_options_1": "survey_assist_closed_question_opt_2",  # pylint: disable=line-too-long
    "survey_assist_interactions_1_response_follow_up_questions_1_select_options_2": "survey_assist_closed_question_opt_3",  # pylint: disable=line-too-long
    "survey_assist_interactions_1_response_follow_up_questions_1_select_options_3": "survey_assist_closed_question_opt_4",  # pylint: disable=line-too-long
    "survey_assist_interactions_1_response_follow_up_questions_1_select_options_4": "survey_assist_closed_question_opt_5",  # pylint: disable=line-too-long
    "survey_assist_interactions_1_response_follow_up_questions_1_select_options_5": "survey_assist_closed_question_opt_6",  # pylint: disable=line-too-long
}

CC_COLUMNS_MINIMAL = ["id", "user", "org_description", "job_description", "job_title"]

CC_COLUMNS_EXTRA = [
    *CC_COLUMNS_MINIMAL,
    "survey_assist_open_question",
    "survey_assist_open_question_response",
]

EVALUATION_COLUMNS = [
    *CC_COLUMNS_EXTRA,
    "direct_lookup_successful",
    "direct_lookup_code",
    "survey_assist_says_codable",
    "survey_assist_assigned_code",
    "survey_assist_alt_candidate_1_code",
    "survey_assist_alt_candidate_2_code",
    "survey_assist_alt_candidate_3_code",
    "survey_assist_alt_candidate_4_code",
    "survey_assist_alt_candidate_5_code",
    "survey_assist_closed_question_response",
    "survey_assist_closed_question_opt_1",
    "survey_assist_closed_question_opt_2",
    "survey_assist_closed_question_opt_3",
    "survey_assist_closed_question_opt_4",
    "survey_assist_closed_question_opt_5",
    "survey_assist_closed_question_opt_6",
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
    parser.add_argument(
        "--include_invalid",
        action="store_true",
        help="Include responses which had issues when parsing from the "
        "Firestore database collection. Default: False.",
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
        if not args.include_invalid:
            logger.debug(f"Marking valid responses for chunk {chunk_id}.")
            chunk["valid_response"] = chunk.apply(assign_response_valid, axis=1)
            logger.debug(f"Filtering out invalid responses for chunk {chunk_id}.")
            chunk = chunk[chunk["valid_response"]]
        logger.debug(
            f"Filtering chunk {chunk_id} for responses after {only_after_timestamp}."
        )
        logger.debug(f"Renaming columns for chunk {chunk_id}.")
        chunk = chunk.rename(columns=COLUMN_NAME_MAPPING)
        cc_chunks.append(chunk)
    cc_df = pd.concat(cc_chunks)
    if not args.include_invalid:
        logger.info("Marking duplicated responses...")
        cc_df["unique_response"] = cc_df.apply(
            lambda row: assign_response_unique(cc_df, row), axis=1
        )
        logger.info("Filtering out duplicated responses...")
        cc_df = cc_df[cc_df["unique_response"]]
        logger.info("Duplicated responses removed")

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
        cc_df["intermediate_feedback_column"] = cc_df.apply(
            lambda row: get_feedback(row, feedback_df), axis=1
        )
        for fc_name, fc_raw_name in zip(FEEDBACK_COLUMN_NAMES, FEEDBACK_COLUMNS):
            extraction_func = make_extract_feedback_field_func(fc_raw_name)
            cc_df[fc_name] = cc_df["intermediate_feedback_column"].apply(
                extraction_func
            )
        del cc_df["intermediate_feedback_column"]
        logger.debug("Completed merging the feedback data.")
        EVALUATION_COLUMNS.extend(FEEDBACK_COLUMN_NAMES)

    logger.info("Saving dataframes to CSV files...")
    cc_df[CC_COLUMNS_EXTRA].to_csv(f"{args.output_name_base}_extra.csv", index=False)
    cc_df[CC_COLUMNS_MINIMAL].to_csv(
        f"{args.output_name_base}_minimal.csv", index=False
    )
    cc_df[EVALUATION_COLUMNS].to_csv(
        f"{args.output_name_base}_evaluation.csv", index=False
    )
    logger.info(
        f"Saved dataframes to {args.output_name_base}_extra.csv, "
        f"{args.output_name_base}_minimal.csv and "
        f"{args.output_name_base}_evaluation.csv"
    )
    logger.info("Survey response reformatting finished.")
