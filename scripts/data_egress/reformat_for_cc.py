#!/usr/bin/env python3
"""This utility script reformats survey response data from intermediate parquet files
saved via the `retrieve_survey_responses.py` script into CSV files suitable for use by
clerical coders.

It reads chunked parquet files from a specified directory (local or GCS),
filters responses based on response validity and a timestamp, renames columns,
and then outputs two CSV files:
- A 'minimal' version with only the participant information and 3 TLFS fields.
- A 'full' version with the survey-assist questions and responses included as well.

"""
import json
import os
from argparse import ArgumentParser as AP
from datetime import datetime

import pandas as pd
from google.cloud import storage

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

CC_COLUMNS_FULL = [
    *CC_COLUMNS_MINIMAL,
    "direct_lookup_successful",
    "direct_lookup_code",
    "survey_assist_says_codable",
    "survey_assist_alt_candidate_1_code",
    "survey_assist_alt_candidate_2_code",
    "survey_assist_alt_candidate_3_code",
    "survey_assist_alt_candidate_4_code",
    "survey_assist_alt_candidate_5_code",
    "survey_assist_open_question",
    "survey_assist_open_question_response",
    "survey_assist_closed_question_opt_1",
    "survey_assist_closed_question_opt_2",
    "survey_assist_closed_question_opt_3",
    "survey_assist_closed_question_opt_4",
    "survey_assist_closed_question_opt_5",
    "survey_assist_closed_question_opt_6",
    "survey_assist_closed_question_response",
]


def setup_logger():
    """Set up the logger."""
    logger_tool = get_logger("data_egress", level=LOG_LEVEL.upper())
    return logger_tool


def setup_parser() -> AP:
    """Sets up a CLI parser."""
    parser = AP()
    parser.add_argument(
        "intermediate_files_path",
        type=str,
        help="path to the files output from the data egress process.",
    )
    parser.add_argument(
        "output_name_base",
        type=str,
        help="The base of the name of the output CSV files.",
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


if __name__ == "__main__":
    logger = setup_logger()
    cli_parser = setup_parser()
    args = cli_parser.parse_args()
    logger.debug("Parsed the CLI arguments")
    folder_is_in_gcp_bucket = args.intermediate_files_path.startswith("gs://")
    logger.debug("Loading the metadata file...")
    metadata = load_metadata(args.intermediate_files_path, folder_is_in_gcp_bucket)
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
    cc_full_chunks = []
    cc_minimal_chunks = []
    logger.info(f"Processing {metadata['number_of_chunks']} chunks...")
    for chunk_id in range(metadata["number_of_chunks"]):
        logger.debug(f"Processing chunk {chunk_id}...")
        chunk = grab_chunk(args.intermediate_files_path, chunk_id)
        if not args.include_invalid:
            logger.debug(f"Filtering out invalid responses for chunk {chunk_id}.")
            chunk = chunk[chunk["valid_response"]]
        logger.debug(
            f"Filtering chunk {chunk_id} for responses after {only_after_timestamp}."
        )
        chunk = chunk[chunk["time_start"] > only_after_timestamp]
        if len(chunk) == 0:
            logger.debug(f"Chunk {chunk_id} is empty after filtering, skipping.")
            continue
        logger.debug(f"Renaming columns for chunk {chunk_id}.")
        chunk = chunk.rename(columns=COLUMN_NAME_MAPPING)
        cc_minimal_chunks.append(chunk[CC_COLUMNS_MINIMAL])
        cc_full_chunks.append(chunk[CC_COLUMNS_FULL])
    cc_full_df = pd.concat(cc_full_chunks)
    cc_minimal_df = pd.concat(cc_minimal_chunks)
    cc_full_df.to_csv(f"{args.output_name_base}_full.csv", index=False)
    cc_minimal_df.to_csv(f"{args.output_name_base}_minimal.csv", index=False)
    logger.info(
        f"Saved dataframes to {args.output_name_base}_full.csv "
        f"and {args.output_name_base}_minimal.csv."
    )
    logger.info("Survey response reformatting finished.")
