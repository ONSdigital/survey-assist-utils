"""Handles the processing of json files."""

import json
import os
from datetime import datetime

import pandas as pd
from google.cloud import storage


def flatten_llm_json_to_dataframe(
    file_path: str, max_candidates: int = 5
) -> pd.DataFrame:
    """Reads LLM response JSON data from a file, flattens it into a Pandas DataFrame.

    Args:
        file_path (str): The path to the JSON file.
        max_candidates (int): The maximum number of SIC candidates to flatten per record.

    Returns:
        pd.DataFrame: A Pandas DataFrame with the flattened JSON data.
    """
    all_flat_records = []

    try:
        with open(file_path, encoding="utf-8") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return pd.DataFrame()

    # Ensure json_data is a list of records
    if isinstance(json_data, dict):
        records_to_process = [
            json_data
        ]  # Handle case where file contains a single JSON object
    elif isinstance(json_data, list):
        records_to_process = json_data
    else:
        print("Error: JSON content is not a list or a single object (dictionary).")
        return pd.DataFrame()

    for record in records_to_process:
        flat_record = {}

        # 1. Add top-level fields
        flat_record["unique_id"] = record.get("unique_id")
        flat_record["classified"] = record.get("classified")
        flat_record["followup"] = record.get("followup")
        # Rename top-level sic_code & sic_description to avoid clashes with candidate fields
        flat_record["chosen_sic_code"] = record.get("sic_code")
        flat_record["chosen_sic_description"] = record.get("sic_description")
        flat_record["reasoning"] = record.get("reasoning")

        # 2. Flatten request_payload
        payload = record.get(
            "request_payload", {}
        )  # Default to empty dict if payload is missing
        flat_record["payload_llm"] = payload.get("llm")
        flat_record["payload_type"] = payload.get("type")
        flat_record["payload_job_title"] = payload.get("job_title")
        flat_record["payload_job_description"] = payload.get("job_description")
        flat_record["payload_industry_descr"] = payload.get("industry_descr")

        # 3. Flatten sic_candidates
        candidates = record.get("sic_candidates", [])  # Default to empty list
        for i in range(max_candidates):
            if i < len(candidates) and isinstance(candidates[i], dict):
                candidate = candidates[i]
                flat_record[f"candidate_{i+1}_sic_code"] = candidate.get("sic_code")
                flat_record[f"candidate_{i+1}_sic_descriptive"] = candidate.get(
                    "sic_descriptive"
                )
                flat_record[f"candidate_{i+1}_likelihood"] = candidate.get("likelihood")
            else:
                # Fill with None if fewer than max_candidates or candidate data is malformed
                flat_record[f"candidate_{i+1}_sic_code"] = None
                flat_record[f"candidate_{i+1}_sic_descriptive"] = None
                flat_record[f"candidate_{i+1}_likelihood"] = None

        all_flat_records.append(flat_record)

    df = pd.DataFrame(all_flat_records)

    return df


def list_files_from_date(directory, date_str):
    """Lists the copied json files from the local store needed for this analysis."""
    # Convert the given date string to a datetime object
    given_date = datetime.strptime(date_str, "%Y%m%d")

    # List to store filenames with a later date
    later_files = []

    # Iterate over files in the directory
    for filename in os.listdir(directory):
        # Check if the filename matches the pattern 'YYYYMMDD_HHMMSS_output.json'
        if filename.endswith("_output.json"):
            file_date_str = filename[:8]
            try:
                file_date = datetime.strptime(file_date_str, "%Y%m%d")
                # Check if the file date is later than the given date
                if file_date >= given_date:
                    later_files.append(filename)
            except ValueError:
                # Skip files with invalid date format
                continue

    return later_files


# Example usage
DIRECTORY_STORE = "data/json_runs"  # Current directory
DATE_STR_LOCAL = "20250522"
needed_json_files = list_files_from_date(DIRECTORY_STORE, DATE_STR_LOCAL)
print("Files with since", DATE_STR_LOCAL, ":", needed_json_files)


full_paths = [os.path.join(DIRECTORY_STORE, filename) for filename in needed_json_files]


def list_gcs_json_files(bucket_name: str, prefix: str = "analysis_outputs/") -> list:
    """Lists the possible json files in the bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    return [
        f"gs://{bucket_name}/{blob.name}"
        for blob in blobs
        if blob.name.endswith(".json")
    ]


dir_list_json = list_gcs_json_files(
    "gs://<bucket_location>/*.json"
)

print("dir_list_json", dir_list_json)

my_columns = [
    "classified",
    "followup",
    "chosen_sic_code",
    "chosen_sic_description",
    "reasoning",
    "payload_llm",
    "payload_type",
    "payload_job_title",
    "payload_job_description",
    "payload_industry_descr",
    "candidate_1_sic_code",
    "candidate_1_sic_descriptive",
    "candidate_1_likelihood",
    "candidate_2_sic_code",
    "candidate_2_sic_descriptive",
    "candidate_2_likelihood",
    "candidate_3_sic_code",
    "candidate_3_sic_descriptive",
    "candidate_3_likelihood",
    "candidate_4_sic_code",
    "candidate_4_sic_descriptive",
    "candidate_4_likelihood",
    "candidate_5_sic_code",
    "candidate_5_sic_descriptive",
    "candidate_5_likelihood",
]

all_data = pd.DataFrame(columns=my_columns)

for this_file in full_paths:
    print(this_file)
    df_llm = flatten_llm_json_to_dataframe(this_file, max_candidates=5)
    print("df_llm", df_llm.shape)
    all_data = pd.concat([all_data, df_llm], ignore_index=True)
    all_data = all_data.drop_duplicates(subset="unique_id")
    print("all_data", all_data.shape)


# all_data.to_csv('all_data.csv')
print("all_data shape ", all_data.shape)
