"""This script processes SIC code batch data through Survey Assist API.
It is based on configurations specified in a .toml file.
Prior to invocation, ensure to run the following CLI commands:
> gcloud config set project "valid-project-name"
> gcloud auth application-default login
By defalut the output file overwritten.

It also requires the following environment variables to be exported:
- API_GATEWAY: The base API gateway URL. This is used to get and refresh
    the token.
- SA_EMAIL: The service account email.
- JWT_SECRET: The path to the JWT secret.

The .toml configuration file should include:
- The path to the batch data file.
- The path to the output file.
- The number of test items to process (if running in test mode).

The script performs the following steps:
1. Loads the configuration from the .toml file.
2. Retrieves the necessary environment variables.
3. Obtains a secret token using the `check_and_refresh_token` function.
4. Loads the batch data.
5. Processes the data either in test mode (processing a subset) or for all items.
6. Writes the results to the file specified in config.toml

Usage:
Run from the root of the project as follows:
    poetry run python scripts/process_tlfs_evaluation_data.py

This now supports Docker implementation with gc buckets as storage.
Ensure the config file points to the appropriate gs:// locations if used in this way.

Functions:
    load_config(config_path: str) -> dict
        Loads the configuration from the specified .toml file.

    process_row(row, secret_code, app_config) -> json
        sends a request to the API using a rof from the batch data and headers
        found in the config.

    process_test_set(secret_code, process_batch_data, app_config) -> None
        Processes the data and writes the output to the specified file path.

"""

import json
import os
import tempfile
import time
from typing import TypedDict, cast

import pandas as pd
import requests
import toml

# load the utils:
from survey_assist_utils.api_token.jwt_utils import (
    check_and_refresh_token,
    resolve_jwt_secret_path,
)
from survey_assist_utils.cloud_store.gcs_utils import download_from_gcs, upload_to_gcs
from survey_assist_utils.logging import get_logger

# Create a logger instance
logger = get_logger(__name__)

WAIT_TIMER = 0.5  # seconds to wait between requests to avoid rate limiting
UPLOAD_ROWS = 5  # upload every 5 rows


# Add a dict for the token
class TokenInformation(TypedDict):
    """Represents authentication and configuration details required for token-based API access.

    Attributes:
        token_start_time (int): The Unix timestamp indicating when the token was issued.
        current_token (str): The current access token string.
        api_gateway (str): The base URL of the API gateway.
        sa_email (str): The service account email associated with the token.
        jwt_secret_path (str): The file path or resolved location of the JWT secret.
    """

    token_start_time: int
    current_token: str
    api_gateway: str
    sa_email: str
    jwt_secret_path: str


# Load the config:
def load_config(config_path):
    """Loads configuration settings from a .toml file.

    Args:
        config_path (str): The path to the .toml configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.

    Example:
        config = load_config("config.toml")
        print(config["settings"]["api_gateway"])
    """
    with open(config_path, encoding="utf-8") as file:
        configuration = toml.load(file)
    return configuration


def process_row(row, token_information, app_config):
    """Process a single row of the DataFrame, make an API request, and return the response.
    If an error occours (404, 503, etc), the UID, payload and error are returned instead.

    Parameters:
    row (pd.Series): A row from the DataFrame.
    secret_code (str): The secret code for API authorization.
    app_config : the loaded configuration toml.

    Returns:
    dict: The response JSON with additional information.
    """
    # Check and refresh the token if necessary
    (
        token_information["token_start_time"],
        token_information["current_token"],
    ) = check_and_refresh_token(
        token_information["token_start_time"],
        token_information["current_token"],
        token_information["api_gateway"],
        token_information["sa_email"],
    )

    base_url = (
        os.getenv("API_GATEWAY", "http://127.0.0.1:5000") + "/survey-assist/classify"
    )
    unique_id = row[app_config["column_names"]["payload_unique_id"]]
    job_title = row[app_config["column_names"]["payload_job_title"]]
    job_description = row[app_config["column_names"]["payload_job_description"]]
    industry_description = row[
        app_config["column_names"]["payload_industry_description"]
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token_information["current_token"]}",
    }
    payload = {
        "llm": "gemini",
        "type": "sic",
        "job_title": job_title,
        "job_description": job_description,
        "industry_descr": industry_description,
    }

    try:
        response = requests.post(
            base_url, headers=headers, data=json.dumps(payload), timeout=10
        )
        response.raise_for_status()
        response_json = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for unique_id {unique_id}: {e}")
        response_json = {
            "unique_id": unique_id,
            "request_payload": payload,
            "error": str(e),
        }

    # Add metadata and payload to the response
    response_json.update({"unique_id": unique_id, "request_payload": payload})

    return response_json


def process_test_set(
    token_information,
    process_batch_data,
    app_config,
):
    """Process the test set CSV file, make API requests, and save the responses to an output file.

    Parameters:
    token_information (dict): The information containing the secret code and
            other related details for API authorisation.
    process_batch_data (dataframe): The data to process.
    app_config : the toml config.
    """
    # Unpack variables from config
    # Where to put the output
    output_filepath = app_config["paths"]["output_filepath"]

    # Determine the subset of data to process - Check the truth
    # of test_mode and cut to only test_num top rows
    if app_config["parameters"]["test_mode"]:
        process_batch_data = process_batch_data.head(
            app_config["parameters"]["test_num"]
        )
        logger.info(
            f"Test mode enabled. Processing first {app_config['parameters']['test_num']} rows."
        )

    is_gcs_output = output_filepath.startswith("gs://")
    total_rows = len(process_batch_data)

    if is_gcs_output:
        # Use a persistent temp file for incremental upload
        tmp_dir = tempfile.gettempdir()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        local_output_path = os.path.join(tmp_dir, f"partial_output_{timestamp}.json")
        logger.info(
            "Using temporary file for intermediate results: {local_output_path}"
        )

        # Set the full path in gcs, inlude date and time to avoid overwriting
        output_filepath = output_filepath.rstrip("/")  # Remove trailing slash
        output_filepath += "/analysis_outputs/"
        output_filepath += time.strftime("%Y%m%d_%H%M%S") + "_output.json"
        logger.info(f"Output will be uploaded to GCS bucket: {output_filepath}")
    else:
        local_output_path = output_filepath

    # Process each row in the DataFrame
    with open(local_output_path, "w+", encoding="utf-8") as target_file:
        # Write the opening array bracket
        target_file.write("[\n")
        for i, (_index, row) in enumerate(process_batch_data.iterrows()):
            logger.info("Processing row {_index}")

            response_json = process_row(row, token_information, app_config=app_config)
            target_file.write(json.dumps(response_json) + ",\n")
            target_file.flush()

            if is_gcs_output and ((i + 1) % UPLOAD_ROWS == 0 or (i + 1) == total_rows):
                logger.info(
                    f"""Uploading intermediate results to GCS bucket:
                     {output_filepath}, i: {i} tot:{total_rows}"""
                )
                upload_to_gcs(local_output_path, output_filepath)

            percent_complete = round(((i + 1) / total_rows) * 100, 2)
            logger.info(
                f"Processed row {i + 1} of {total_rows} {percent_complete:.2f}%"
            )
            time.sleep(WAIT_TIMER)  # Wait between requests to avoid rate limiting

        # Remove the last comma and close the array
        target_file.seek(target_file.tell() - 2, os.SEEK_SET)
        target_file.write("\n]")
        logger.info("Finished processing rows.")

    # Final upload: now the file is valid JSON
    if is_gcs_output:
        upload_to_gcs(local_output_path, output_filepath)
        logger.info("Final upload completed.")

        os.remove(local_output_path)
        logger.info("Deleted local temp file.")


if __name__ == "__main__":

    logger.info("Starting batch processing script.")
    # Load configuration from .toml file
    config = load_config("config.toml")

    # Where the input data csv is
    batch_filepath = config["paths"]["batch_filepath"]

    # Create a dictionary to hold the TOKEN variables
    raw_jwt_env = os.getenv("JWT_SECRET", "")

    # Check if the JWT_SECRET is a file path or a JSON string
    # It will be a JSON string when run in GCP
    jwt_secret_path = cast(str, resolve_jwt_secret_path(raw_jwt_env))

    token_information_init: TokenInformation = {
        "token_start_time": 0,
        "current_token": "",
        "api_gateway": os.getenv("API_GATEWAY", ""),
        "sa_email": os.getenv("SA_EMAIL", ""),
        "jwt_secret_path": jwt_secret_path,
    }

    logger.info(f"API Gateway: {token_information_init['api_gateway']}")
    logger.info(f"Service Account Email: {token_information_init["sa_email"]}")

    if batch_filepath.startswith("gs://"):
        logger.info(f"Downloading batch file from GCS: {batch_filepath}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            download_from_gcs(batch_filepath, tmp_file.name)
            local_csv_path = tmp_file.name
            logger.info("Downloaded GCS file {batch_filepath} to {local_csv_path}")
    else:
        local_csv_path = batch_filepath

    # Load the data
    batch_data = pd.read_csv(batch_filepath, delimiter=",", dtype=str)

    # Option to skip rows:
    skip_n_rows = config["parameters"]["rows_to_skip"]
    if skip_n_rows > 0:
        logger.info(f"Skipping {skip_n_rows} rows")
        batch_data = batch_data.iloc[skip_n_rows:]

    logger.info(f"Processing {len(batch_data)} rows of data ")

    # Get token initially:
    (
        token_information_init["token_start_time"],
        token_information_init["current_token"],
    ) = check_and_refresh_token(
        token_information_init["token_start_time"],
        token_information_init["current_token"],
        token_information_init["api_gateway"],
        token_information_init["sa_email"],
    )

    # process file:
    process_test_set(
        token_information_init, process_batch_data=batch_data, app_config=config
    )
