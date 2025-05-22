"""This script processes SIC code batch data through Survey Assist API.
It is based on configurations specified in a .toml file.
Prior to invocation, ensure to run the following CLI commands:
> gcloud config set project "valid-project-name"
> gcloud auth application-default login
By defalut the output file overwritten.

Run from the root of the project as follows:

poetry run python scripts/batch.py

It also requires the following environment variables to be exported:
- API_GATEWAY: The base API gateway URL. This is used to get and refresh
    the token and is different to the API destination in the config.toml.
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
    poetry run python scripts/batch.py


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
import logging
import os
import tempfile
import time

import pandas as pd
import requests
import toml

# load the utils:
from survey_assist_utils.api_token.jwt_utils import (
    check_and_refresh_token,
    resolve_jwt_secret_path,
)
from survey_assist_utils.cloud_store.gcs_utils import download_from_gcs, upload_to_gcs

WAIT_TIMER = 0.5  # seconds to wait between requests to avoid rate limiting
UPLOAD_ROWS = 5  # upload every 5 rows


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


def process_row(row, secret_code, app_config):
    """Process a single row of the DataFrame, make an API request, and return the response.
    If an error occours (404, 503, etc), the UID, payload and error are returned instead.

    Parameters:
    row (pd.Series): A row from the DataFrame.
    secret_code (str): The secret code for API authorization.
    app_config : the loaded configuration toml.

    Returns:
    dict: The response JSON with additional information.
    """
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
        "Authorization": f"Bearer {secret_code}",
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
        logging.error("Request failed for unique_id %s: %s", unique_id, e)
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
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Unpack variables from config

    # for test, select a small sample:
    test_limit = app_config["parameters"]["test_num"]
    test_mode_option = app_config["parameters"]["test_mode"]

    # Where to put the output
    output_filepath = app_config["paths"]["output_filepath"]

    # Determine the subset of data to process
    if test_mode_option:
        process_batch_data = process_batch_data.head(test_limit)
        logging.info("Test mode enabled. Processing first %s rows.", test_limit)

    is_gcs_output = output_filepath.startswith("gs://")
    total_rows = len(process_batch_data)

    if is_gcs_output:
        # Use a persistent temp file for incremental upload
        tmp_dir = tempfile.gettempdir()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        local_output_path = os.path.join(tmp_dir, f"partial_output_{timestamp}.json")
        logging.info(
            "Using temporary file for intermediate results: %s", local_output_path
        )

        # Set the full path in gcs, inlude date and time to avoid overwriting
        output_filepath = output_filepath.rstrip("/")  # Remove trailing slash
        output_filepath += "/analysis_outputs/"
        output_filepath += time.strftime("%Y%m%d_%H%M%S") + "_output.json"
        logging.info("Output will be uploaded to GCS bucket: %s", output_filepath)
    else:
        local_output_path = output_filepath

    # Process each row in the DataFrame
    with open(local_output_path, "w+", encoding="utf-8") as target_file:
        # Write the opening array bracket
        target_file.write("[\n")
        for i, (_index, row) in enumerate(process_batch_data.iterrows()):
            logging.info("Processing row %s", _index)
            # Check and refresh the token if necessary
            token_information["token_start_time"], token_information["current_token"] = check_and_refresh_token(
                token_information["token_start_time"], 
                token_information["current_token"], 
                token_information["jwt_secret_path"], 
                token_information["api_gateway"], 
                token_information["sa_email"]
    )

            response_json = process_row(row, token_information["current_token"], app_config=app_config)
            target_file.write(json.dumps(response_json) + ",\n")
            target_file.flush()

            if is_gcs_output and ((i + 1) % UPLOAD_ROWS == 0 or (i + 1) == total_rows):
                logging.info(
                    "Uploading intermediate results to GCS bucket: %s, i: %s tot:%s",
                    output_filepath,
                    i,
                    total_rows,
                )

                upload_to_gcs(local_output_path, output_filepath)

            percent_complete = round(((i + 1) / total_rows) * 100, 2)
            logging.info(
                "Processed row %d of %d (%.2f%%)", i + 1, total_rows, percent_complete
            )
            time.sleep(WAIT_TIMER)  # Wait between requests to avoid rate limiting

        # Remove the last comma and close the array
        target_file.seek(target_file.tell() - 2, os.SEEK_SET)
        target_file.write("\n]")
        logging.info("Finished processing rows.")

    # Final upload: now the file is valid JSON
    if is_gcs_output:
        upload_to_gcs(local_output_path, output_filepath)
        logging.info("Final upload completed.")

        # Optional: clean up
        os.remove(local_output_path)
        logging.info("Deleted local temp file.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Starting batch processing script.")
    # Load configuration from .toml file
    config = load_config("config.toml")

    # Where the input data csv is
    batch_filepath = config["paths"]["batch_filepath"]

    # Create a dictionary to hold the TOKEN variables
    raw_jwt_env = os.getenv("JWT_SECRET", "")

    # Check if the JWT_SECRET is a file path or a JSON string
    # It will be a JSON string when run in GCP
    jwt_secret_path = resolve_jwt_secret_path(raw_jwt_env)

    token_information = {
        "token_start_time":0,
        "current_token": "",
        "api_gateway": os.getenv("API_GATEWAY", ""),
        "sa_email": os.getenv("SA_EMAIL", ""),
        "jwt_secret_path": resolve_jwt_secret_path(raw_jwt_env)
    }

    logging.info("API Gateway: %s", token_information["api_gateway"][:10])
    logging.info("Service Account Email: %s", token_information["sa_email"][:10])


    if batch_filepath.startswith("gs://"):
        logging.info("Downloading batch file from GCS: %s", batch_filepath)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            download_from_gcs(batch_filepath, tmp_file.name)
            local_csv_path = tmp_file.name
            logging.info("Downloaded GCS file %s to %s", batch_filepath, local_csv_path)
    else:
        local_csv_path = batch_filepath

    # Load the data
    batch_data = pd.read_csv(batch_filepath, delimiter=",", dtype=str)

    # Get token:
    token_information["token_start_time"], token_information["current_token"] = check_and_refresh_token(
        token_information["token_start_time"], 
        token_information["current_token"], 
        token_information["jwt_secret_path"], 
        token_information["api_gateway"], 
        token_information["sa_email"]
    )

    # process file:
    process_test_set(
        token_information, process_batch_data=batch_data, app_config=config
    )
