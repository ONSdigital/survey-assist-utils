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
import time

import pandas as pd
import requests
import toml

# load the utils:
from survey_assist_utils.api_token.jwt_utils import check_and_refresh_token

WAIT_TIMER = 0.5  # seconds to wait between requests to avoid rate limiting

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
    base_url = os.getenv("API_GATEWAY", "http://127.0.0.1:5000") + "/survey-assist/classify"
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
    secret_code,
    process_batch_data,
    app_config,
):
    """Process the test set CSV file, make API requests, and save the responses to an output file.

    Parameters:
    secret_code (str): The secret code for API authorisation.
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

    # Process each row in the DataFrame
    with open(output_filepath, "w", encoding="utf-8") as file:
        # Write the opening array bracket
        file.write("[\n")
        for _index, row in process_batch_data.iterrows():
            logging.info("Processing row %s", _index)
            response_json = process_row(row, secret_code, app_config=app_config)
            file.write(json.dumps(response_json) + ",\n")
            time.sleep(WAIT_TIMER)  # Wait between requests to avoid rate limiting
        # Remove the last comma and close the array
        file.seek(file.tell() - 2, os.SEEK_SET)
        file.write("\n]")
        logging.info("Finished processing rows.")


if __name__ == "__main__":

    # Load configuration from .toml file
    config = load_config("config.toml")

    # Where the input data csv is
    batch_filepath = config["paths"]["batch_filepath"]

    # Get a secret token - initially set to empty, then refresh it.
    TOKEN_START_TIME = 0
    CURRENT_TOKEN = ""

    api_gateway = os.getenv("API_GATEWAY", "")
    sa_email = os.getenv("SA_EMAIL", "")
    jwt_secret_path = os.getenv("JWT_SECRET", "")

    # Load the data
    batch_data = pd.read_csv(batch_filepath, delimiter=",", dtype=str)

    # Get token:
    TOKEN_START_TIME, CURRENT_TOKEN = check_and_refresh_token(
        TOKEN_START_TIME, CURRENT_TOKEN, jwt_secret_path, api_gateway, sa_email
    )

    # process file:
    process_test_set(
        secret_code=CURRENT_TOKEN, process_batch_data=batch_data, app_config=config
    )
