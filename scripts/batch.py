"""This script processes SIC code batch data through Survey Assist API.
It is based on configurations specified in a .toml file.
Prior to invocation, ensure to run the following CLI commands:
> gcloud config set project "valid-priject-name"
> gcloud auth application-default login
By defalut the output file is appended to, not overwritten.
Delete it before running, or change the name if this is not want you want.

Run from the root of the project as follows:

path/to/python /home/user/survey-assist-utils/scripts/batch.py

It also requires the following environment variables to be exported:
- API_GATEWAY: The API gateway URL.
- SA_EMAIL: The service account email.
- JWT_SECRET: The path to the JWT secret.

The .toml configuration file should include:
- The path to the gold standard data file.
- The path to the output file.
- The number of test items to process (if running in test mode).

The script performs the following steps:
1. Loads the configuration from the .toml file.
2. Retrieves the necessary environment variables.
3. Obtains a secret token using the `check_and_refresh_token` function.
4. Loads the gold standard data.
5. Processes the data either in test mode (processing a subset) or for all items.
6. Writes (appends) the results to the file specified in config.toml

Usage:
    poetry run python scripts/batch.py

Example .toml configuration file (config.toml):
    [settings]
    api_gateway = "your_api_gateway"
    sa_email = "your_sa_email"
    jwt_secret_path = "your_jwt_secret_path"
    file_path = "data/all_examples_comma.csv"
    output_filepath = "data/output.csv"
    test_num = 3

Functions:
    load_config(config_path: str) -> dict
        Loads the configuration from the specified .toml file.

    process_test_set(secret_code: str, csv_filepath: str, output_filepath: str,
            test_mode: bool, test_limit: int) -> None
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
from utils.api_token.jwt_utils import (
    check_and_refresh_token,  # this does everything we need
)

# Define a constant for the threshold value
THRESHOLD_VALUE = 10


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


def read_sic_data(file_path):
    """Reads a tab-separated CSV file and returns a DataFrame with specified column names.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    column_names = [
        "unique_id",
        "sic_section",
        "sic2007_employee",
        "sic2007_self_employed",
        "sic_ind1",
        "sic_ind2",
        "sic_ind3",
        "sic_ind_code_flag",
        "soc2020_job_title",
        "soc2020_job_description",
        "sic_ind_occ1",
        "sic_ind_occ2",
        "sic_ind_occ3",
        "sic_ind_occ_flag",
    ]

    # Read the CSV file with the specified delimiter and column names
    sic_data = pd.read_csv(file_path, delimiter=",", names=column_names, dtype=str)

    return sic_data


def process_row(row, secret_code):
    """Process a single row of the DataFrame, make an API request, and return the response.

    Parameters:
    row (pd.Series): A row from the DataFrame.
    secret_code (str): The secret code for API authorization.

    Returns:
    dict: The response JSON with additional information.
    """
    unique_id = row["unique_id"]
    job_title = row["soc2020_job_title"]
    job_description = row["soc2020_job_description"]
    industry_descr = row["sic2007_employee"]

    url = "https://example-api-gateway-d90b4xu9.nw.gateway.dev/survey-assist/classify"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {secret_code}",
    }
    payload = {
        "llm": "gemini",
        "type": "sic",
        "job_title": job_title,
        "job_description": job_description,
        "industry_descr": industry_descr,
    }

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=10
        )
        response.raise_for_status()
        response_json = response.json()
    except requests.exceptions.RequestException as e:
        logging.error("Request failed for unique_id %s: %s", unique_id, e)
        response_json = {
            "unique_id": unique_id,
            "job_title": job_title,
            "job_description": job_description,
            "industry_descr": industry_descr,
            "error": str(e),
        }

    # Add metadata and payload to the response
    response_json.update({"unique_id": unique_id, "request_payload": payload})

    return response_json


def process_test_set(
    secret_code, csv_filepath, output_filepath, test_mode=False, test_limit=2
):
    """Process the test set CSV file, make API requests, and save the responses to an output file.

    Parameters:
    secret_code (str): The secret code for API authorization.
    csv_filepath (str): The file path to the test set CSV file.
    output_filepath (str): The file path for the output responses.
    test_mode (bool): If True, process only a limited number of rows for testing.
    test_limit (int): The number of rows to process in test mode.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Read the CSV file into a DataFrame
    gold_df = pd.read_csv(csv_filepath, delimiter=",", dtype=str)

    # Determine the subset of data to process
    if test_mode:
        gold_df = gold_df.head(test_limit)
        logging.info("Test mode enabled. Processing first %s rows.", test_limit)

    # Process each row in the DataFrame
    with open(output_filepath, "a", encoding="utf-8") as file:
        for _index, row in gold_df.iterrows():
            response_json = process_row(row, secret_code)
            file.write(json.dumps(response_json) + "\n")
            time.sleep(10)  # Wait between requests to avoid rate limiting


if __name__ == "__main__":

    # Load configuration from .toml file
    # We are loading  the following:
    # Input path for the expert-coded gold standard data (relative to project root)
    # gold_standard_csv = "data/evaluation_data/coding_df_with_validated.csv"

    # Output directory for analysis results (relative to project root)
    # output_dir = "data/analysis_outputs"
    config = load_config("config.toml")

    # Where the input data csv is
    gold_standard_csv = config["paths"]["gold_standard_csv"]

    # Where to put the output
    output_file_path = config["paths"]["output_filepath"]

    # Get a secret token:
    TOKEN_START_TIME = 0
    CURRENT_TOKEN = ""

    api_gateway = os.getenv("API_GATEWAY", "")
    sa_email = os.getenv("SA_EMAIL", "")
    jwt_secret_path = os.getenv("JWT_SECRET", "")

    # Load the data
    batch_data = pd.read_csv(gold_standard_csv, delimiter=",", dtype=str)

    # for test, select a small sample:
    test_num = config["parameters"]["test_num"]
    test_mode_option = config["parameters"]["test_mode"]

    print(batch_data.head(test_num))

    # Get token:
    TOKEN_START_TIME, CURRENT_TOKEN = check_and_refresh_token(
        TOKEN_START_TIME, CURRENT_TOKEN, jwt_secret_path, api_gateway, sa_email
    )

    # process file:
    process_test_set(
        secret_code=CURRENT_TOKEN,
        csv_filepath=gold_standard_csv,
        output_filepath=output_file_path,
        test_mode=test_mode_option,
        test_limit=test_num,
    )
