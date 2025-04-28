import logging
import json
import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple # Added type hinting

import numpy as np
import pandas as pd
import requests

# Assuming jwt_utils is structured correctly
from utils.api_token.jwt_utils import check_and_refresh_token

# --- Constants ---
# Define column names centrally
SIC_COLUMN_NAMES: List[str] = [
    "unique_id", "sic_section", "sic2007_employee", "sic2007_self_employed",
    "sic_ind1", "sic_ind2", "sic_ind3", "sic_ind_code_flag",
    "soc2020_job_title", "soc2020_job_description", "sic_ind_occ1",
    "sic_ind_occ2", "sic_ind_occ3", "sic_ind_occ_flag"
]

# Columns containing sensitive missing value codes
FILTER_NA_COLUMNS: List[str] = ["soc2020_job_title", "soc2020_job_description", "sic2007_employee"]
FILTER_NA_VALUES: List[str] = ['-9', '-8']

# API Configuration
API_URL: str = 'https://example-api-gateway-d90b4xu9.nw.gateway.dev/survey-assist/classify' # Use constant
API_SLEEP_DURATION: int = 10 # Seconds

# --- Logging Configuration ---
# Configure once at the module level or in main
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use module-level logger

# --- Data Loading ---

def load_data(file_path: str, delimiter: str = '\t', column_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Reads a delimited file into a pandas DataFrame with specified options.

    Args:
        file_path (str): The path to the input file.
        delimiter (str): The delimiter used in the file (e.g., '\t', ',').
        column_names (Optional[List[str]]): A list of column names. If None, assumes
                                            the file has a header row.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file_path does not exist.
        pd.errors.EmptyDataError: If the file is empty.
        Exception: For other potential pandas read errors.
    """
    logger.info(f"Loading data from: {file_path} with delimiter '{delimiter}'")
    try:
        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            names=column_names if column_names else None, # Use names only if provided
            header=0 if column_names is None else None,   # Assume header if names not given
            dtype=str,       # Load everything as string initially
            na_filter=False, # Treat empty fields as "" not NaN
            encoding='utf-8' # Specify encoding is good practice
        )
        logger.info(f"Successfully loaded {len(df)} rows.")
        # Basic validation (optional but recommended)
        if column_names and not all(col in df.columns for col in column_names):
             logger.warning(f"Loaded DataFrame missing expected columns. Found: {list(df.columns)}")
        return df
    except FileNotFoundError:
        logger.exception(f"Error: File not found at {file_path}") # Use logger.exception for errors
        raise
    except pd.errors.EmptyDataError:
        logger.exception(f"Error: File is empty at {file_path}")
        raise
    except Exception as e:
        logger.exception(f"Error loading data from {file_path}: {e}")
        raise

# --- Data Filtering ---

def filter_sic_data(df: pd.DataFrame,
                    na_columns: List[str] = FILTER_NA_COLUMNS,
                    na_values: List[str] = FILTER_NA_VALUES,
                    sic_column: str = 'sic_ind1',
                    min_sic_frequency: int = 11 # (was > 10)
                    ) -> pd.DataFrame:
    """
    Filters the DataFrame based on missing value codes and SIC code criteria.

    Args:
        df (pd.DataFrame): The input DataFrame (expects string types).
        na_columns (List[str]): Columns to check for special NA values.
        na_values (List[str]): Special string values indicating missing/invalid data.
        sic_column (str): The column containing the SIC code to filter on.
        min_sic_frequency (int): The minimum frequency a 5-digit SIC code must have to be kept.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    logger.info(f"Starting filtering on DataFrame with shape {df.shape}")
    original_rows = len(df)

    # 1. Drop rows with special NA values ('-9', '-8')
    # Ensure columns exist before filtering
    valid_na_columns = [col for col in na_columns if col in df.columns]
    if not valid_na_columns:
        logger.warning("None of the specified NA columns found in DataFrame. Skipping NA filter.")
    else:
        na_mask = df[valid_na_columns].apply(lambda row: row.isin(na_values).any(), axis=1)
        rows_with_na = na_mask.sum()
        df = df[~na_mask].copy() # Use .copy() to avoid SettingWithCopyWarning
        logger.info(f"Removed {rows_with_na} rows containing special NA values {na_values} in columns {valid_na_columns}.")

    # Check if sic_column exists
    if sic_column not in df.columns:
        logger.error(f"SIC column '{sic_column}' not found in DataFrame. Skipping SIC filtering.")
        return df

    # 2. Filter for valid 5-digit SIC codes
    five_digit_pattern = re.compile(r"^[0-9]{5}$")
    sic_mask = df[sic_column].str.match(five_digit_pattern)
    # Handle potential NaNs introduced IF na_filter=True was used or dtype wasn't str
    # Although with na_filter=False and dtype=str, this should primarily catch non-matches
    sic_mask = sic_mask.fillna(False)
    df_five_digits = df[sic_mask].copy()
    logger.info(f"Found {len(df_five_digits)} rows with valid 5-digit SIC codes in '{sic_column}'.")

    if df_five_digits.empty:
        logger.warning("No rows with valid 5-digit SIC codes found after initial filtering.")
        return df_five_digits # Return empty DataFrame

    # 3. Filter based on frequency of 5-digit codes
    value_counts = df_five_digits[sic_column].value_counts()
    valid_sic_codes = value_counts[value_counts >= min_sic_frequency].index.tolist() # Use >=
    logger.info(f"Found {len(valid_sic_codes)} SIC codes with frequency >= {min_sic_frequency}.")

    # Filter the original (NA-filtered) dataframe, not just the 5-digit one,
    # to keep rows that might have had other valid codes before this step
    # (assuming 'sic_ind1' was the *target* but maybe other sic columns are relevant?)
    # Re-evaluate if this is the desired logic. If only rows matching the frequent
    # 5-digit codes are needed, filter df_five_digits instead.
    # Let's assume we filter the df that passed the NA check:
    final_df = df[df[sic_column].isin(valid_sic_codes)].copy()
    removed_count = original_rows - len(final_df)
    logger.info(f"Filtering complete. Final shape: {final_df.shape}. Total rows removed: {removed_count}")

    return final_df

# --- API Interaction ---

def process_row_via_api(row: pd.Series, api_token: str) -> Dict[str, Any]:
    """
    Processes a single data row by making an API request to the classification service.

    Args:
        row (pd.Series): A row from the DataFrame, expecting specific column names
                         ('unique_id', 'soc2020_job_title', etc.).
        api_token (str): The JWT Bearer token for authorization.

    Returns:
        Dict[str, Any]: A dictionary containing the API response, potentially with
                        error information, merged with key input fields.
    """
    # Extract data safely using .get with defaults for potentially missing columns
    unique_id = row.get('unique_id', 'N/A')
    job_title = row.get('soc2020_job_title', '') # Default to empty string
    job_description = row.get('soc2020_job_description', '')
    industry_descr = row.get('sic2007_employee', '') # Assuming this is the correct mapping

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_token}'
    }
    payload = {
        'llm': 'gemini',
        'type': 'sic',
        'job_title': job_title,
        'job_description': job_description,
        'industry_descr': industry_descr
    }

    response_data = {}
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=30) # Add timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        # Handle potential JSON decoding errors even if status code is 2xx
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            logger.exception(f"Failed to decode JSON response for unique_id {unique_id}. Status: {response.status_code}, Response text: {response.text[:200]}")
            response_data = {'error': 'Invalid JSON response from API', 'status_code': response.status_code, 'response_text': response.text[:200]}

    except requests.exceptions.Timeout:
        logger.error(f"Request timed out for unique_id {unique_id}")
        response_data = {'error': 'Request timed out'}
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred for unique_id {unique_id}: {e}. Response: {e.response.text[:200]}")
        response_data = {'error': f'HTTP Error: {e.response.status_code}', 'response_text': e.response.text[:200]}
    except requests.exceptions.RequestException as e:
        logger.exception(f"Request failed for unique_id {unique_id}: {e}") # Use logger.exception here too
        response_data = {'error': f'Request failed: {e}'}

    # Combine input identifiers with the response/error data
    result = {
        'unique_id': unique_id,
        'job_title_sent': job_title, # Distinguish sent data from response fields
        'job_description_sent': job_description,
        'industry_descr_sent': industry_descr,
        **response_data # Merge API response or error dict
    }
    return result


def process_dataframe(df: pd.DataFrame, api_token: str, output_filepath: str, test_mode: bool = False, test_limit: int = 2):
    """
    Processes a DataFrame by sending each row to an API and saving responses.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        api_token (str): The JWT Bearer token for API authorization.
        output_filepath (str): Path to save the JSON line responses.
        test_mode (bool): If True, process only the first `test_limit` rows.
        test_limit (int): Number of rows to process in test mode.
    """
    data_to_process = df.head(test_limit) if test_mode else df
    num_rows = len(data_to_process)
    logger.info(f"Processing {num_rows} rows. Test mode: {test_mode}.")

    # Use 'w' mode for writing to ensure fresh results, consider timestamping filenames
    # or checking existence if append is truly needed.
    try:
        with open(output_filepath, 'w', encoding='utf-8') as file:
            for index, row in data_to_process.iterrows():
                logger.debug(f"Processing row index {index}, unique_id {row.get('unique_id', 'N/A')}")
                response_json = process_row_via_api(row, api_token)
                file.write(json.dumps(response_json) + '\n')

                # Avoid sleeping on the very last iteration
                if index < num_rows - 1: # Check if it's not the last item based on count
                     time.sleep(API_SLEEP_DURATION)

        logger.info(f"Finished processing. Responses saved to {output_filepath}")

    except IOError as e:
        logger.exception(f"Error writing to output file {output_filepath}: {e}")
        raise


# --- Token Management ---

def get_api_token() -> Optional[str]:
    """
    Retrieves required configuration from environment variables and gets an API token.

    Returns:
        Optional[str]: The API token string, or None if configuration is missing or token generation fails.
    """
    api_gateway = os.getenv("API_GATEWAY")
    sa_email = os.getenv("SA_EMAIL")
    jwt_secret_path = os.getenv("JWT_SECRET")

    if not all([api_gateway, sa_email, jwt_secret_path]):
        logger.error("Missing required environment variables: API_GATEWAY, SA_EMAIL, JWT_SECRET")
        return None

    try:
        # Assuming check_and_refresh_token handles initial check correctly
        # We don't need to track token_start_time here if the util handles expiry.
        _, current_token = check_and_refresh_token(
            token_start_time=0, # Or appropriate initial value if needed by util
            current_token="",   # Start fresh
            jwt_secret_path=jwt_secret_path,
            api_gateway=api_gateway, # Ensure kwarg name matches util
            sa_email=sa_email
        )
        if not current_token:
             logger.error("Failed to retrieve a valid API token.")
             return None
        logger.info("Successfully obtained API token.")
        return current_token
    except Exception as e:
        logger.exception(f"Error obtaining API token: {e}")
        return None

# --- Main Execution ---

def main():
    """Main execution function."""
    logger.info("Starting script execution.")

    # Configuration (Consider using argparse or a config file for more complex scenarios)
    input_file = "data/all_examples_comma.csv" # Assumed comma delimited based on name
    output_file = "data/output_responses.jsonl" # Use .jsonl for JSON lines
    use_test_mode = True
    test_sample_size = 3

    # 1. Get API Token
    api_token = get_api_token()
    if not api_token:
        logger.error("Exiting script due to token retrieval failure.")
        return # Exit gracefully

    # 2. Load Data
    # Determine delimiter based on filename or config (using comma here)
    try:
        # Assuming comma delimiter based on filename in main block
        # Use SIC_COLUMN_NAMES only if the file structure matches, otherwise load with header
        # Let's assume it doesn't have the header and structure matches SIC_COLUMN_NAMES
        # If it *does* have a header, set column_names=None
        raw_df = load_data(input_file, delimiter=',', column_names=SIC_COLUMN_NAMES)
    except (FileNotFoundError, pd.errors.EmptyDataError, Exception):
        logger.error(f"Exiting script due to data loading failure from {input_file}.")
        return

    # 3. Filter Data
    filtered_df = filter_sic_data(raw_df)

    if filtered_df.empty:
        logger.warning("DataFrame is empty after filtering. Nothing to process.")
        return

    # Log a sample for verification
    logger.info(f"Sample of filtered data (first {min(test_sample_size, len(filtered_df))} rows):\n{filtered_df.head(test_sample_size)}")

    # 4. Process Data via API
    try:
        process_dataframe(
            df=filtered_df,
            api_token=api_token,
            output_filepath=output_file,
            test_mode=use_test_mode,
            test_limit=test_sample_size
        )
    except Exception as e:
         logger.exception(f"An error occurred during DataFrame processing: {e}")

    logger.info("Script execution finished.")


if __name__ == "__main__":
    main()