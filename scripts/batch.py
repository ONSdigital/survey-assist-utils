import pandas as pd
import re
import requests
import json
import time
import logging
import pandas as pd
import numpy as np
import re 
from pprint import pprint
import matplotlib.pyplot as plt
import os

# load the utils:
from utils.api_token.jwt_utils import (
    check_and_refresh_token # this does everything we need
)


def read_sic_data(file_path):
    """
    Reads a tab-separated CSV file and returns a DataFrame with specified column names.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    column_names = [
        "unique_id", "sic_section", "sic2007_employee", "sic2007_self_employed", 
        "sic_ind1", "sic_ind2", "sic_ind3", "sic_ind_code_flag", 
        "soc2020_job_title", "soc2020_job_description", "sic_ind_occ1", 
        "sic_ind_occ2", "sic_ind_occ3", "sic_ind_occ_flag"
    ]
    
    # Read the CSV file with the specified delimiter and column names
    sic_data = pd.read_csv(file_path, delimiter='\t', names=column_names, dtype=str)
    
    return sic_data


def subset_data(file_path):
    """
    Subsets the data based on specified criteria.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the subsetted data.
    """
    column_names = [
        "unique_id", "sic_section", "sic2007_employee", "sic2007_self_employed", 
        "sic_ind1", "sic_ind2", "sic_ind3", "sic_ind_code_flag", 
        "soc2020_job_title", "soc2020_job_description", "sic_ind_occ1", 
        "sic_ind_occ2", "sic_ind_occ3", "sic_ind_occ_flag"
    ]
    
    # Read the CSV file with the specified delimiter and column names
    sic_data = pd.read_csv(file_path, delimiter='\t', names=column_names, dtype=str)
    
    # Define the columns to check for specific values
    prompt_columns = ["soc2020_job_title", "soc2020_job_description", "sic2007_employee"]
    
    # Count rows with value '-9' or '-8' in specified columns
    count_rows = sic_data[prompt_columns].apply(lambda row: row.isin(['-9', '-8']).any(), axis=1).sum()
    print(f"Number of rows with '-9' or '-8' in specified columns: {count_rows}")
    
    # Drop rows with value '-9' or '-8' in specified columns
    sic_data = sic_data[~sic_data[prompt_columns].apply(lambda row: row.isin(['-9', '-8']).any(), axis=1)]
    
    # Filter rows containing only 5-digit codes using the pattern
    five_digit_pattern = re.compile(r"^[0-9]{5}$")
    df_five_digits = sic_data[sic_data['sic_ind1'].str.match(five_digit_pattern)].copy()
    matched_five_list = sorted(df_five_digits['sic_ind1'].unique())
    print(f"Shape of DataFrame with 5-digit codes: {df_five_digits.shape}")
    print(f"Unique 5-digit codes: {matched_five_list}")
    
    # Get frequency of 5-digit codes
    freq_five_digits = df_five_digits['sic_ind1'].value_counts()
    print(f"Frequency of 5-digit codes:\n{freq_five_digits}")
    
    # Filter data based on frequency of 5-digit codes
    value_counts = df_five_digits['sic_ind1'].value_counts()
    filtered_indexes = value_counts[value_counts > 10].index.tolist()
    sic_data = sic_data[sic_data['sic_ind1'].isin(filtered_indexes)]
    print(f"Shape of filtered DataFrame: {sic_data.shape}")
    
    return sic_data


def process_test_set(secret_code, csv_filepath, output_filepath, test_mode=False, test_limit=2):
    """
    Process the test set CSV file, make API requests, and save the responses to an output file.

    Parameters:
    secret_code (str): The secret code for API authorization.
    csv_filepath (str): The file path to the test set CSV file.
    output_filepath (str): The file path for the output responses.
    test_mode (bool): If True, process only a limited number of rows for testing.
    test_limit (int): The number of rows to process in test mode.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Read the CSV file into a DataFrame
    gold_df = pd.read_csv(csv_filepath, delimiter=',', dtype=str)

    # Determine the subset of data to process
    if test_mode:
        gold_df = gold_df.head(test_limit)
        logging.info(f"Test mode enabled. Processing first {test_limit} rows.")

    # Process each row in the DataFrame
    with open(output_filepath, 'a') as file:
        for index, row in gold_df.iterrows():
            response_json = process_row(row, secret_code)
            file.write(json.dumps(response_json) + '\n')
            time.sleep(10)  # Wait between requests to avoid rate limiting


def process_row(row, secret_code):
    """
    Process a single row of the DataFrame, make an API request, and return the response.

    Parameters:
    row (pd.Series): A row from the DataFrame.
    secret_code (str): The secret code for API authorization.

    Returns:
    dict: The response JSON with additional information.
    """
    unique_id = row['unique_id']
    job_title = row['soc2020_job_title']
    job_description = row['soc2020_job_description']
    industry_descr = row['sic2007_employee']

    url = 'https://example-api-gateway-d90b4xu9.nw.gateway.dev/survey-assist/classify'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {secret_code}'
    }
    payload = {
        'llm': 'gemini',
        'type': 'sic',
        'job_title': job_title,
        'job_description': job_description,
        'industry_descr': industry_descr
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        response_json = response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for unique_id {unique_id}: {e}")
        response_json = {
            'unique_id': unique_id,
            'job_title': job_title,
            'job_description': job_description,
            'industry_descr': industry_descr,
            'error': str(e)
        }

    response_json.update({
        'unique_id': unique_id,
        'job_title': job_title,
        'job_description': job_description,
        'industry_descr': industry_descr
    })

    return response_json


if __name__ == "__main__":


    # Get a secret token:
    token_start_time = 0
    current_token = ""
    api_gateway = os.getenv("API_GATEWAY")
    sa_email = os.getenv("SA_EMAIL")
    jwt_secret_path = os.getenv("JWT_SECRET")
    token_start_time, current_token = check_and_refresh_token(
                                        token_start_time,
                                        current_token,
                                        jwt_secret_path,
                                        api_gateway,
                                        sa_email) 


    # load the data and filter it
    file_path = "data/all_examples_comma.csv"
    output_filepath = "data/output.csv"
    subsetted_data = subset_data(file_path)

    # for test, select a small sample:
    test_num = 3
    print(subsetted_data.head(test_num))

    # process file:
    process_test_set(secret_code = current_token, csv_filepath = file_path, output_filepath = output_filepath, test_mode = True, test_limit = test_num)

