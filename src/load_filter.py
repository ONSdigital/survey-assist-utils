"""Load and Filter utilities.
The .toml configuration file should include:
- The path to the gold standard data file.
- The path to the output file.

The utilities allow for the following:
1. Loads the configuration from the .toml file..
2. Loads and filters the gold standard data.
3. Summarises the data using EDA.
"""

# Load required libraries

import pandas as pd
import toml


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
    """Reads a comma-separated CSV file and returns a DataFrame with specified column names.

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
