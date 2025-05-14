"""Load and Filter utilities.

The .toml configuration file should include:
- The path to the test data file.
- Column names mapping if they differ from defaults.

The utilities allow for the following:
1. Loads the configuration from the .toml file.
2. Loads the test data.
3. Adds data quality flag columns to the DataFrame.
4. Utility functions for analysing and visualising data quality flags
and SIC code distributions.
"""

import logging
import os
import re
from typing import Any, Optional

import pandas as pd

from utils.config_loader import load_config  # Using the imported load_config

# --- Default Configuration Values (if not found in config) ---
DEFAULT_OUTPUT_DIR = "analysis_outputs"
DEFAULT_SIC_OCC1_COL = "sic_ind_occ1"
DEFAULT_SIC_OCC2_COL = "sic_ind_occ2"
DEFAULT_SIC_OCC3_COL = "sic_ind_occ3"

SPECIAL_SIC_NOT_CODEABLE = "-9"
SPECIAL_SIC_MULTIPLE_POSSIBLE = "4+"

# --- Constants for Data Quality ---
EXPECTED_SIC_LENGTH = 5
X_COUNT_FOR_MATCH_3 = 2
X_COUNT_FOR_MATCH_2 = 3

# Load configuration from .toml file
main_config = load_config("config.toml")
log_config = main_config.get("logging", {})

# Extract values with defaults
log_level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
log_format = log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s")
log_file = log_config.get("file")
print("log_file", log_file)

# Set up logging
logger = logging.getLogger()
logger.setLevel(log_level)
formatter = logging.Formatter(log_format)


def _calculate_num_answers(
    df: pd.DataFrame, col_occ1: str, col_occ2: str, col_occ3: str
) -> pd.Series:
    """Calculates the number of provided answers in SIC occurrence columns.

    An answer is considered provided if it's not NaN, not an empty string,
    and not "NA" (case-insensitive).

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_occ1 (str): Name of the primary SIC code column.
        col_occ2 (str): Name of the secondary SIC code column.
        col_occ3 (str): Name of the tertiary SIC code column.

    Returns:
        pd.Series: A pandas Series of integers representing the count of answers (0 to 3).
    """
    num_answers = pd.Series(0, index=df.index, dtype="int")
    for col_name in [col_occ1, col_occ2, col_occ3]:
        if col_name in df.columns:
            # An entry is considered valid if it's not NaN, not empty string, and not 'NA'
            is_valid_entry = (
                ~df[col_name].isna()
                & (df[col_name].astype(str).str.strip() != "")
                & (df[col_name].astype(str).str.upper() != "NA")
            )
            num_answers += is_valid_entry.astype(int)
        else:
            logger.warning(
                "Column '%s' not found for num_answers calculation. It will be ignored.",
                col_name,
            )

    return num_answers


# --- Helper Function for SIC Code Matching ---
def _create_sic_match_flags(sic_series: pd.Series) -> dict[str, pd.Series]:
    """Calculates various SIC code format match flags for a given Series.

    Args:
        sic_series (pd.Series): A pandas Series containing SIC codes as strings.
                                Missing values should be pre-filled (e.g., with '').

    Returns:
        Dict[str, pd.Series]: A dictionary where keys are flag names
                              (e.g., "Match_5_digits") and values are
                              boolean pandas Series.
    """
    flags = {}

    # Match 5 digits: ^[0-9]{5}$
    flags["Match_5_digits"] = sic_series.str.match(r"^[0-9]{5}$", na=False)

    # For partial matches (N digits + X 'x's)
    is_len_expected = sic_series.str.len() == EXPECTED_SIC_LENGTH
    x_count = sic_series.str.count("x", re.I)  # Count 'x' case-insensitively
    only_digits_or_x = sic_series.str.match(r"^[0-9xX]*$", na=False)
    non_x_part = sic_series.str.replace("x", "", case=False)
    # Ensure non_x_part is not empty before checking if it's all digits
    are_non_x_digits = (non_x_part != "") & non_x_part.str.match(r"^[0-9]*$", na=False)

    base_partial_check = is_len_expected & only_digits_or_x & are_non_x_digits
    flags["Match_3_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_3)
    flags["Match_2_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_2)

    return flags


def _extract_sic_division(
    sic_occ1_series: pd.Series,
    not_codeable_flag: pd.Series,
    multiple_possible_flag: pd.Series,
) -> pd.Series:
    """Extracts the first two digits (division) from the sic_ind_occ1 series.

    Args:
        sic_occ1_series (pd.Series): The Series containing sic_ind_occ1 codes (as strings).
        not_codeable_flag (pd.Series): Boolean Series indicating where sic_ind_occ1 is '-9'.
        multiple_possible_flag (pd.Series): Boolean Series indicating where sic_ind_occ1 is '4+'.

    Returns:
        pd.Series: A Series containing the first two digits as strings, or an
                   empty string if not applicable or for special codes.
    """
    logger.debug("Extracting SIC division (first two digits) from sic_ind_occ1.")
    # Default to empty string
    sic_division = pd.Series("", index=sic_occ1_series.index, dtype=str)

    # Condition for valid extraction:
    # Must start with at least two digits AND not be a special code.
    # Using .str.match() ensures we are dealing with strings.
    starts_with_two_digits = sic_occ1_series.str.match(
        r"^[0-9]{2}"
    )  # Matches if starts with 2+ digits

    # Rows eligible for extraction
    eligible_for_extraction = (
        starts_with_two_digits & ~not_codeable_flag & ~multiple_possible_flag
    )

    # Extract first two digits for eligible rows
    sic_division[eligible_for_extraction] = sic_occ1_series[
        eligible_for_extraction
    ].str[:2]

    logger.debug("Finished extracting SIC division.")
    return sic_division


# --- Data Quality Flagging ---
def add_data_quality_flags(
    df: pd.DataFrame, loaded_config: Optional[dict[str, Any]] = None
) -> pd.DataFrame:
    """Adds data quality flag columns to the DataFrame based on SIC/SOC codes.

    Args:
        df (pd.DataFrame): The input DataFrame
        loaded_config (Optional[dict]): Loaded configuration dictionary to get column names.

    Returns:
        pd.DataFrame: The DataFrame with added boolean quality flag columns.
                      Returns original DataFrame if essential columns are missing.
    """
    logger.info("Adding data quality flag columns...")
    df_out = df.copy()  # Work on a copy

    # Get column names from config or use defaults
    col_occ1 = (
        loaded_config.get("column_names", {}).get("sic_ind_occ1", "sic_ind_occ1")
        if loaded_config
        else "sic_ind_occ1"
    )
    col_occ2 = (
        loaded_config.get("column_names", {}).get("sic_ind_occ2", "sic_ind_occ2")
        if loaded_config
        else "sic_ind_occ2"
    )
    col_occ3 = (
        loaded_config.get("column_names", {}).get("sic_ind_occ3", "sic_ind_occ3")
        if loaded_config
        else "sic_ind_occ3"
    )

    # Check if essential columns exist
    required_input_cols = [col_occ1, col_occ2, col_occ3]
    if not all(col_name in df_out.columns for col_name in required_input_cols):
        missing_cols = set(required_input_cols) - set(df_out.columns)
        logger.error(
            "Input DataFrame missing columns for quality flags: %s. Skipping flag generation.",
            missing_cols,
        )
        return df  # Return original df

    # --- 1. Special SIC Code Flags for col_occ1 ---
    df_out["Not_Codeable"] = df_out[col_occ1] == SPECIAL_SIC_NOT_CODEABLE
    df_out["Multiple_Possible_SICs"] = df_out[col_occ1] == SPECIAL_SIC_MULTIPLE_POSSIBLE

    # Add SIC division (2 digits)
    df_out["SIC_Division"] = _extract_sic_division(
        df_out[col_occ1], df_out["Not_Codeable"], df_out["Multiple_Possible_SICs"]
    )

    # --- 2. Number of Answers ---
    df_out["num_answers"] = _calculate_num_answers(df_out, col_occ1, col_occ2, col_occ3)

    # handle the edge cases of -9:
    df_out.loc[df_out[col_occ1] == SPECIAL_SIC_NOT_CODEABLE, "num_answers"] = 0
    # and of 4+
    df_out.loc[df_out[col_occ1] == SPECIAL_SIC_MULTIPLE_POSSIBLE, "num_answers"] = 4

    # --- 3. Digit/Character Match Flags for col_occ1 ---
    s_occ1 = df_out[col_occ1].fillna("").astype(str)
    match_flags_dict = _create_sic_match_flags(s_occ1)

    # --- 4.  col_occ1 ---
    # Create a new column 'Combined' with a list of values from columns A, B, and C
    df_out["All_Clerical_codes"] = df_out.apply(
        lambda row: [row[col_occ1], row[col_occ2], row[col_occ3]], axis=1
    )

    for flag_name, flag_series in match_flags_dict.items():
        df_out[flag_name] = flag_series

    # --- 3. Unambiguous Flag ---
    # Ensure "Match_5_digits" was created by the helper
    if "Match_5_digits" in df_out.columns:
        df_out["Unambiguous"] = df_out["num_answers"] == 1 & df_out[
            "Match_5_digits"
        ].fillna(False)
    else:
        # Handle case where Match_5_digits might not be created if s_occ1 was problematic
        # Though _create_sic_match_flags should always return it.
        df_out["Unambiguous"] = False
        logger.warning(
            "'Match_5_digits' column not found for 'Unambiguous' flag calculation."
        )

    # --- 4. Convert to Pandas Nullable Boolean Type ---
    flag_cols_list = [
        "Match_5_digits",
        "Match_3_digits",
        "Match_2_digits",
        "Unambiguous",
    ]

    for flag_col_name in flag_cols_list:
        if flag_col_name in df_out.columns:
            try:
                df_out[flag_col_name] = df_out[flag_col_name].astype("boolean")
            except (TypeError, ValueError) as e:  # Catch specific errors
                logger.warning(
                    "Could not convert column '%s' to boolean dtype: %s",
                    flag_col_name,
                    e,
                )

    logger.info("Finished adding data quality flag columns.")
    logger.debug("Flag columns added: %s", flag_cols_list)

    if logger.isEnabledFor(logging.DEBUG):
        print(f"--- DataFrame info after adding flags (for {len(df_out)} rows) ---")
        df_out.info()
        print("----------------------------------------------------------")

    return df_out


if __name__ == "__main__":

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Where the input data csv is. We'll use the batch filepath from batch script
    analysis_filepath = main_config["paths"]["batch_filepath"]

    # We'll write the post analysis csv here:
    analysis_csv = main_config["paths"]["analysis_csv"]

    # Check the location exists:
    # Extract the directory from the file path
    output_dir = os.path.dirname(analysis_csv)

    # Check if it exists
    try:
        if os.path.exists(output_dir):
            logging.info("Directory exists: %s ", output_dir)
        else:
            logging.warning("Directory does not exist: %s ", output_dir)
            raise FileNotFoundError(
                f"Required output directory not found: {output_dir}"
            )
    except KeyError as e:
        logging.error("Missing configuration key: %s", e)

    # Load the data
    sic_dataframe = pd.read_csv(analysis_filepath, delimiter=",", dtype=str)

    # add quality flags
    sic_dataframe_with_flags = add_data_quality_flags(sic_dataframe, main_config)

    print("\nDataFrame Head with Quality Flags:")
    print(sic_dataframe_with_flags.head())

    # Explanation of logic:

    print("Match_5_digits - a value in sic_ind_occ1 has 5 numeric digits")
    print("Match_3_digits - a value in sic_ind_occ1 has 3 numeric digits and 2 of x")
    print("Match_2_digits - a value in sic_ind_occ1 has 2 numeric digits and 3 of x")
    print("Unambiguous - value in ")

    print("\nValue Counts for Quality Flags:")
    flag_cols_to_show = [
        "Match_5_digits",
        "Match_3_digits",
        "Match_2_digits",
        "Unambiguous",
    ]

    for col_to_show in flag_cols_to_show:
        if col_to_show in sic_dataframe_with_flags.columns:
            print(f"\n--- {col_to_show} ---")
            print(sic_dataframe_with_flags[col_to_show].value_counts(dropna=False))

    # write new dataframe out:
    sic_dataframe_with_flags.to_csv(analysis_csv, index=False)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
