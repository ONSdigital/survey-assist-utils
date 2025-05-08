"""Load and Filter utilities.

The .toml configuration file should include:
- The path to the test data file.
- Column names mapping if they differ from defaults.

The utilities allow for the following:
1. Loads the configuration from the .toml file.
2. Loads the test data.
3. Adds data quality flag columns to the DataFrame.
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from src.config_loader import load_config  # Using the imported load_config

logger = logging.getLogger(__name__)

# --- Constants for Data Quality ---
EXPECTED_SIC_LENGTH = 5
X_COUNT_FOR_MATCH_4 = 1
X_COUNT_FOR_MATCH_3 = 2
X_COUNT_FOR_MATCH_2 = 3
X_COUNT_FOR_MATCH_1 = 4

# --- Data Quality Flagging ---


# pylint: disable=too-many-locals
def add_data_quality_flags(
    df: pd.DataFrame, loaded_config: Optional[dict[str, Any]] = None  # Renamed 'config'
) -> pd.DataFrame:
    """Adds data quality flag columns to the DataFrame based on SIC/SOC codes.

    Args:
        df (pd.DataFrame): The input DataFrame (typically loaded by read_sic_data).
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

    required_input_cols = [col_occ1, col_occ2, col_occ3]
    if not all(col_name in df_out.columns for col_name in required_input_cols):
        missing_cols = set(required_input_cols) - set(df_out.columns)
        logger.error(
            "Input DataFrame missing columns for quality flags: %s. Skipping flag generation.",
            missing_cols,
        )
        return df  # Return original df

    # --- 1. Single Answer Flag ---
    is_occ2_missing = df_out[col_occ2].isna() | (
        df_out[col_occ2].astype(str).str.upper() == "NA"
    )
    is_occ3_missing = df_out[col_occ3].isna() | (
        df_out[col_occ3].astype(str).str.upper() == "NA"
    )
    df_out["Single_Answer"] = is_occ2_missing & is_occ3_missing

    # --- 2. Digit/Character Match Flags for col_occ1 ---
    s_occ1 = df_out[col_occ1].fillna("").astype(str)
    df_out["Match_5_digits"] = s_occ1.str.match(r"^[0-9]{5}$", na=False)
    is_len_expected = s_occ1.str.len() == EXPECTED_SIC_LENGTH
    x_count = s_occ1.str.count("x", re.I)
    only_digits_or_x = s_occ1.str.match(r"^[0-9xX]*$", na=False)
    non_x_part = s_occ1.str.replace("x", "", case=False)
    are_non_x_digits = non_x_part.str.match(r"^[0-9]*$", na=False) & (non_x_part != "")
    base_partial_check = is_len_expected & only_digits_or_x & are_non_x_digits

    df_out["Match_4_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_4)
    df_out["Match_3_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_3)
    df_out["Match_2_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_2)
    df_out["Match_1_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_1)

    # --- 3. Unambiguous Flag ---
    df_out["Unambiguous"] = df_out["Single_Answer"].fillna(False) & df_out[
        "Match_5_digits"
    ].fillna(False)

    # --- 4. Convert to Pandas Nullable Boolean Type ---
    flag_cols_list = [  # Renamed to avoid clash if 'flag_cols' is used elsewhere
        "Single_Answer",
        "Match_5_digits",
        "Match_4_digits",
        "Match_3_digits",
        "Match_2_digits",
        "Match_1_digits",
        "Unambiguous",
    ]
    # W0621: Renamed loop variable 'col' to 'flag_col_name'
    for flag_col_name in flag_cols_list:
        if flag_col_name in df_out.columns:
            try:
                df_out[flag_col_name] = df_out[flag_col_name].astype("boolean")
            except (TypeError, ValueError) as e:
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
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    
    # Load configuration from .toml file
    main_config = load_config("config.toml")

    # Where the input data csv is. We'll use the batch filepath from batch script
    analysis_filepath = main_config["paths"]["batch_filepath"]

    # Load the data
    sic_dataframe = pd.read_csv(analysis_filepath, delimiter=",", dtype=str)

    # add quality flags
    sic_dataframe_with_flags = add_data_quality_flags(
        sic_dataframe, main_config
    )




    print("\nDataFrame Head with Quality Flags:")
    print(sic_dataframe_with_flags.head())

    print("\nValue Counts for Quality Flags:")
    flag_cols_to_show = [
        "Single_Answer",
        "Match_5_digits",
        "Match_4_digits",
        "Match_3_digits",
        "Match_2_digits",
        "Match_1_digits",
        "Unambiguous",
    ]

    for col_to_show in flag_cols_to_show:
        if col_to_show in sic_dataframe_with_flags.columns:
            print(f"\n--- {col_to_show} ---")
            print(
                sic_dataframe_with_flags[col_to_show].value_counts(
                    dropna=False
                )
            )
