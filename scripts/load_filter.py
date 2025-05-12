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
import re
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config_loader import load_config  # Using the imported load_config

logger = logging.getLogger(__name__)

# --- Default Configuration Values (if not found in config) ---
DEFAULT_OUTPUT_DIR = "analysis_outputs"
DEFAULT_SIC_OCC1_COL = "sic_ind_occ1"
DEFAULT_SIC_OCC2_COL = "sic_ind_occ2"
TOP_N_HISTOGRAM = 20  # Number of top items to show in SIC code histograms

# --- Constants for Data Quality ---
EXPECTED_SIC_LENGTH = 5
X_COUNT_FOR_MATCH_4 = 1
X_COUNT_FOR_MATCH_3 = 2
X_COUNT_FOR_MATCH_2 = 3
X_COUNT_FOR_MATCH_1 = 4


def plot_sic_code_histogram(
    df: pd.DataFrame,
    column_name: str,
    output_dir: Path,
    top_n: int = TOP_N_HISTOGRAM,
    filename_suffix: str = "",
) -> None:
    """Generates and saves a histogram (bar plot) for the value counts of a SIC code column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the SIC code column to analyze.
        output_dir (Path): The directory to save the plot.
        top_n (int): Number of top most frequent items to display.
        filename_suffix (str): Suffix to add to the plot filename.
    """
    if column_name not in df.columns:
        logger.warning(
            "Column '%s' not found in DataFrame. Skipping histogram generation.",
            column_name,
        )
        return
    if df[column_name].isnull().all():
        logger.warning(
            "Column '%s' contains all null/NaN values. Skipping histogram generation.",
            column_name,
        )
        return

    logger.info("Generating histogram for column: %s", column_name)
    plt.figure(figsize=(12, 8))
    counts = df[column_name].value_counts().nlargest(top_n)

    if counts.empty:
        logger.warning("No data to plot for histogram of column '%s'.", column_name)
        plt.close()  # Close the empty figure
        return

    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.title(
        f"Top {top_n} Most Frequent Codes in '{column_name}' (Total Rows: {len(df)})"
    )
    plt.xlabel(f"{column_name} Code")
    plt.ylabel("Frequency (Count)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_filename = f"{column_name.lower()}_distribution{filename_suffix}.png"
    output_path = output_dir / output_filename
    plt.savefig(output_path)
    logger.info("Histogram saved to %s", output_path)

    plt.close()


def plot_boolean_flag_summary(df: pd.DataFrame, flag_columns: list[str]) -> None:
    """Generates and saves a bar plot summarising counts and percentages of True values
    for specified boolean flag columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        flag_columns (list[str]): A list of boolean column names to analyze.
    """
    valid_flag_columns = [col for col in flag_columns if col in df.columns]
    if not valid_flag_columns:
        logger.warning(
            "None of the specified flag columns found. Skipping summary plot."
        )
        return

    logger.info("Generating summary plot for boolean flags: %s", valid_flag_columns)

    true_counts = []
    percentages = []
    total_rows = len(df)

    if total_rows == 0:
        logger.warning("Input DataFrame is empty. Skipping boolean flag summary.")
        return

    for col in valid_flag_columns:
        # Ensure column is boolean or can be safely summed (True=1, False=0)
        # Pandas nullable boolean 'boolean' dtype handles sum correctly (ignores NA)
        if pd.api.types.is_bool_dtype(df[col]) or df[col].dtype.name == "boolean":
            count = df[col].sum()  # True values are summed as 1
            true_counts.append(count)
            percentages.append((count / total_rows) * 100 if total_rows > 0 else 0)

    summary_df = pd.DataFrame(
        {
            "Flag": valid_flag_columns,
            "Count_True": true_counts,
            "Percentage_True": percentages,
        }
    )

    if summary_df.empty:
        logger.warning("No data to plot for boolean flag summary.")
        return

    # Plotting Counts
    plt.figure(figsize=(12, 7))
    barplot = sns.barplot(x="Flag", y="Count_True", data=summary_df, palette="Blues_d")
    plt.title(f"Count of TRUE Values for Quality Flags (Total Rows: {total_rows})")
    plt.xlabel("Flag Name")
    plt.ylabel("Count of TRUE")
    plt.xticks(rotation=45, ha="right")

    # Add percentage annotations
    for i, p in enumerate(barplot.patches):
        percentage_val = summary_df.loc[i, "Percentage_True"]
        barplot.text(
            p.get_x() + p.get_width() / 2.0,
            p.get_height() + 0.01 * total_rows,  # Offset text slightly above bar
            f"{p.get_height()}\n({percentage_val:.1f}%)",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    plt.savefig("boolean_flags_summary.png")
    logger.info("Boolean flags summary plot saved to boolean_flags_summary.png")

    plt.close()


def perform_graphical_analysis(
    df: pd.DataFrame,
    config: Optional[dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> None:
    """Performs graphical analysis on the DataFrame for SIC codes and quality flags.

    Args:
        df (pd.DataFrame): The DataFrame to analyze (e.g., sic_dataframe_with_flags).
        config (Optional[dict[str, Any]]): Loaded configuration dictionary.
        run_id (Optional[str]): An optional ID to append to filenames for uniqueness.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping graphical analysis.")
        return

    # Determine output directory
    output_dir_str = DEFAULT_OUTPUT_DIR
    project_root = Path.cwd()  # Default if config or project_root not easily found
    if config and "paths" in config and "output_dir" in config["paths"]:
        # Config paths are relative to a project root
        output_dir_str = config["paths"]["output_dir"]
        output_dir = Path(output_dir_str)
    else:
        output_dir = project_root / output_dir_str
        logger.warning(
            "Output directory not found in config. Using default: %s", output_dir
        )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.exception(
            "Could not create output directory %s: %s. Plots will not be saved.",
            output_dir,
            e,
        )
        return  # Cannot save plots if directory creation fails

    filename_suffix = f"_{run_id}" if run_id else ""

    # Get column names from config or use defaults
    col_names_config = config.get("column_names", {}) if config else {}
    sic_occ1_col = col_names_config.get(
        "gold_sic", DEFAULT_SIC_OCC1_COL
    )  # Using 'gold_sic' from config
    # Assuming 'sic_ind_occ2' is not explicitly in config, use a default or add it
    sic_occ2_col = col_names_config.get("input_csv_columns_map", {}).get(
        "sic_ind_occ2", DEFAULT_SIC_OCC2_COL
    )

    # Analysis for sic_ind_occ1 and sic_ind_occ2
    plot_sic_code_histogram(
        df, sic_occ1_col, output_dir, filename_suffix=filename_suffix
    )
    plot_sic_code_histogram(
        df, sic_occ2_col, output_dir, filename_suffix=filename_suffix
    )

    # Analysis for Boolean Flag Columns
    boolean_flags_to_analyze = [
        "Single_Answer",
        "Unambiguous",
        "Match_5_digits",
        "Match_2_digits",
    ]
    plot_boolean_flag_summary(df, boolean_flags_to_analyze)

    logger.info("Graphical analysis complete. Plots saved to %s", output_dir)


# --- Data Quality Flagging ---
def add_data_quality_flags(
    df: pd.DataFrame, loaded_config: Optional[dict[str, Any]] = None
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

    # --- 1. Single Answer Flag ---
    is_occ2_missing = df_out[col_occ2].isna() | (
        df_out[col_occ2].astype(str).str.upper() == "NA"
    )
    df_out["Single_Answer"] = is_occ2_missing

    # --- 2. Digit/Character Match Flags for col_occ1 ---
    s_occ1 = df_out[col_occ1].fillna("").astype(str)
    df_out["Match_5_digits"] = s_occ1.str.match(r"^[0-9]{5}$", na=False)

    # Check are all 5 digits long and consist of only 'x' or numeric
    x_count = s_occ1.str.count("x", re.I)
    only_digits_or_x = s_occ1.str.match(r"^[0-9xX]*$", na=False)
    non_x_part = s_occ1.str.replace("x", "", case=False)
    are_non_x_digits = non_x_part.str.match(r"^[0-9]*$", na=False) & (non_x_part != "")
    base_partial_check = (
        (s_occ1.str.len() == EXPECTED_SIC_LENGTH) & only_digits_or_x & are_non_x_digits
    )

    df_out["Match_3_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_3)
    df_out["Match_2_digits"] = base_partial_check & (x_count == X_COUNT_FOR_MATCH_2)

    # --- 3. Unambiguous Flag ---
    df_out["Unambiguous"] = df_out["Single_Answer"].fillna(False) & df_out[
        "Match_5_digits"
    ].fillna(False)

    # --- 4. Convert to Pandas Nullable Boolean Type ---
    flag_cols_list = [
        "Single_Answer",
        "Match_5_digits",
        "Match_3_digits",
        "Match_2_digits",
        "Unambiguous",
    ]

    # Add flags from list
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

    # We'll write the post analysis csv here:
    analysis_csv = main_config["paths"]["analysis_csv"]

    # Load the data
    sic_dataframe = pd.read_csv(analysis_filepath, delimiter=",", dtype=str)

    # add quality flags
    sic_dataframe_with_flags = add_data_quality_flags(sic_dataframe, main_config)

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
            print(sic_dataframe_with_flags[col_to_show].value_counts(dropna=False))

    # write new dataframe out:
    sic_dataframe_with_flags.to_csv(analysis_csv, index=False)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # Display an analysis of the results
    perform_graphical_analysis(sic_dataframe_with_flags, config=main_config)
