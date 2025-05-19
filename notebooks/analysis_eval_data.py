# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: survey-assist-utils-PWI-TvqZ-py3.12
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Main Section
#

# %%

"""Summary analysis of the TLFS_evaluation_data_IT2 dataset,
    prepared via `prepare_evaluation_data_for_analysis`.

Key outputs include:
- Coverage of 2-digit SIC groups
- Distribution of possible SIC codes per record
- Codeability and unambiguous coding stats at 2-digit and 5-digit levels
- Analysis of derived SIC from `sic_ind_occ1`
- Identification of uncodeable records and those requiring follow-up

Developed in a notebook and converted to script for version control and reproducibility.
"""

# %% [markdown]
# ## Subsection
# Set constants, import graphical package, logging

import logging
from pathlib import Path

import matplotlib.pyplot as plt

# %%
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# %%
cols_wanted = [
    "unique_id",
    "sic_ind_occ1",
    "sic_ind_occ2",
    "sic_ind_occ3",
    "sic_ind_occ_flag",
    "Not_Codeable",
    "Four_Or_More",
    "SIC_Division",
    "num_answers",
    "All_Clerical_codes",
    "Match_5_digits",
    "Match_3_digits",
    "Match_2_digits",
    "Unambiguous",
]

# Load your DataFrame
eval_data = pd.read_csv(
    "../data/analysis_outputs/TLFS_evaluation_data_IT2_output.csv",
    usecols=cols_wanted,
    dtype={"SIC_Division": str},
)

# %% [markdown]
# ### Set constants

# %%
DEFAULT_OUTPUT_DIR = "analysis_outputs"
DEFAULT_SIC_OCC1_COL = "sic_ind_occ1"
DEFAULT_SIC_OCC2_COL = "sic_ind_occ2"
TOP_N_HISTOGRAM = 10  # Number of top items to show in SIC code histograms

# --- Constants for Data Quality ---
EXPECTED_SIC_LENGTH = 5
X_COUNT_FOR_MATCH_3 = 2
X_COUNT_FOR_MATCH_2 = 3


# %%
def plot_sic_code_histogram(
    df: pd.DataFrame,
    column_name: str,
    output_dir: Path,
    show_percent=False,
    filename_suffix: str = "",
) -> None:
    """Generates and saves a histogram (bar plot) for the value counts of a SIC code column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the SIC code column to analyze.
        output_dir (Path): The directory to save the plot.
        show_percent (bool): Whether to display percentages instead of raw counts.
        filename_suffix (str): Suffix to add to the plot filename.
    """
    top_n = TOP_N_HISTOGRAM
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

    plt.figure(figsize=(12, 8))

    # Option for percentages
    if show_percent:
        print(show_percent)
        counts = df[column_name].value_counts(normalize=True).nlargest(top_n) * 100
        ylabel_text = "Percentage"
    else:
        counts = df[column_name].value_counts().nlargest(top_n)
        ylabel_text = "Frequency (Count)"

    if counts.empty:
        logger.warning("No data to plot for histogram of column '%s'.", column_name)
        plt.close()  # Close the empty figure
        return

    # sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette="viridis",
        dodge=False,
        legend=False,
    )

    plt.title(
        f"Top {top_n} Most Frequent Codes in '{column_name}' (Total Rows: {len(df)})"
    )
    plt.xlabel(f"{column_name} Code")
    plt.ylabel(ylabel_text)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_filename = f"{column_name.lower()}_distribution{filename_suffix}.png"
    output_path = output_dir / output_filename
    plt.savefig(output_path)
    logger.info("Histogram saved to %s", output_path)
    plt.show()

    plt.close()


# %% [markdown]
# ### Sub-subsection
#
# First make a histogram of the top 10 2digit codes (Division)

# %%
output_dir_path = Path("/home/user/survey-assist-utils/notebooks")
plot_sic_code_histogram(
    eval_data,
    column_name="SIC_Division",
    output_dir=output_dir_path,
    show_percent=True,
    filename_suffix="SIC_Division",
)
