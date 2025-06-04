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
# # Analysis of TLFS SIC Evaluation Dataset
# Intro: this notebook contain the following analysis of TLFS_evaluation_data_IT2.csv
#  - Breakdown of  2 digit codes (Division)
#  - Breakdown of all codes across all three choices
#  - 4+ represent a small %, less than 1%, and are excluded from this.
#  - Analysis of the number of codes CCs applied (truncated at 4)
#  - Most frequent codes when unambiguous only
#  - proportion of codeable at 5-digit across the total set
#
# Key takeaway for business value:
#  - Unambiguous Codes represent 60% of this dataset.
#  - A confirmation for early stopping of the quesioning would be a potential business value.
#  - Ambiguous answers representing the remainder, contain only 6.8% uncodeable (represented by
# the code -9 in this data) - an opportunity for SA to add value.
#  - A specific set, 28% are coded to two digits, and would benefit from a follow up question.
#
# It is worth knowing that SIC/SOC project described this data as from the set most likely to
# give problems.
#
# Summary Statistics:
# Group memberships:
# - Uncodeable to any digits: 6.79 %
# - Codeable to two digits only:  27.6 %
# - Codeable to five Digits (ambiguous and unambiguous): 63.8 %
# - Codeable unambiguously to five digits: 59.7 %
# - Codeable ambiguously to five digits: 4.1 %
# - Codeable at 2 or more digits, 92%
#
#
#
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
# ## Set import graphical package, logging

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


# %% [markdown]
# ### Make a graphing function


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
        counts = df[column_name].value_counts(normalize=True).nlargest(top_n) * 100
        ylabel_text = "Percentage"
    else:
        counts = df[column_name].value_counts().nlargest(top_n)
        ylabel_text = "Frequency (Count)"

    if counts.empty:
        logger.warning("No data to plot for histogram of column '%s'.", column_name)
        plt.close()  # Close the empty figure
        return

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
# ## Section 1: Distribution of the SIC codes of the labelled set

# %% [markdown]
# ### Histogram of the top 10 2 digit codes (Division)

# %%
output_dir_path = Path("/home/user/survey-assist-utils/notebooks")
plot_sic_code_histogram(
    eval_data,
    column_name="SIC_Division",
    output_dir=output_dir_path,
    show_percent=True,
    filename_suffix="SIC_Division",
)

# %%
# Calculate membership of 86 and 87  combined
print(
    f"""Fraction of the dataset being in Division 86 or 87:
    {100 * eval_data['SIC_Division'].isin(['86', '87']).sum() / len(eval_data):.1f}%"""
)

# %% [markdown]
# ### Finding 1 - Strong bias to Divisions 86, 87: 68.2%
#
# ### Implication for our work:
#
# A wider range of evaluation data is requested to remove the risks
# associated with bias, such as a difference performance in one division compared to another.

# %% [markdown]
# ### Histogram of the top 10 of all codes across all three choices, including 2 digits

# %%
combined_data = pd.concat(
    [eval_data["sic_ind_occ1"], eval_data["sic_ind_occ2"], eval_data["sic_ind_occ3"]]
)
combined_df = pd.DataFrame(combined_data, columns=["combined"])
plot_sic_code_histogram(
    combined_df,
    column_name="combined",
    output_dir=output_dir_path,
    show_percent=True,
    filename_suffix="Combined_sic_codes",
)

# %% [markdown]
# ### Histogram for unambiguous only:

# %%
filtered_data = eval_data[eval_data["Unambiguous"]]
plot_sic_code_histogram(
    filtered_data,
    column_name="sic_ind_occ1",
    output_dir=output_dir_path,
    show_percent=True,
    filename_suffix="Unambiguous_only",
)

# %% [markdown]
# ### Histogram of the number of codes CCs applied:
# Distribution of number of possible SIC codes (uncodeable - 0 code, 1 code, 2 codes, 3 codes)

# %%
plot_sic_code_histogram(
    eval_data,
    column_name="num_answers",
    output_dir=output_dir_path,
    show_percent=True,
    filename_suffix="",
)

# %%
print(
    f"""Single answer given {100*(eval_data["num_answers"] == 1).sum() /
        len(eval_data["num_answers"]):.2f} %"""
)

# %% [markdown]
# ### Finding 2: A single answer is given in 76% of cases.
# In over three quareters of cases, coders are sufficiently confident to provide a single answer
# only (with either 2 or five digits defined).
#
# ### Implication for our work:
#
# Within this 3/4ths majority, there is sufficient information given that a single
# definite answer is arrived at.
#
# We should understand whether the LLM is able to do the same.
#
#
#

# %% [markdown]
# ## Section 2: levels of codeability in the labelled set:
#
# ### Calculate proportion of codeable at 5-digit across the total set.
# This is to answer the question how many responses don't need a follow up?
#
# - Uncodeable to any digits: 6.79 %
# - Codeable to two digits only:  27.6 %
# - Codeable to five Digits (ambiguous and unambiguous): 63.8 %
# - Codeable unambiguously to five digits: 59.7 %
# - Codeable ambiguously to five digits: 4.1 %

# %%
print("Group memberships:")
print(
    f"""Uncodeable to any digits: {100*(eval_data["num_answers"] == 0).sum() /
        len(eval_data["num_answers"]):.2f} %"""
)
print(
    f"""Codeable to two digits only:  {100*eval_data['Match_2_digits'].sum() /
        len(eval_data):.1f} %"""
)
print(
    f"""Codeable to five Digits (ambiguous and unambiguous):
        {100*eval_data['Match_5_digits'].sum() /len(eval_data):.1f} %"""
)
print(
    f"""Codeable unambiguously to five digits:
        {100*eval_data['Unambiguous'].sum() / len(eval_data["Match_5_digits"]):.1f} %"""
)

ambiguous_data = eval_data[~eval_data["Unambiguous"]]
print(
    f"""Codeable ambiguously to five digits:
        {100*ambiguous_data["Match_5_digits"].sum() / len(eval_data["Match_5_digits"]):.1f} %"""
)

print(f"Number of True in 'Match_5_digits': {eval_data['Match_5_digits'].sum()}")
print(
    f"Fraction of True 'Match_5_digits': {eval_data['Match_5_digits'].sum() /len(eval_data):.1f}"
)

print(f"Number of True in 'Match_2_digits': {eval_data['Match_2_digits'].sum()}")
print(
    f"Fraction of True 'Match_2_digits': {eval_data['Match_2_digits'].sum() /len(eval_data):.2f}"
)

filtered_data = eval_data[
    eval_data["Match_5_digits"]
    | eval_data["Match_3_digits"]
    | eval_data["Match_2_digits"]
]
true_count_2_or_more = len(filtered_data)
fraction_2_or_more = true_count_2_or_more / len(eval_data)
print(
    f"""Number of True values in 'Match_2_digits',
 'Match_3_digits' or 'Match_5_digits': {true_count_2_or_more}"""
)
print(
    f"Fraction of True values in 'Match_2_digits' or more digits: {fraction_2_or_more:.2f}"
)
print(f"Total Dataset: {len(eval_data)}")


# %% [markdown]
# ### Histogram of Codeable at 2 or more digits for 'Unambiguous' category:

# %%
# codeable at 2d
filtered_data = eval_data[
    eval_data["Match_5_digits"]
    | eval_data["Match_3_digits"]
    | eval_data["Match_2_digits"]
]
plot_sic_code_histogram(
    filtered_data,
    column_name="sic_ind_occ1",
    output_dir=output_dir_path,
    show_percent=True,
    filename_suffix="Codeable",
)

# %% [markdown]
# ### Findings and implications:
# Strong skew to 86xxx and 87xxx, 68.2%, representing the divisions that this data were taken from.
# A more diverse data set has been requested.
#
# Uncodeable are a small percentage 6.8%.
#
# Items that have been coded only to two digits will benefit from SA as a system, 28% in an attempt
# to identify the detail of the remaining three digits.
#
# The unambiguous set of 60% represent an opportunity to trigger an early stop of the survey if the
# LLM is able to match this with confidence.
#
# 75% having a single answer represent an opportunity to test whether the LLM can also arrive at a
#  definitive answer for this set.
#
#
