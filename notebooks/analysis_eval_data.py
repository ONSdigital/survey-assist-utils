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

# %%

"""Summary analysis of the TLFS_evaluation_data_IT2 dataset."""

# %% [markdown]
# # Analysis of TLFS SIC Evaluation Dataset
#
# What the data are:
#
# The first iteration of the TLFS SIC Evaluation Dataset which contains 13,274 TLFS records.
# SIC codes are derived from both the Occupation and Industry responses (sic_ind_occ variables)
#
# Key outputs include:
# - Coverage of 2-digit SIC groups
# - Distribution of possible SIC codes per record
# - Codeability and unambiguous coding stats at 2-digit and 5-digit levels
# - Analysis of derived SIC from `sic_ind_occ1`
# - Identification of uncodeable records and those requiring follow-up
#
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
# - Codeable to three digits only	0.7 %
# - Codeable unambiguously to five digits: 59.7 %
# - Codeable ambiguously to five digits: 4.1 %
# - Having 4 or more codes: 1%
# - Other (coded to four digits): 0.1%
# - Total	items 13274
#
#
# ### Unambiguouisly codeable being 60% of the data.
#  - An opportunity to test whether the LLM is able to match the human coder's
#  assessment.
#
#

# %% [markdown]
# ## Set import graphical package, logging
#
# we import the dependancies next

# %%
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

# %%
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# %%
# Load the DataFrame
eval_data = pd.read_csv(
    "../data/analysis_outputs/TLFS_evaluation_data_IT2_output.csv",
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
X_COUNT_FOR_MULTI_ANSWERS = 4

output_dir_path = Path("/home/user/survey-assist-utils/notebooks")


# %% [markdown]
# ## Section 1: Group Membership of Total Data
#  This section calculates and displays a pie chart representing the main group
#  memberships within the dataset:
#  - Uncodeable to any digits
#  - Codeable to two digits only
#  - Codeable to three digits only
#  - Codeable unambiguously to five digits
#  - Codeable ambiguously to five digits
#  - 4 or more answers given
#  - Any remaining data as 'Other'
#
#  ## Finding 1: A single answer is given in 76% of cases.
# In over three quarters of cases, coders are sufficiently confident to provide a single answer
# only (with either 2 or five digits defined).
#
# # Implication for our work:
# - Make an initial test of the LLM's ability to match the human codres' results.
# - Use this as a benchmark for ongoing improvements.

# %%
# --- Calculate counts for each group membership category ---
N = len(eval_data)

# 1. Uncodeable to any digits
count_uncodeable = (eval_data["num_answers"] == 0).sum()

# 2. Codeable to two digits only
count_2_digits_only = eval_data["Match_2_digits"].sum()

# 3. Codeable unambiguously to five digits
count_5_digits_unambiguous = eval_data["Unambiguous"].sum()

# 4. Codeable ambiguously to five digits
# These are items that Match_5_digits but are NOT Unambiguous.
count_5_digits_ambiguous = (
    eval_data["Match_5_digits"] & ~eval_data["Unambiguous"]
).sum()

# Check what is in the small remaining quantity:
count_3_digits_only = eval_data["Match_3_digits"].sum()
count_four_plus = (eval_data["num_answers"] == X_COUNT_FOR_MULTI_ANSWERS).sum()

# --- Define data for the pie chart ---
group_counts_for_pie = [
    count_uncodeable,
    count_four_plus,
    count_2_digits_only,
    count_3_digits_only,
    count_5_digits_unambiguous,
    count_5_digits_ambiguous,
]

group_labels_template = [
    "Uncodeable",
    "4+ Codes",
    "Codeable to 2 digits only",
    "Codeable to 3 digits only",
    "Codeable unambiguously to 5 digits",
    "Codeable ambiguously to 5 digits",
]


# Check for 'Other' category (data not falling into the above groups)
total_categorised_count = sum(group_counts_for_pie)
count_other = N - total_categorised_count
group_counts_for_pie.append(count_other)
group_labels_template.append("Other")

# Calculate percentages for precise labeling
calculated_percentages = [(count / N) * 100 for count in group_counts_for_pie]

# Create labels with percentages
chart_labels = [
    f"{label}\n({percent:.1f}%)"
    for label, percent in zip(group_labels_template, calculated_percentages)
]

# --- Create and save the pie chart ---
plt.figure(figsize=(10, 10))
plt.pie(
    group_counts_for_pie,
    labels=chart_labels,
    autopct=lambda p: (f"{p:.1f}%" if p > 1 else ""),  # Show autopct % for slices > 1%
    startangle=90,
    wedgeprops={"edgecolor": "white"},  # Adds a white border to slices
    colors=sns.color_palette(
        "pastel", len(group_counts_for_pie)
    ),  # Use a seaborn palette
)
plt.title("Distribution of Group Membership in Total Data", fontsize=16)
plt.axis("equal")
plt.tight_layout()

# Add a comment about the 'other'
plt.text(
    -1,
    -1.2,
    (f"*Other - 4 digit coded\n({calculated_percentages[-1]:.1f}%)"),
    bbox={"facecolor": "lightgrey", "alpha": 0.5},
    ha="center",
    fontsize=10,
)

# Save the figure
output_pie_chart_path = output_dir_path / "group_membership_pie_chart.png"
plt.savefig(output_pie_chart_path)

plt.show()
plt.close()

# Create and display the summary table
# Add totals
group_labels_template.append("Total")
group_counts_for_pie.append(sum(group_counts_for_pie))
calculated_percentages.append(sum(calculated_percentages))
df_summary = pd.DataFrame(
    {
        "Category": group_labels_template,
        "Count": group_counts_for_pie,
        "Percentage": calculated_percentages,
    }
)
print(df_summary)

# %% [markdown]
# ## Section 2: Distribution of the SIC codes of the labelled set
#
# ### Finding 1 - Strong bias to Divisions 86, 87: 68.2%
#
# Division 86 is "Human Health Activities"
#
# Division 87 is "Residential Care"
#
# ### Implication for our work:
#
# A wider range of evaluation data is requested to remove the risks
# associated with bias, such as a difference performance in one division compared to another.
#


# %%
def plot_sic_code_histogram(
    df: pd.DataFrame,
    column_name: str,
    show_percent=False,
    filename_suffix: str = "",
    relabel_remap: Optional[dict] = None,
) -> None:
    """Generates and saves a histogram (bar plot) for the value counts of a SIC code column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the SIC code column to analyze.
        show_percent (bool): Whether to display percentages instead of raw counts.
        filename_suffix (str): Suffix to add to the plot filename.
        relabel_remap (dict, optional): A dictionary to remap x-axis
            tick labels for better readability.

    """
    output_dir = Path("/home/user/survey-assist-utils/notebooks")
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
        total_count = 100
        ylabel_text = "Percentage"
    else:
        counts = df[column_name].value_counts().nlargest(top_n)
        # Calculate the 'Others' category
        total_count = eval_data[column_name].value_counts().sum()
        ylabel_text = "Frequency (Count)"

    # Append 'Others' to counts
    counts["Others"] = total_count - counts.sum()
    # Calculate the percentage that top_n represents
    top_n_percentage = (counts.sum() - counts["Others"]) / total_count * 100

    if counts.empty:
        logger.warning("No data to plot for histogram of column '%s'.", column_name)
        plt.close()  # Close the empty figure
        return

    ax = sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette="viridis",
        dodge=False,
        legend=False,
    )
    ax.set_title(
        f"""Top {top_n} Most Frequent Codes in '{column_name}',
        representing {top_n_percentage:.1f}% of (Total Rows: {len(eval_data)})"""
    )
    if relabel_remap:
        new_labels = [
            relabel_remap.get(label.get_text(), label.get_text())
            for label in ax.get_xticklabels()
        ]
        ax.set_xticklabels(new_labels)

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
# ### Histogram of the top 10 2 digit codes (Division)

# %%

plot_sic_code_histogram(
    eval_data,
    column_name="SIC_Division",
    show_percent=True,
    filename_suffix="SIC_Division",
)

# %%
# Calculate membership of 86, "Human Health Activities"  and 87, "Residential Care" combined
print(
    f"""Fraction of the dataset being in Division 86 or 87:
    {100 * eval_data['SIC_Division'].isin(['86', '87']).sum() / len(eval_data):.1f}%"""
)

# %% [markdown]
# ### Histogram for unambiguous only:
#
# 5 digit codes by definition.

# %%
filtered_data = eval_data[eval_data["Unambiguous"]]
plot_sic_code_histogram(
    filtered_data,
    column_name="sic_ind_occ1",
    show_percent=True,
    filename_suffix="Unambiguous_only",
)

# %% [markdown]
# ### Histogram of the number of codes CCs applied at 5-digits:
#
# Distribution of number of possible SIC codes (uncodeable - 0 code, 1 code, 2 codes, 3 codes)
#
# relabel the 4 to 4+ to remove ambiguity

# %%
# relabel the 4 to 4+
relabel_remap_four = {"4": "4+"}
plot_sic_code_histogram(
    eval_data,
    column_name="num_answers",
    show_percent=True,
    filename_suffix="",
    relabel_remap=relabel_remap_four,
)

# %%
print(
    f"""Single answer given {100*(eval_data["num_answers"] == 1).sum() /
        len(eval_data["num_answers"]):.2f} %"""
)

# %% [markdown]
# ### Conclusion:
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
