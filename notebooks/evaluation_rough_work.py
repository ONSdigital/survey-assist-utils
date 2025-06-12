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
"""This script defines a suite of metric functions for evaluating classification and matching
performance between predicted and actual Standard Industrial Classification (SIC) codes in
a dataset.

1. metric_a_first_choice_rate_percent:
   - Calculates the percentage of exact matches between 'sic_ind_occ1' and 'chosen_sic_code'.

2. metric_b_division_agreement:
   - Computes the number and percentage of matches and mismatches between 'SIC_Division' and
    'code_2'.

3. metric_b2_n_digits:
   - Evaluates hierarchical agreement at different SIC code levels (2, 3, 4, and full code).

4. metric_c_rank_first_choice:
   - Determines the average rank position at which 'sic_ind_occ1' appears among five candidate
    SIC codes.

5. metric_c2_other_choice_found:
   - Checks whether 'candidate_1_sic_code' appears in any of the alternative SIC code choices
     ('sic_ind_occ1', 'sic_ind_occ2', 'sic_ind_occ3') and returns a frequency table of match
     ranks.

6. metric_d_unambiguous_agreement:
   - Measures agreement between 'sic_ind_occ1' and 'chosen_sic_code' for rows marked as
   'Unambiguous'.

7. metric_e_distances:
   - Merges the main dataset with a cosine similarity lookup table and computes group-level
   statistics
     (mean, standard deviation, and group size) for each 'sic_ind_occ1'.

These functions are designed to support evaluation of machine learning or rule-based classification
systems that assign SIC codes to entities based on textual or structured input.

Note:
- Assumes the presence of two CSV files: the main dataset and a cosine similarity lookup table.
"""


import numpy as np

# %%
import pandas as pd

# Add some helper columns

# Read the CSV file with all columns as strings
analysis_data = pd.read_csv("merged_df.csv", dtype=str)

# Define the substring lengths and corresponding column names
code_lengths = [2, 3, 4]
code_cols = [f"code_{n}" for n in code_lengths]
code_cc_cols = [f"code_cc_{n}" for n in code_lengths]

# Generate substrings for 'chosen_sic_code'
for n, make_col in zip(code_lengths, code_cols):
    analysis_data[make_col] = analysis_data["chosen_sic_code"].str[:n]

# Generate substrings for 'sic_ind_occ1'
for n, make_col in zip(code_lengths, code_cc_cols):
    analysis_data[make_col] = analysis_data["sic_ind_occ1"].str[:n]


print(analysis_data.shape)


# %%
def metric_a_first_choice_rate_percent(data_in):
    """Calculates the percentage of exact matches between the 'sic_ind_occ1' and 'chosen_sic_code'
    columns.

    This function compares two columns in a DataFrame to determine how often the values match.
    The result is rounded to one decimal place.

    Parameters:
    ----------
    analysis_data : pandas.DataFrame
        A DataFrame containing at least the columns 'sic_ind_occ1' and 'chosen_sic_code'.

    Returns:
    -------
    float
        The percentage of rows where 'sic_ind_occ1' matches 'chosen_sic_code',
        including missing values in the denominator.
    """
    # Ensure the relevant columns are treated as strings
    analysis_data["sic_ind_occ1"] = analysis_data["sic_ind_occ1"].astype(str)
    analysis_data["chosen_sic_code"] = data_in["chosen_sic_code"].astype(str)

    # Calculate total comparisons including NA in chosen_sic_code
    total = (
        (data_in["sic_ind_occ1"] == data_in["chosen_sic_code"]).sum()
        + (data_in["sic_ind_occ1"] != data_in["chosen_sic_code"]).sum()
        + data_in["chosen_sic_code"].isna().sum()
    )

    # Calculate matching values
    matching = (data_in["sic_ind_occ1"] == data_in["chosen_sic_code"]).sum()

    # Calculate percentage
    matching_percent = round(100 * matching / total, 1) if total > 0 else 0.0

    return matching_percent


# %%
def metric_b_division_agreement(data_in):
    """Evaluates the agreement between 'SIC_Division' and the first two digits
    of chosen_sic_code, which for simplicity has been extracted to code_2.
    It returns both the raw counts and the proportions of matches and mismatches
    relative to the total number of rows.

    Parameters:
    ----------
    data_in : pandas.DataFrame
        A DataFrame containing at least the columns 'SIC_Division' and 'code_2'.

    Returns:
    -------
    dict
        A dictionary with the following keys:
        - 'not_matched': Number of rows where the values differ.
        - 'not_matched_pc': Proportion of mismatches (float between 0 and 1).
        - 'matched': Number of rows where the values match.
        - 'matched_pc': Proportion of matches (float between 0 and 1).
        - 'total': Total number of rows evaluated.
    """
    # Ensure the relevant columns are treated as strings
    data_in["SIC_Division"] = data_in["SIC_Division"].astype(str)
    data_in["code_2"] = data_in["code_2"].astype(str)

    # Calculate matched and not matched counts
    matched = (data_in["SIC_Division"] == data_in["code_2"]).sum()
    not_matched = (data_in["SIC_Division"] != data_in["code_2"]).sum()
    total = len(data_in)

    # Calculate percentages
    matched_pc = matched / total if total > 0 else 0.0
    not_matched_pc = not_matched / total if total > 0 else 0.0

    # Create results as a dictionary
    results = {
        "not_matched": not_matched,
        "not_matched_pc": not_matched_pc,
        "matched": matched,
        "matched_pc": matched_pc,
        "total": total,
    }

    return results


# %%
def metric_b2_n_digits(data_in):
    """Evaluates hierarchical agreement between predicted and actual SIC codes at multiple levels
    of precision.

    This function compares the predicted SIC code ('chosen_sic_code') and the actual SIC code
    ('sic_ind_occ1') at increasing levels of granularity: 2-digit (Division), 3-digit, 4-digit,
    and full 5-digit codes.
    It returns the number of matches at each level, providing insight into how closely the
    predictions align with the actual codes across different classification depths.

    Column comparisons:
    - 'SIC_Division' vs. 'code_2': 2-digit level
    - 'code_cc_3' vs. 'code_3': 3-digit level
    - 'code_cc_4' vs. 'code_4': 4-digit level
    - 'sic_ind_occ1' vs. 'chosen_sic_code': full 5-digit level

    Parameters:
    ----------
    data_in : pandas.DataFrame
        A DataFrame containing the relevant columns for SIC code comparison at multiple levels.

    Returns:
    -------
    dict
        A dictionary with the following keys:
        - 'two': Number of matches at the 2-digit level.
        - 'three': Number of matches at the 3-digit level.
        - 'four': Number of matches at the 4-digit level.
        - 'five': Number of exact matches at the full 5-digit level.
        - 'total': Total number of rows evaluated.
    """
    # Ensure relevant columns are treated as strings
    columns_to_convert = [
        "SIC_Division",
        "code_2",
        "code_cc_3",
        "code_3",
        "code_cc_4",
        "code_4",
        "sic_ind_occ1",
        "chosen_sic_code",
    ]
    for col in columns_to_convert:
        data_in[col] = data_in[col].astype(str)

    # Calculate matches
    two = (data_in["SIC_Division"] == data_in["code_2"]).sum()
    three = (data_in["code_cc_3"] == data_in["code_3"]).sum()
    four = (data_in["code_cc_4"] == data_in["code_4"]).sum()
    five = (data_in["sic_ind_occ1"] == data_in["chosen_sic_code"]).sum()
    total = len(data_in)

    # Return results as a dictionary
    results = {"two": two, "three": three, "four": four, "five": five, "total": total}

    return results


# %%
def metric_c_rank_first_choice(data_in):
    """Calculates the average rank position at which the actual SIC code ('sic_ind_occ1')
    appears among the top five predicted candidate SIC codes.

    This function evaluates how well the model ranks the correct SIC code by checking
    its position among five candidate predictions: 'candidate_1_sic_code' through
    'candidate_5_sic_code'. It returns the mean rank of the correct code across all
    rows where 'sic_ind_occ1' is not missing. If the correct code is not found in the
    candidate list, the row is excluded from the average.

    Parameters:
    ----------
    data_in : pandas.DataFrame
        A DataFrame containing the actual SIC code ('sic_ind_occ1') and five candidate
        SIC code predictions ('candidate_1_sic_code' to 'candidate_5_sic_code').

    Returns:
    -------
    float
        The average rank (1 to 5) at which the correct SIC code appears among the
        candidates. Returns NaN if no matches are found.
    """
    # Define the candidate columns
    candidate_list = [
        "candidate_1_sic_code",
        "candidate_2_sic_code",
        "candidate_3_sic_code",
        "candidate_4_sic_code",
        "candidate_5_sic_code",
    ]

    # Filter out rows where 'sic_ind_occ1' is missing
    filtered_data = data_in[data_in["sic_ind_occ1"].notna()]

    # Function to find the rank of the match
    def find_match_rank(row):
        for i, col in enumerate(candidate_list, start=1):
            if row["sic_ind_occ1"] == row[col]:
                return i
        return np.nan

    # Apply the function to each row
    filtered_data["match_rank"] = filtered_data.apply(find_match_rank, axis=1)

    # Calculate and return the mean rank
    return filtered_data["match_rank"].mean()


# %%
def metric_c2_other_choice_found(data_in):
    """Evaluates whether the top predicted SIC code appears among any of the top three actual SIC
    codes and returns a frequency distribution of its rank position.

    This function checks if 'candidate_1_sic_code' (the model's top prediction) appears in
    any of the actual SIC code choices: 'sic_ind_occ1', 'sic_ind_occ2', or 'sic_ind_occ3'.
    It assigns a rank (1, 2, or 3) based on the position of the match.
    If no match is found, the rank is recorded as 0.

    Parameters:
    ----------
    data_in : pandas.DataFrame
        A DataFrame containing the columns:
        - 'candidate_1_sic_code': the model's top predicted SIC code
        - 'sic_ind_occ1', 'sic_ind_occ2', 'sic_ind_occ3': the top three actual SIC codes

    Returns:
    -------
    pandas.Series
        A frequency table (value counts) of match ranks:
        - 1: match with 'sic_ind_occ1'
        - 2: match with 'sic_ind_occ2'
        - 3: match with 'sic_ind_occ3'
        - 0: no match found
    """
    # Define the columns to check against
    candidate_list = ["sic_ind_occ1", "sic_ind_occ2", "sic_ind_occ3"]

    # Check if candidate_1_sic_code appears in any of the candidate columns
    data_in["match_any"] = data_in.apply(
        lambda row: row["candidate_1_sic_code"] in [row[col] for col in candidate_list],
        axis=1,
    )

    # Determine the match rank (position of the match)
    def get_match_rank(row):
        for i, col in enumerate(candidate_list, start=1):
            if row["candidate_1_sic_code"] == row[col]:
                return i
        return 0  # No match found

    data_in["match_rank"] = data_in.apply(get_match_rank, axis=1)

    # Generate frequency table of match ranks
    results = data_in["match_rank"].value_counts().sort_index()

    return results


# %%
def metric_d_unambiguous_agreement(data_in):
    """Calculates agreement statistics between predicted and actual SIC codes for unambiguous
    cases.

    This function filters the dataset to include only rows where the 'Unambiguous' flag is
    set to "True". It then compares the predicted SIC code ('chosen_sic_code') with the
    actual SIC code ('sic_ind_occ1') to compute:

    - The number of matches and mismatches
    - The total number of unambiguous cases
    - The percentage of matches and mismatches among unambiguous cases

    Parameters:
    ----------
    data_in : pandas.DataFrame
        A DataFrame containing the columns:
        - 'sic_ind_occ1': the actual SIC code
        - 'chosen_sic_code': the predicted SIC code
        - 'Unambiguous': a string column indicating whether the case is unambiguous ("True")

    Returns:
    -------
    dict
        A dictionary with the following keys:
        - 'matched': Number of rows where the predicted and actual SIC codes match
        - 'not_matched': Number of rows where they differ
        - 'total_unambiguous': Total number of unambiguous rows
        - 'matched_pc': Percentage of matches among unambiguous rows
        - 'not_matched_pc': Percentage of mismatches among unambiguous rows
    """
    # Ensure relevant columns are treated as strings
    data_in["sic_ind_occ1"] = data_in["sic_ind_occ1"].astype(str)
    data_in["chosen_sic_code"] = data_in["chosen_sic_code"].astype(str)
    data_in["Unambiguous"] = data_in["Unambiguous"].astype(str)

    # Filter rows where Unambiguous is "True"
    unambiguous_data = data_in[data_in["Unambiguous"] == "True"]

    # Calculate matched and not matched counts
    matched = (
        unambiguous_data["sic_ind_occ1"] == unambiguous_data["chosen_sic_code"]
    ).sum()
    not_matched = (
        unambiguous_data["sic_ind_occ1"] != unambiguous_data["chosen_sic_code"]
    ).sum()
    total_unambiguous = len(unambiguous_data)

    # Calculate percentages
    matched_pc = (
        round(100 * matched / total_unambiguous) if total_unambiguous > 0 else 0
    )
    not_matched_pc = (
        round(100 * not_matched / total_unambiguous) if total_unambiguous > 0 else 0
    )

    # Return results as a dictionary
    results = {
        "not_matched": not_matched,
        "matched": matched,
        "total_unambiguous": total_unambiguous,
        "not_matched_pc": not_matched_pc,
        "matched_pc": matched_pc,
    }

    return results


# %%
def metric_e_distances(data_in, distance_file="distance_measures.csv"):
    """Merges cosine similarity scores between predicted and actual SIC codes into the main dataset
    and computes group-level statistics for each actual SIC code.

    This function loads a lookup table containing cosine similarity values between SIC code pairs,
    which were precomputed using embeddings from a language model (LLM). It merges these similarity
    scores with the main dataset based on the actual SIC code ('sic_ind_occ1') and the predicted
    SIC code ('chosen_sic_code').

    After merging, it calculates the following statistics for each group of rows sharing the same
    'sic_ind_occ1':
    - Mean cosine similarity
    - Standard deviation of cosine similarity
    - Group size (number of rows per SIC code)

    Parameters:
    ----------
    data_in : pandas.DataFrame
        The main dataset containing at least the following columns:
        - 'unique_id'
        - 'sic_ind_occ1': actual SIC code
        - 'chosen_sic_code': predicted SIC code
        - 'num_answers'
        - 'candidate_1_likelihood'

    distance_file : str, optional
        Path to the CSV file containing cosine similarity scores between SIC code pairs.
        Default is "distance_measures.csv".

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing the merged data along with the following additional columns:
        - 'cosine_similarity': similarity score between actual and predicted SIC codes
        - 'mean_value': mean cosine similarity for each 'sic_ind_occ1' group
        - 'SD': standard deviation of cosine similarity for each group
        - 'group_size': number of rows in each group
    """
    # Load the lookup table with specified column types
    col_types = {
        "X": "Int64",
        "CC_sic": "string",
        "SA_sic": "string",
        "CC_Activity": "string",
        "Activity": "string",
        "cosine_similarity": "float",
    }
    lookup = pd.read_csv(distance_file, dtype=col_types)

    # Rename columns to match the main dataset
    lookup.rename(
        columns={"CC_sic": "sic_ind_occ1", "SA_sic": "chosen_sic_code"}, inplace=True
    )

    # Keep only relevant columns
    lookup = lookup[["sic_ind_occ1", "chosen_sic_code", "cosine_similarity"]]

    # Reduce the initial data to relevant columns
    check_cols = [
        "unique_id",
        "sic_ind_occ1",
        "num_answers",
        "chosen_sic_code",
        "candidate_1_likelihood",
    ]
    pairs = data_in[check_cols]

    # Merge the datasets on 'sic_ind_occ1' and 'chosen_sic_code'
    pairs_with_dist = pd.merge(
        pairs, lookup, on=["sic_ind_occ1", "chosen_sic_code"], how="left"
    )

    # Compute group-wise statistics
    pairs_with_dist["mean_value"] = pairs_with_dist.groupby("sic_ind_occ1")[
        "cosine_similarity"
    ].transform("mean")
    pairs_with_dist["SD"] = pairs_with_dist.groupby("sic_ind_occ1")[
        "cosine_similarity"
    ].transform("std")
    pairs_with_dist["group_size"] = pairs_with_dist.groupby("sic_ind_occ1")[
        "cosine_similarity"
    ].transform("count")

    return pairs_with_dist


# %%

metric_a_first_choice_rate_result = metric_a_first_choice_rate_percent(analysis_data)
print(metric_a_first_choice_rate_result)

# %%
metric_b_division_agreement_result = metric_b_division_agreement(analysis_data)
print(metric_b_division_agreement_result)

# %%
metric_b2_n_digits_result = metric_b2_n_digits(analysis_data)
print(metric_b2_n_digits_result)

# %%
metric_c_rank_first_choice_restul = metric_c_rank_first_choice(analysis_data)
print(metric_c_rank_first_choice_restul)

# %%

metric_c2_other_choice_found_result = metric_c2_other_choice_found(analysis_data)
print(metric_c2_other_choice_found_result)

# %%

metric_d_unambiguous_agreement_result = metric_d_unambiguous_agreement(analysis_data)
print(metric_d_unambiguous_agreement_result)

# %%
cosine_table = metric_e_distances(analysis_data)
print(cosine_table)
