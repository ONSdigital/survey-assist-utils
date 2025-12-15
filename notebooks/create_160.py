# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: survey-assist-utils-PWI-TvqZ-py3.12
#     language: python
#     name: python3
# ---

# %%
"""Get metrics from 160."""

# Load in the calc function:
import dotenv
import numpy as np
import pandas as pd
from google.cloud import storage

from survey_assist_utils.configs.column_config import ColumnConfig

# Load the prep function
from survey_assist_utils.data_cleaning.prep_data import (
    prep_clerical_codes,
    prep_model_codes,
)
from survey_assist_utils.evaluation.coder_alignment import (
    LabelAccuracy,
)
from survey_assist_utils.processing.flag_generator import FlagGenerator

MAX_SIC_CODE = 6
SUBSET_CHOICE = False
my_list = [
    "EV001778",
    "EV001794",
    "EV001803",
    "EV001828",
    "EV001840",
    "EV001843",
    "EV001846",
]

# %% [markdown]
# OO: One-to-One match on a subset where the true label as well as the model's
# label are not ambiguous.<br>
# OM: One-to-Many match on a subset where the true label is not ambiguous.
# (Is the true label in the model's shortlist?)<br>
# MO: Many-to-One match on a subset where the model is not ambiguous.
# (Is the model's label in the true label shortlist?)<br>
# MM: Many-to-Many match on the full set. (Is there any overlap between the true label's
# and model's shortlists?)

# %%
# Ger the data:
# Lets load some data to check each path

bucket_prefix = dotenv.get_key("../.env", "BUCKET_PREFIX")

# Initialize the client
client = storage.Client()

# Get the values into my dataset:
MODEL_PROMPT2_FILE = (
    f"{bucket_prefix}two_prompt_pipeline/2025_09_full_2k_gemini25/STG5.parquet"
)
prompt2_df = pd.read_parquet(MODEL_PROMPT2_FILE)

# load clerical data
CLERICAL_IT2_FILE = f"{bucket_prefix}original_datasets/DSC_Rep_Sample_IT2.csv"
CLERICAL_IT2_4PLUS_FILE = (
    f"{bucket_prefix}original_datasets/Codes_for_4_plus_DSC_Rep_Sample_IT2.csv"
)

cc_it2_df = pd.read_csv(CLERICAL_IT2_FILE)
cc_it2_4plus_df = pd.read_csv(CLERICAL_IT2_4PLUS_FILE)


# %%
# Expand the list of codes to multiple columns:
def expand_clerical_codes(df, column="clerical_codes", max_cols=10):
    """Tbd."""
    expanded_rows = []
    for _, row in df.iterrows():
        codes = list(row[column])  # Convert set to list
        new_row = row.drop(column).to_dict()
        # Fill up to max_cols with NaN if fewer codes
        for i in range(1, max_cols + 1):
            new_row[f"clerical_code_{i}"] = codes[i - 1] if i <= len(codes) else np.nan
        expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)


# %%
DIGITS = 5

clerical_codes_it2 = prep_clerical_codes(cc_it2_df, cc_it2_4plus_df, digits=DIGITS)

model_prompt2 = prep_model_codes(prompt2_df, digits=DIGITS, out_col="sa_initial_codes")

combined_dataframe_m2 = model_prompt2.merge(
    clerical_codes_it2, on="unique_id", how="inner"
)

# Preprocess codes:
clerical_codes_it2_expanded = expand_clerical_codes(clerical_codes_it2)


# %%
def subset_data(df, uid_list):
    """SUBSET THE DATA."""
    if SUBSET_CHOICE:
        df = df[df["unique_id"].isin(uid_list)]
    return df


prompt2_df = subset_data(prompt2_df, my_list)
cc_it2_df = subset_data(cc_it2_df, my_list)
cc_it2_4plus_df = subset_data(cc_it2_4plus_df, my_list)


# %%
def expand_sic_candidates(df, column="alt_sic_candidates", max_codes=7):
    """Expands codes."""

    def extract_codes(candidates):
        codes = [
            item.get("code")
            for item in candidates
            if isinstance(item, dict) and "code" in item
        ]
        return codes[:max_codes] if max_codes else codes

    # Apply extraction
    codes_series = df[column].apply(extract_codes)

    # Expand into columns
    codes_df = pd.DataFrame(codes_series.tolist(), index=df.index)
    codes_df.columns = [f"sic_code_{i+1}" for i in range(codes_df.shape[1])]

    return pd.concat([df.drop(columns=[column]), codes_df], axis=1)


# %%
# Join it2 onto Prompt 2 df:
expanded_prompt2 = expand_sic_candidates(prompt2_df)


# %%
# Prioritise 'initial _code' and overwrite the first sic_code with it.
expanded_prompt2["sic_code_1"] = np.where(
    expanded_prompt2["initial_code"].notna() & (expanded_prompt2["initial_code"] != ""),
    expanded_prompt2["initial_code"],
    expanded_prompt2["sic_code_1"],
)


# %%
combined_dataframe_mark = expanded_prompt2.merge(
    clerical_codes_it2_expanded, on="unique_id", how="inner"
)

# Set fake liklihoods, as we don't capture those in prompt2!

# Define the model_score_cols
model_score_cols = [f"likelihood_{i}" for i in range(1, MAX_SIC_CODE)]

# Set all likelihood columns to 0.9
for col in model_score_cols:
    combined_dataframe_mark[col] = 0.9


flag_generator = FlagGenerator()
combined_dataframe_mark = flag_generator.add_flags(combined_dataframe_mark)


# %%
# Add in initial code:
print(len(combined_dataframe_mark))
combined_dataframe_mark = subset_data(combined_dataframe_mark, my_list)


# %%

# To make these, we simply use the config:
# MM: Many-to-Many match on the full set. (Is there any overlap between the true
# label's and model's shortlists?)
config_MM = ColumnConfig(
    model_label_cols=[f"sic_code_{i}" for i in range(1, MAX_SIC_CODE)],
    model_score_cols=[f"likelihood_{i}" for i in range(1, MAX_SIC_CODE)],
    clerical_label_cols=[f"clerical_code_{i}" for i in range(1, 8)],
    id_col="unique_id",
    filter_unambiguous=False,
)

print(config_MM)

# Many LLM options to only one CC option
# OM: One-to-Many match on a subset where the true label is not ambiguous.
# (Is the true label in the model's shortlist?)<br>
# Hence, multiple Model answers, only one CC
config_OM = ColumnConfig(
    model_label_cols=[f"sic_code_{i}" for i in range(1, MAX_SIC_CODE)],
    model_score_cols=[f"likelihood_{i}" for i in range(1, MAX_SIC_CODE)],
    clerical_label_cols=[f"clerical_code_{i}" for i in range(1, 2)],
    id_col="unique_id",
    filter_unambiguous=True,
)
print(config_OM)

# MO: Many-to-One match on a subset where the model is not ambiguous.
# (Is the model's label in the true label shortlist?)<br>
# Hence multiple CC, and only one Model.
# NOTE! The differences here are that I don't check *IF* they are unambiguous,
# I just use the first one and ignore the rest.
config_MO = ColumnConfig(
    model_label_cols=[f"sic_code_{i}" for i in range(1, 2)],
    model_score_cols=[f"likelihood_{i}" for i in range(1, 2)],
    clerical_label_cols=[f"clerical_code_{i}" for i in range(1, 8)],
    id_col="unique_id",
    filter_unambiguous=False,
)
print(config_OM)


# OO: One-to-One match on a subset where the true label as well as the model's
# label are not ambiguous.<br>
config_OO = ColumnConfig(
    model_label_cols=[f"sic_code_{i}" for i in range(1, 2)],
    model_score_cols=[f"likelihood_{i}" for i in range(1, 2)],
    clerical_label_cols=[f"clerical_code_{i}" for i in range(1, 2)],
    id_col="unique_id",
    filter_unambiguous=True,
)
print(config_OO)


# %%
def do_tests(config, variable_name):
    """TBD."""
    analyzer_main = LabelAccuracy(df=combined_dataframe_mark, column_config=config)

    # Collect results in a list of dicts
    results = []

    # First: full match
    full_acc_stats = analyzer_main.get_accuracy(match_type="full", extended=True)
    uid_list_true = analyzer_main.df[analyzer_main.df["is_correct"]][
        "unique_id"
    ].tolist()
    uid_list_false = analyzer_main.df[not analyzer_main.df["is_correct"]][
        "unique_id"
    ].tolist()

    for label, metrics in full_acc_stats.items():
        if label != "uid":
            results.append(
                {
                    "variable_name": variable_name,
                    "match_type": "full",
                    "label": label,
                    "metrics": metrics,
                    "uid_list_true": uid_list_true,
                    "uid_list_false": uid_list_false,
                }
            )

    # Second: two-digit match
    two_digit_acc_stats = analyzer_main.get_accuracy(
        match_type="2-digit", extended=True
    )
    uid_list_true = analyzer_main.df[analyzer_main.df["is_correct"]][
        "unique_id"
    ].tolist()
    uid_list_false = analyzer_main.df[not analyzer_main.df["is_correct"]][
        "unique_id"
    ].tolist()

    for label, metrics in two_digit_acc_stats.items():
        if label != "uid":
            results.append(
                {
                    "variable_name": variable_name,
                    "match_type": "2-digit",
                    "label": label,
                    "metrics": metrics,
                    "uid_list_true": uid_list_true,
                    "uid_list_false": uid_list_false,
                }
            )

    return results


# %%
all_results = []

config_list = [config_MM, config_OM, config_MO, config_OO]

for this_config in config_list:
    variable_name_main = next(
        name for name, value in locals().items() if value is this_config
    )
    all_results.extend(do_tests(this_config, variable_name_main))

# Convert to DataFrame
results_df_main = pd.DataFrame(all_results)


# %%
results_df_main.to_csv("results_df_new.csv", index=False)

# %%
main_df = combined_dataframe_m2[["unique_id"]]


def build_uid_flags(results_df, combined_df, match_type="full", label="uid_all"):
    """Build a dataframe with flags for metrics, uid_list_true, and uid_list_false
    for each config, merged into combined_df.

    Parameters
    ----------
    results_df : pd.DataFrame
        Source dataframe with columns: variable_name, match_type, label,
        metrics, uid_list_true, uid_list_false.
    combined_df : pd.DataFrame
        Must contain 'unique_id' column.
    match_type : str
        Filter for match_type column.
    label : str
        Filter for label column.

    Returns:
    -------
    pd.DataFrame
        combined_df with additional columns for each config:
        - {config}_Considered
        - {config}_True
        - {config}_False
    """
    build_df = combined_df.copy()
    configs = results_df["variable_name"].unique()

    for cfg in configs:
        # Filter for this config
        subset = results_df[
            (results_df["variable_name"] == cfg)
            & (results_df["match_type"] == match_type)
            & (results_df["label"] == label)
        ][["metrics", "uid_list_true", "uid_list_false"]]

        if subset.empty:
            # Add empty columns if no data for this config
            build_df[f"{cfg}_Considered"] = False
            build_df[f"{cfg}_True"] = False
            build_df[f"{cfg}_False"] = False
            continue

        uids = build_df["unique_id"]

        # Flatten lists into sets
        metrics_set = {x for sublist in subset["metrics"] for x in sublist}
        true_set = {x for sublist in subset["uid_list_true"] for x in sublist}
        false_set = {x for sublist in subset["uid_list_false"] for x in sublist}

        # Add columns

        build_df[f"{cfg}_Considered"] = uids.isin(metrics_set)
        build_df[f"{cfg}_True"] = uids.isin(true_set)
        build_df[f"{cfg}_False"] = uids.isin(false_set)

    return build_df


# Usage:
all_results = combined_dataframe_m2[["unique_id"]]
main_df = build_uid_flags(results_df_main, all_results)
print(main_df.head())
# main_df.to_csv('main_df_new.csv', index = False)

# %%


def add_result_columns(df):
    """For each config in the dataframe, create a {config}_Result column.
    - True if {config}_True is True
    - False if {config}_False is True
    - "" otherwise.
    """
    # Identify configs by finding columns ending with "_True"
    configs = [col.replace("_True", "") for col in df.columns if col.endswith("_True")]

    result_df = df.copy()

    for cfg in configs:
        true_col = f"{cfg}_True"
        false_col = f"{cfg}_False"
        result_col = f"{cfg}_Result"
        cond_true = result_df[true_col].astype(bool)
        cond_false = result_df[false_col].astype(bool)

        result_df[result_col] = np.select(
            [cond_true, cond_false], [True, False], default=""
        )

    return result_df


# %%
main_df = add_result_columns(main_df)

# %%
# check:
print(main_df.filter(like="_True").apply(sum, axis=0))
print(main_df.filter(like="_False").apply(sum, axis=0))
print(main_df["config_MM_Result"].value_counts())
print(main_df["config_MO_Result"].value_counts())
print(main_df["config_OM_Result"].value_counts())
print(main_df["config_OO_Result"].value_counts())

# %%
main_df.to_csv("mark_value_counts.csv", index=False)

PATH = "manual_review_finished/metrics_comparison_samples/mark_value_counts.csv"
OUTPUT_FILE = f"{bucket_prefix}{PATH}"
main_df.to_csv(OUTPUT_FILE, index=False)
