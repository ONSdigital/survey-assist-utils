"""Notebook to read November test data and clean up SA initial&final codes.

It only needs to be run once, before other analysis notebooks.
Saves the cleaned data to a parquet file for further analysis.

Usage:
    1. Set the PREPROD_DATA_BUCKET in .env file to point to the correct data bucket.
    2. Run this notebook to output data for final code assignement (around line 100).
    3. Run the SIC classification pipeline in `sic-classification-utils` to get final codes.
    4. After getting the final codes, run this notebook again to merge final codes and
       assess codability gain/loss.

For testing purposes, you can set `out_dir` to a local folder to save the output files locally.
For production runs, set `out_dir` equal to `work_dir`.
"""

# pylint: disable=C0301,C0103,R0801
# %%
import dotenv
import pandas as pd

from survey_assist_utils.data_cleaning.prep_data import prep_model_codes
from survey_assist_utils.data_cleaning.sic_codes import (
    asses_codability_gain,
    get_clean_n_digit_codes,
    get_codability_level,
)

data_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET") or ""

# %%
work_dir = data_bucket + "analysis-interim-results"
out_dir = work_dir  # set to None to skip saving

# %%
# read data exported from firesore
eval_df = pd.read_parquet(
    work_dir + "/survey_results_reformatted_for_cc_evaluation.parquet"
)


# %%
def get_sa_initial_codes(row):
    """Get the initial SIC codes based on data format from SurveyAssist UI.

    Args:
        row: A row of the DataFrame containing SurveyAssist classification results.

    Returns:
        A set of cleaned initial SIC codes assigned by SurveyAssist.
    """
    if row["direct_lookup_classified"]:
        return get_clean_n_digit_codes({row["direct_lookup_assigned_code"]}, n=5)
    codes = [row["survey_assist_assigned_code"]]
    if not row["survey_assist_classified"]:
        codes = codes + [
            row[f"survey_assist_alt_candidate_code_{i}"] for i in range(1, 5)
        ]
    codes = {code for code in codes if pd.notna(code)}
    return get_clean_n_digit_codes(codes, n=5)


# %%
eval_df["sa_initial_codes"] = eval_df.apply(
    lambda row: get_sa_initial_codes(row)[0], axis=1
)
eval_df["sa_initial_codability_level"] = eval_df["sa_initial_codes"].apply(
    get_codability_level
)

print(eval_df["sa_initial_codability_level"].value_counts())

# %%
# prepare data for final coding pipeline (then run the script in `sic-classification-utils`)
pipe_df = (
    eval_df[
        [
            "unique_id",
            "job_title",
            "job_description",
            "org_description",
            "survey_assist_open_question",
            "survey_assist_open_question_response",
        ]
    ]
    .copy()
    .rename(
        columns={
            "job_title": "soc2020_job_title",
            "job_description": "soc2020_job_description",
            "org_description": "merged_industry_desc",
            "survey_assist_open_question": "followup_question",
            "survey_assist_open_question_response": "followup_answer",
        }
    )
    .fillna("")
)

pipe_df.to_parquet(
    out_dir + "/SA_final_code_tmp/sic_coding_pipeline_input.parquet", index=False
)

# %%
# after the final classification script run:
final_code_df = pd.read_parquet(out_dir + "/SA_final_code_tmp/STG7.parquet")
final_code_df = prep_model_codes(
    final_code_df,
    out_col="sa_final_codes_open_q",
    codes_col="final_code",
    alt_codes_col="alt_sic_candidates_final",
)
combined_df = eval_df.merge(final_code_df, on="unique_id", how="outer")
msk = combined_df["sa_initial_codability_level"] == "Sub-class (5-digits)"
combined_df.loc[msk, "sa_final_codes_open_q"] = combined_df.loc[msk, "sa_initial_codes"]
combined_df["sa_final_codability_level_open_q"] = combined_df[
    "sa_final_codes_open_q"
].apply(get_codability_level)

print(combined_df.head())
combined_df.groupby(
    ["sa_initial_codability_level", "sa_final_codability_level_open_q"]
).size().unstack(fill_value=0)


# %%
# assign final codability level for closed question responses
msk = combined_df["survey_assist_closed_question_response"].isna() | (
    combined_df["survey_assist_closed_question_response"] != "none of the above"
)
combined_df["sa_final_codability_level_closed_q"] = "Uncodable"
combined_df.loc[msk, "sa_final_codability_level_closed_q"] = "Sub-class (5-digits)"

print(combined_df.head())
combined_df.groupby(
    ["sa_initial_codability_level", "sa_final_codability_level_closed_q"]
).size().unstack(fill_value=0)


# %%
# assess codability gain
combined_df["sa_codability_gain_open_q"] = combined_df.apply(
    asses_codability_gain,
    axis=1,
    initial_level_col="sa_initial_codability_level",
    final_level_col="sa_final_codability_level_open_q",
)
combined_df["sa_codability_gain_closed_q"] = combined_df.apply(
    asses_codability_gain,
    axis=1,
    initial_level_col="sa_initial_codability_level",
    final_level_col="sa_final_codability_level_closed_q",
)


# %%
# get most likely sic section
def get_most_likely_section(
    row,
    columns_to_consider: tuple[str, ...] = (
        "sa_final_codes_open_q",  #'sa_final_codes_closed_q',
        "survey_assist_alt_candidate_code_1",
    ),
) -> str | None:
    """Get the most likely SIC section if only one section is present in the codes."""
    for col in columns_to_consider:
        codes = get_clean_n_digit_codes(row[col], n=0)
        if len(codes) == 1:
            return next(iter(codes))
    return None


combined_df["most_likely_sic_section"] = combined_df.apply(
    get_most_likely_section, axis=1
)

# %%
# save the pre-processed data with clean SA codes
if out_dir:
    combined_df.to_parquet(
        out_dir + "/evaluation_df_with_sa_clean_codes.parquet", index=False
    )

# %%
# prepare feedback file for Katrina
combined_df["survey_assist_closed_question_options"] = combined_df[
    "survey_assist_closed_question_option_1"
]
for i in range(2, 7):
    msk = ~combined_df[f"survey_assist_closed_question_option_{i}"].isna()
    combined_df.loc[msk, "survey_assist_closed_question_options"] += (
        "; " + combined_df.loc[msk, f"survey_assist_closed_question_option_{i}"]
    )

columns_for_katrina = [
    "unique_id",
    "user",
    "job_title",
    "job_description",
    "org_description",
    "direct_lookup_classified",
    "survey_assist_classified",
    "sa_initial_codability_level",
    "survey_assist_open_question",
    "survey_assist_open_question_response",
    "sa_final_codability_level_open_q",
    "survey_assist_closed_question_options",
    "survey_assist_closed_question_response",
    "sa_final_codability_level_closed_q",
    "most_likely_sic_section",
    "feedback_age_range",
    "feedback_survey_ease",
    "feedback_survey_relevance",
    "feedback_survey_comfort",
    "feedback_comments",
]

if out_dir:
    combined_df[columns_for_katrina].to_csv(
        out_dir + "/feedback_file_for_katrina.csv", index=False
    )
# %%
