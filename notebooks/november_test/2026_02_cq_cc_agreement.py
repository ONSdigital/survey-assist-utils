# %%
"""Work in progess.

Initial analysis of survey responses, focusing on Closed Follow up questions.

Create .env file with bucket variables, such as EVALUATION_BUCKET = "gs://<bucket-name>/<folder>/",
and ANALYSIS_BUCKET similarly.
"""

# %%
# pylint: disable=C0103, C0116, C0301, C0114, R0801
# ruff: noqa: PLR2004

# %%
import os

import dotenv
import numpy as np
from helper_load_data import load_data
from scipy.stats import chi2_contingency

# %%
data_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET") or ""
work_dir = data_bucket + "analysis-interim-results"
out_dir = (
    "data/figures/"  # needs local folder unfortunately, set to None to skip saving
)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

full_data = load_data(work_dir)

# %%
evaluation_bucket = dotenv.get_key(".env", "EVALUATION_BUCKET")
analysis_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET")
if not evaluation_bucket:
    raise ValueError("EVALUATION_BUCKET not found in .env file. Please set it.")
if not analysis_bucket:
    raise ValueError("PREPROD_DATA_BUCKET not found in .env file. Please set it.")


# %%
closed_question_data = full_data[
    ~full_data["survey_assist_closed_question_response"].isna()
]

# %%
question_column = "do_you_think_it_is_possible_to_get_a_single_5-digit_code_with_a_single,_open_question,_based_on_the_initial_tlfs_responses?"

# %%
closed_question_selected_cols = closed_question_data[
    [
        "sa_initial_codes",
        "clerical_code_final",
        "cc_initial_codes",
        "sa_final_codes_closed_q",
        question_column,
    ]
]

# %%
closed_question_data[question_column].value_counts()

# %%
closed_question_data[question_column] = closed_question_data[question_column].apply(
    lambda x: x.lower() if isinstance(x, str) else "-9"
)
closed_question_data[question_column] = closed_question_data[question_column].apply(
    lambda x: x if len(x) < 3 else x[0]
)

# %%
closed_question_data[question_column].value_counts()

# %% [markdown]
# 122 CCs did not provide their opinion whether they think it is possible to get a single SIC code based on the initial TLFS responses
#
# check if there is a correlation between CCs saying "no" and respondents selecting "None of the above".

# %%
cc_opinion_given = closed_question_data[closed_question_data[question_column] != "-9"]

# %%
nota_y = cc_opinion_given[cc_opinion_given["sa_final_codes_closed_q"].str.len() == 0][
    question_column
].value_counts()["y"]
nota_n = cc_opinion_given[cc_opinion_given["sa_final_codes_closed_q"].str.len() == 0][
    question_column
].value_counts()["n"]

# %%
selected_y = cc_opinion_given[
    cc_opinion_given["sa_final_codes_closed_q"].str.len() > 0
][question_column].value_counts()["y"]
selected_n = cc_opinion_given[
    cc_opinion_given["sa_final_codes_closed_q"].str.len() > 0
][question_column].value_counts()["n"]

# %%
print(nota_n, selected_n)
print(nota_y, selected_y)

# %% [markdown]
# When looking at rows, where CC provided their opinion (yes or no), there is 86 respondents who selected NOTA.

# %% [markdown]
# When looking at rows, where CC provided their opinion (yes or no), there is 355 respondents who selected one of the answers.

# %% [markdown]
# null: there is no differenece between the CCs opinion and the respondent selecting NOTA.

# %%
row_no = [nota_n, selected_n]
row_yes = [nota_y, selected_y]
contingency_table = [row_no, row_yes]

# %%
print(contingency_table)

# %%
chi_square_p = chi2_contingency(contingency_table).pvalue

# %%
print(chi_square_p)

# %% [markdown]
# With p-value at 0.84, we reject null hypothesis, suggesting that there is no evidence of a relationship between CCs opinion and respondent selecting NOTA.

# %%
print((selected_n + selected_y) / (selected_n + selected_y + nota_n + nota_y) * 100)

# %% [markdown]
# 80.5% of respondents selected one of the codes.

# %%
final_codability_sa_cc = (
    full_data[full_data.survey_assist_open_question.notna()]
    .groupby(["cc_initial_codability_level", "sa_final_codability_level_closed_q"])
    .size()
    .unstack()
)

# %% [markdown]
# Comparing CCs' final codability with SA final codability.

# %%
print(final_codability_sa_cc)

# %%
cl = [
    final_codability_sa_cc["Sub-class (5-digits)"]["Class (4-digits)"],
    final_codability_sa_cc["Uncodable"]["Class (4-digits)"],
]
di = [
    final_codability_sa_cc["Sub-class (5-digits)"]["Division (2-digits)"],
    final_codability_sa_cc["Uncodable"]["Division (2-digits)"],
]
gr = [
    final_codability_sa_cc["Sub-class (5-digits)"]["Group (3-digits)"],
    final_codability_sa_cc["Uncodable"]["Group (3-digits)"],
]
se = [
    final_codability_sa_cc["Sub-class (5-digits)"]["Section (letter)"],
    final_codability_sa_cc["Uncodable"]["Section (letter)"],
]
su = [
    final_codability_sa_cc["Sub-class (5-digits)"]["Sub-class (5-digits)"],
    final_codability_sa_cc["Uncodable"]["Sub-class (5-digits)"],
]
un = [
    final_codability_sa_cc["Sub-class (5-digits)"]["Uncodable"],
    final_codability_sa_cc["Uncodable"]["Uncodable"],
]

# %%
table = np.array([cl, di, gr, se, su, un])

# %% [markdown]
# null hypothesis: there is no differenece between likelihoods of success between rows (there is no difference between cc and sa successes).

# %%
p_value = chi2_contingency(table).pvalue
expected = chi2_contingency(table).expected_freq

# %%
print(p_value)

# %% [markdown]
# The p-value is very low (4.7852e-20). This allows to reject the null hypothesis. There is a differnece between successes found by cc and sa.

# %%
sr_cl = round(cl[0] / sum(cl) * 100, 2)
sr_di = round(di[0] / sum(di) * 100, 2)
sr_gr = round(gr[0] / sum(gr) * 100, 2)
sr_se = round(se[0] / sum(se) * 100, 2)
sr_su = round(su[0] / sum(su) * 100, 2)
sr_un = round(un[0] / sum(un) * 100, 2)

# %%
print(sr_cl, sr_di, sr_gr, sr_se, sr_su, sr_un)

# %%
sr_all = round(
    final_codability_sa_cc["Sub-class (5-digits)"].sum()
    / (
        final_codability_sa_cc["Sub-class (5-digits)"].sum()
        + final_codability_sa_cc["Uncodable"].sum()
    )
    * 100,
    2,
)

# %%
print(sr_all)

# %%
obs_sum = table.sum()

# %%
rows_total = table.sum(axis=1, keepdims=True)

# %%
cols_total = table.sum(axis=0, keepdims=True)

# %%
variance = expected * (1 - rows_total / obs_sum) * (1 - cols_total / obs_sum)

# %%
adj_residuals = (table - expected) / variance**0.5

# %%
print(adj_residuals)

# %%
print(adj_residuals > 1.96)

# %% [markdown]
# ## CQ - CC disagreement

# %%
method = "cc"
msk = (
    full_data[f"{method}_final_codability_level_open_q"] == "Sub-class (5-digits)"
) & ~full_data["survey_assist_open_question"].isna()
full_data[f"{method}_final_codes_open_q_within_offered_options"] = full_data.apply(
    lambda row: row[f"{method}_final_codes_open_q"].issubset(row["sa_initial_codes"]),
    axis=1,
)
full_data[f"{method}_final_codes_open_q_vs_selected_by_user_in_closed"] = (
    full_data.apply(
        lambda row: (
            "none of the above"
            if len(row["sa_final_codes_closed_q"]) == 0
            else (
                "same code selected"
                if row[f"{method}_final_codes_open_q"].issubset(
                    row["sa_final_codes_closed_q"]
                )
                else "different selected"
            )
        ),
        axis=1,
    )
)
full_data[msk].groupby(
    [
        f"{method}_final_codes_open_q_vs_selected_by_user_in_closed",
        f"{method}_final_codes_open_q_within_offered_options",
    ]
).size().unstack(fill_value=0)

# %%
full_data["both_final_codes_open_q_vs_selected_by_user_in_closed"] = full_data.apply(
    lambda row: (
        "none of the above"
        if len(row["sa_final_codes_closed_q"]) == 0
        else (
            "same code selected"
            if row["cc_final_codes_open_q"].issubset(row["sa_final_codes_closed_q"])
            and row["sa_final_codes_open_q"].issubset(row["sa_final_codes_closed_q"])
            else (
                "sa_code selected"
                if row["sa_final_codes_open_q"].issubset(row["sa_final_codes_closed_q"])
                else (
                    "cc_code selected"
                    if row["cc_final_codes_open_q"].issubset(
                        row["sa_final_codes_closed_q"]
                    )
                    else "different selected"
                )
            )
        )
    ),
    axis=1,
)

# %%
cc_final_5dig = full_data[
    full_data["cc_final_codability_level_open_q"] == "Sub-class (5-digits)"
]

# %%
cc_5dig_open_question = cc_final_5dig[
    ~(cc_final_5dig["survey_assist_open_question"].isna())
]

# %%
columns_to_investigate = [
    "job_title",
    "job_description",
    "org_description",
    "survey_assist_open_question",
    "survey_assist_open_question_response",
    "survey_assist_closed_question_response",
    "sa_initial_codes",
    "sa_initial_codability_level",
    "sa_final_codes_open_q",
    "sa_final_codability_level_open_q",
    "sa_final_codability_level_closed_q",
    "sa_codability_gain_closed_q",
    "most_likely_sic_section",
    "SIC Section",
    "clerical_code_initial",
    "clerical_code_final",
    "cc_initial_codes",
    "cc_initial_codability_level",
    "cc_final_codes_open_q",
    "cc_final_codability_level_open_q",
    "sa_final_codes_closed_q",
    "cc_final_codes_open_q_vs_selected_by_user_in_closed",
]

# %%
cc_5dig_only_columns = cc_5dig_open_question[
    cc_5dig_open_question["cc_final_codes_open_q_vs_selected_by_user_in_closed"]
    == "different selected"
][columns_to_investigate].copy()

# %%
cc_resp_disagreemnet = cc_5dig_only_columns[
    cc_5dig_only_columns.apply(
        lambda row: row["cc_final_codes_open_q"].issubset(row["sa_initial_codes"]),
        axis=1,
    )
]

# %%
print(cc_resp_disagreemnet["cc_final_codes_open_q"].value_counts())

# %%
columns_to_display = [
    "job_title",
    "job_description",
    "org_description",
    "survey_assist_open_question",
    "survey_assist_open_question_response",
    "survey_assist_closed_question_response",
    "sa_final_codes_closed_q",
    "cc_final_codes_open_q",
    "SIC Section",
]

# %%
# conditions for filtering row for analysis, provide codes to investigate the disagreement between
cc_condition = cc_resp_disagreemnet["cc_final_codes_open_q"] == {"88990"}
sa_condition = cc_resp_disagreemnet["sa_final_codes_closed_q"] == {"86900"}

# %%
cc_resp_disagreemnet[cc_condition & sa_condition][columns_to_display].reset_index(
    drop=True
)
