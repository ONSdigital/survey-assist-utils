# %%
"""Work in progess.

Initial analysis of survey responses, focusing on resons for selecting
"None of the above" as response to Closed Follow up questions.

Create .env file with bucket variables, such as
PREPROD_DATA_BUCKET = "gs://<bucket-name>/<folder>/".
"""

# %%
# pylint: disable=C0103, C0116, C0301, C0114, R0801
# ruff: noqa: PLR2004

# %%
import dotenv
import numpy as np
import pandas as pd
from scipy.stats import (
    chi2_contingency,
    contingency,
    fisher_exact,
    mannwhitneyu,
    shapiro,
    ttest_ind,
)

from survey_assist_utils.data_cleaning.sic_codes import get_clean_n_digit_codes

# %%
preprod_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET")
if not preprod_bucket:
    raise ValueError("PREPROD_DATA_BUCKET not found in .env file. Please set it.")

# %% [markdown]
# ## None of the above (NOTA)
#
# Find reasons why respondent selected "None of the above" when presented with closed question.

# %%
data = pd.read_parquet(
    f"{preprod_bucket}analysis-interim-results/evaluation_df_with_sa_clean_codes.parquet"
)

# %%
data_closed_question = data[data["survey_assist_closed_question_response"].notna()]
data_nota = data_closed_question[
    data_closed_question["survey_assist_closed_question_response"]
    == "none of the above"
].copy()
data_not_nota = data_closed_question[
    data_closed_question["survey_assist_closed_question_response"]
    != "none of the above"
].copy()

# %%
print(f"Repsonses with code selected: {len(data_not_nota)}")
print(f'Repsonses with "None of the above" selected: {len(data_nota)}')

# %%
# check against sections


def possible_sections(response_row: pd.Series, code_digits: int = 0):
    """Checks the count of options presented to the user. Counts the number of unique sections of the presented options, and adds to the dataframe.

    Args:
        response_row (pd.Series): a row containing survey results.
        code_digits (int): number of SIC code digits
    """
    column_placeholder = "survey_assist_alt_candidate_code_"
    unique_section_list_by_row = []

    codes = 0
    row_codes = []
    k = 1
    while k < 6:
        column_name = column_placeholder + str(k)
        if response_row[column_name] is not None:
            row_codes.append(response_row[column_name])
            codes += 1
        k += 1

    unique_section_list_by_row = list(
        get_clean_n_digit_codes(row_codes, n=code_digits)[0]
    )

    return unique_section_list_by_row, codes


# %%
# Data preparation: add columns 'unique_sections' and 'codes_count'
unique_sections_codes_count_not_nota = pd.DataFrame(
    data_not_nota.apply(possible_sections, axis=1, args=(2,)).to_list(),
    index=data_not_nota.index,
    columns=["unique_sections", "codes_count"],
)
unique_sections_codes_count_nota = pd.DataFrame(
    data_nota.apply(possible_sections, axis=1, args=(2,)).to_list(),
    index=data_nota.index,
    columns=["unique_sections", "codes_count"],
)

data_not_nota.loc[:, ["unique_sections", "codes_count"]] = (
    unique_sections_codes_count_not_nota
)
data_nota.loc[:, ["unique_sections", "codes_count"]] = unique_sections_codes_count_nota

# %% [markdown]
# ## Effect of Number of Options

# %%
codes_count = {}
for i in range(2, 6):
    success = len(data_not_nota[data_not_nota["codes_count"] == i])
    failure = len(data_nota[data_nota["codes_count"] == i])
    codes_count[i] = [success]
    codes_count[i].append(failure)
df_codes_count = pd.DataFrame(codes_count, index=["success", "failure"]).T

# %%
pvalue_codes_count = chi2_contingency(df_codes_count).pvalue

# %%
print(pvalue_codes_count)

# %% [markdown]
# The p-value for the number of codes presented vs success (selecting one of the options, not NOTA) is 0.57, which means there is no evidence that the number of options influenced the decision.

# %% [markdown]
# ## Frequency of the word “other” in phrasing of options presented
#
# Check if the ratio of options with "other" in the description had impact on the selected response.
#
# Null hypothesis: There is no relationship between the frequency of the word "other" in the description presented to the respondent and the respondent's final selection (one of the descriptions or NOTA).


# %%
def count_others(response_row: pd.Series) -> int:
    """Counts the number of option descriptions that contains word 'other'.

    Args:
        response_row (pd.Series): a row containing survey assist closed question options.

    Returns:
        other_count (int): count of possible options containing word "other"
    """
    column_placeholder = "survey_assist_closed_question_option_"
    k = 1
    other_count = 0

    while k < 7:
        column_name = column_placeholder + str(k)
        row_string = response_row[column_name]

        if row_string.lower() == "none of the above":
            # options_presented = k
            k = 7

        else:
            if "other" in row_string.lower():
                other_count += 1
            k += 1

    return other_count


# %%
options_with_other_not_nota = pd.DataFrame(
    data_not_nota.apply(count_others, axis=1).to_list(),
    index=data_not_nota.index,
    columns=["options_with_other"],
)
options_with_other_nota = pd.DataFrame(
    data_nota.apply(count_others, axis=1).to_list(),
    index=data_nota.index,
    columns=["options_with_other"],
)

data_not_nota.loc[:, ["options_with_other"]] = options_with_other_not_nota
data_nota.loc[:, ["options_with_other"]] = options_with_other_nota

# %%
ratio_nota = data_nota["options_with_other"] / data_nota["codes_count"]
ratio_not_nota = data_not_nota["options_with_other"] / data_not_nota["codes_count"]


data_nota.loc[:, "other_ratio"] = ratio_nota
data_not_nota.loc[:, "other_ratio"] = ratio_not_nota

# %%
pvalue_other_ratios = ttest_ind(
    data_nota["other_ratio"], data_not_nota["other_ratio"], equal_var=False
).pvalue

# %%
print(pvalue_other_ratios)

# %% [markdown]
# The p-value (0.047) is very close to the alpha 0.05. Check for normality of the data using Shapiro-Wilk test.

# %%
pvalue_shapiro_nota_other = shapiro(data_nota["other_ratio"]).pvalue
pvalue_shapiro_not_nota_other = shapiro(data_not_nota["other_ratio"]).pvalue

# %%
print(pvalue_shapiro_not_nota_other)
print(pvalue_shapiro_nota_other)

# %% [markdown]
# The data is not normally distributed (p values from Shapiro-Wilk tests for nota and not nota data subsets are close to 0). Use Mann-Whitney U test.

# %%
p_value_whitney_other = mannwhitneyu(
    data_nota["other_ratio"], data_not_nota["other_ratio"], alternative="greater"
).pvalue

# %%
print(p_value_whitney_other)

# %% [markdown]
# Again, the p-value obtained from Mann-Whitney test (0.021) is close but below to 0.05. This allows to reject the hypothesis - there is a difference.

# %%
print(
    f"Average percentage of options containing 'other' in the options presented when NOTA selected: {data_nota['other_ratio'].mean()}"
)
print(
    f"Average percentage of options containing 'other' in the options presented when NOTA not selected: {data_not_nota['other_ratio'].mean()}"
)

# %% [markdown]
# The word 'other' appeared more often when NOTA was selected (25% v 20%). The Mann-Whitney test p value confirms that this difference is statistically significant, i.e. the more often the word 'other' appears in the descriptions presented, respondents tend to select NOTA.

# %% [markdown]
# ## Options from one or multiple Sections

# %% [markdown]
#
# Calculate the **odds ratio** for sections.
#
# ||NOTA selected|Othar than NOTA|
# |---|---|---|
# |options from one section|a|b|
# |options from multiple sections|c|d|
#
# OR = (a * d) / (b * c)

# %%
nota_sections_count = data_nota["unique_sections"].str.len()
not_nota_sections_count = data_not_nota["unique_sections"].str.len()

# %%
nota_one_section = (nota_sections_count == 1).sum()
nota_many_section = (nota_sections_count > 1).sum()

not_nota_one_section = (not_nota_sections_count == 1).sum()
not_nota_many_section = (not_nota_sections_count > 1).sum()

# %%
row1 = [nota_one_section, not_nota_one_section]
row2 = [nota_many_section, not_nota_many_section]

OR = contingency.odds_ratio([row1, row2])
print(OR.statistic)

# %%
print(row1)
print(row2)

# %%
CI = OR.confidence_interval(confidence_level=0.95)
print(CI)

# %% [markdown]
# Odds Ratio <1, with Confidence Interval (0.33, 1.14). This suggests the difference between groups (respondents who selected NOTA and those who did not select NOTA) is not statistically significant.

# %%
# Success rate with one v multiple sections

same_section = not_nota_one_section / (nota_one_section + not_nota_one_section)
multiple_section = not_nota_many_section / (nota_many_section + not_nota_many_section)

# %%
print(f"Success rate when options presented from the same section: {same_section}")
print(f"Success rate when options presented from multiple sections: {multiple_section}")

# %%
contingency_table = np.array(
    [
        [not_nota_one_section, nota_one_section],
        [not_nota_many_section, nota_many_section],
    ]
)

p_value_success_rate = chi2_contingency(contingency_table).pvalue

# %%
print(p_value_success_rate)

# %% [markdown]
# The p-value for success rate is 0.15. There is no statistical significance between number of sections the presented options came from.

# %% [markdown]
# ## Options from one or multiple Divisions

# %%
# prepare division unique codes

# data_not_nota[["unique_divisions", "codes_count_division"]] = pd.DataFrame(
#     data_not_nota.apply(possible_sections, axis=1, args=(2,)).to_list(),
#     index=data_not_nota.index,
# )
# data_nota[["unique_divisions", "codes_count_division"]] = pd.DataFrame(
#     data_nota.apply(possible_sections, axis=1, args=(2,)).to_list(),
#     index=data_nota.index,
# )


divisions_codes_count_nota = pd.DataFrame(
    data_nota.apply(possible_sections, axis=1, args=(2,)).to_list(),
    index=data_nota.index,
    columns=["unique_divisions", "codes_count_division"],
)
divisions_codes_count_not_nota = pd.DataFrame(
    data_not_nota.apply(possible_sections, axis=1, args=(2,)).to_list(),
    index=data_not_nota.index,
    columns=["unique_divisions", "codes_count_division"],
)

data_nota.loc[:, ["unique_divisions", "codes_count_division"]] = (
    divisions_codes_count_nota
)
data_not_nota.loc[:, ["unique_divisions", "codes_count_division"]] = (
    divisions_codes_count_not_nota
)

# %%
nota_divisions_count = data_nota["unique_divisions"].str.len()
not_nota_divisions_count = data_not_nota["unique_divisions"].str.len()

# %%
nota_one_division = (nota_divisions_count == 1).sum()
nota_many_division = (nota_divisions_count > 1).sum()

not_nota_one_division = (not_nota_divisions_count == 1).sum()
not_nota_many_division = (not_nota_divisions_count > 1).sum()

# %%
row1_division = [nota_one_division, not_nota_one_division]
row2_division = [nota_many_division, not_nota_many_division]

OR_division = contingency.odds_ratio([row1_division, row2_division])
print(OR_division.statistic)

# %%
print(row1_division)
print(row2_division)

# %%
CI_division = OR_division.confidence_interval(confidence_level=0.95)
print(CI_division)

# %%
success_one_division = not_nota_one_division / (
    nota_one_division + not_nota_one_division
)
success_multiple_division = not_nota_many_division / (
    nota_many_division + not_nota_many_division
)

# %%
print(round(success_one_division * 100, 1))
print(round(success_multiple_division * 100, 1))

# %%
# Fisher's exact test

fisher_array = np.array([row1_division, row2_division])
p_value_fishers = fisher_exact(fisher_array).pvalue

# %%
print(p_value_fishers)

# %% [markdown]
# ## Effect of time spent on the survey
#
# Check if respondents just went with NOTA, to finish the survey sooner. Use surveys that didn't get asked closed question as a baseline.

# %%
time_data_nota = data[
    data["survey_assist_closed_question_response"] == "none of the above"
].copy()
time_data_not = data[
    data["survey_assist_closed_question_response"] != "none of the above"
].copy()
time_data_not_nota = time_data_not[
    time_data_not["survey_assist_closed_question_response"].notna()
].copy()
time_data_no_closed_question = data[
    data["survey_assist_closed_question_response"].isna()
].copy()

# %%
time_nota = pd.Timedelta(0)
for option in range(len(time_data_nota)):
    time_nota += (
        time_data_nota["time_end"].iloc[option]
        - time_data_nota["time_start"].iloc[option]
    )
print(
    f"Average time for finishing a survey when NOTA is selected {time_nota / len(time_data_nota)}"
)

# %%
time_not_nota = pd.Timedelta(0)
for option in range(len(time_data_not_nota)):
    time_not_nota += (
        time_data_not_nota["time_end"].iloc[option]
        - time_data_not_nota["time_start"].iloc[option]
    )
print(
    f"Average time for finishing a survey when NOTA is not selected {time_not_nota / len(time_data_not_nota)}"
)

# %%
time_no_closed_question = pd.Timedelta(0)
for option in range(len(time_data_no_closed_question)):
    time_no_closed_question += (
        time_data_no_closed_question["time_end"].iloc[option]
        - time_data_no_closed_question["time_start"].iloc[option]
    )
print(
    f"Average time for finishing a survey when no closed question is asked {time_no_closed_question / len(time_data_no_closed_question)}"
)


# %%
def get_duration_seconds(df: pd.DataFrame) -> pd.Timedelta:
    """Calculate time differennce between start and end of the survey.

    Args:
        df: dataframe with 'time_start' and 'time_end' columns

    Return: total time in seconds
    """
    start = df["time_start"]
    end = df["time_end"]
    return (end - start).dt.total_seconds()  # time in seconds


# %%
# get response time for nota / not nota / no question
time_nota = get_duration_seconds(time_data_nota)
time_not_nota = get_duration_seconds(time_data_not_nota)
time_no_closed_question = get_duration_seconds(time_data_no_closed_question)

time_data_nota.loc[:, "response_time"] = time_nota
time_data_not_nota.loc[:, "response_time"] = time_not_nota
time_data_no_closed_question.loc[:, "response_time"] = time_no_closed_question

# %%
print(
    "Min time no quesiton asked:",
    round(time_data_no_closed_question["response_time"].min()),
    "seconds",
)
print(
    "Max time no question asked:",
    round(time_data_no_closed_question["response_time"].max()),
    "seconds",
)
print(
    "Median time no question asked:",
    round(time_data_no_closed_question["response_time"].median()),
    "seconds",
)

# %%
print("Min time NOTA:", round(time_data_nota["response_time"].min()), "seconds")
print("Max time NOTA:", round(time_data_nota["response_time"].max()), "seconds")
print("Median time NOTA:", round(time_data_nota["response_time"].median()), "seconds")

# %%
print(
    "Min time code selected:",
    round(time_data_not_nota["response_time"].min()),
    "seconds",
)
print(
    "Max time code selected:",
    round(time_data_not_nota["response_time"].max()),
    "seconds",
)
print(
    "Median time code selected:",
    round(time_data_not_nota["response_time"].median()),
    "seconds",
)

# %%
print(
    "Time difference between NOTA answers and baseline:",
    round(time_data_nota["response_time"].median())
    - round(time_data_no_closed_question["response_time"].median()),
)
print(
    "Time difference between not NOTA answers and baseline:",
    round(time_data_not_nota["response_time"].median())
    - round(time_data_no_closed_question["response_time"].median()),
)
print(
    "Time difference between not NOTA answers and NOTA:",
    round(time_data_not_nota["response_time"].median())
    - round(time_data_nota["response_time"].median()),
)

# %% [markdown]
# Mann-Whitney test to check for difference between median time when NOTA selected and NOTA not selected.
# Null hypothesis: "There is no difference in the distribution of time spent between two groups".

# %%
pvalue_time = mannwhitneyu(
    time_data_not_nota["response_time"],
    time_data_nota["response_time"],
    alternative="two-sided",
).pvalue
print(pvalue_time)

# %% [markdown]
# The median response time when the closed question was asked for both, NOTA and not NOTA answers were very similar (4 seconds difference). The p-value (0.87) using Mann-Whitney U-test suggests there is no difference between two groups.
#
# Respondents spent the same amount of time reading and considering possible answers; neither NOTA answers came from respondents skipping through, nor the quality of the options was a problem (no options lead to respondents needing to think too much on the answers).

# %%
# IQRs (interquartile range)

iqr_nota = time_data_nota["response_time"].quantile(0.75) - time_data_nota[
    "response_time"
].quantile(0.25)
iqr_not_nota = time_data_not_nota["response_time"].quantile(0.75) - time_data_not_nota[
    "response_time"
].quantile(0.25)
iqr_no_closed_question = time_data_no_closed_question["response_time"].quantile(
    0.75
) - time_data_no_closed_question["response_time"].quantile(0.25)

# %%
print("IQR for NOTA:", round(iqr_nota))
print("IQR for not NOTA:", round(iqr_not_nota))
print("IQR for no question:", round(iqr_no_closed_question))

# %% [markdown]
# This suggests that there is more variety when an answer was selected, than when NOTA was selected. I.e. some matches are perfect fits (quicker answer), but other require more time to think which one fits best.
#
#
# ## Time spent on the survey and Number of Sections
# Analyse the duration time vs number of sections.

# %%
data_not_none = data[data["survey_assist_closed_question_response"].notna()]

# %%
not_none_unique_section = pd.DataFrame(
    data_not_none.apply(possible_sections, axis=1).to_list(),
    index=data_not_none.index,
    columns=["unique_sections", "codes_count"],
)

data_not_none.loc[:, ["unique_sections", "codes_count"]] = not_none_unique_section

# %%
# calculate time vs section

list_length = data_not_none["unique_sections"].str.len() == 1
data_not_none_one_section = data_not_none[list_length].copy()
data_not_none_multi_section = data_not_none[~list_length].copy()

# %%
one_sec_response_time = get_duration_seconds(data_not_none_one_section)
multi_sec_response_time = get_duration_seconds(data_not_none_multi_section)


data_not_none_one_section.loc[:, "response_time"] = one_sec_response_time
data_not_none_multi_section.loc[:, "response_time"] = multi_sec_response_time

# %%
print(
    "Min time one section:",
    round(data_not_none_one_section["response_time"].min()),
    "seconds",
)
print(
    "Max time one section:",
    round(data_not_none_one_section["response_time"].max()),
    "seconds",
)
print(
    "Median time one section:",
    round(data_not_none_one_section["response_time"].median()),
    "seconds",
)

# %%
print(
    "Min time multiple sections:",
    round(data_not_none_multi_section["response_time"].min()),
    "seconds",
)
print(
    "Max time multiple sections:",
    round(data_not_none_multi_section["response_time"].max()),
    "seconds",
)
print(
    "Median time multiple sections:",
    round(data_not_none_multi_section["response_time"].median()),
    "seconds",
)

# %%
pvalue_time_sections = mannwhitneyu(
    data_not_none_one_section["response_time"],
    data_not_none_multi_section["response_time"],
    alternative="two-sided",
).pvalue
print(pvalue_time_sections)

# %% [markdown]
# There is 16 seconds difference between the median time it takes to select any answer (one associated with a code, or NOTA). When options are from multiple sections, the time it takes to make a decision is longer (152s), than when all options are from one section (136s).
