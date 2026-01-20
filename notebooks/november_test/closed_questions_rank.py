# %%
"""Work in progess.

Initial analysis of survey responses, focusing on Closed Follow up questions.

Create .env file with bucket variables, such as EVALUATION_BUCKET = "gs://<bucket-name>/<folder>/",
and PREPROD_DATA_BUCKET similarly.
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
    chisquare,
    contingency,
    mannwhitneyu,
    shapiro,
    ttest_ind,
)

from survey_assist_utils.data_cleaning.sic_codes import get_clean_n_digit_codes

# %%
evaluation_bucket = dotenv.get_key(".env", "EVALUATION_BUCKET")
analysis_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET")
if not evaluation_bucket:
    raise ValueError("EVALUATION_BUCKET not found in .env file. Please set it.")
if not analysis_bucket:
    raise ValueError("PREPROD_DATA_BUCKET not found in .env file. Please set it.")

# %%
data = pd.read_parquet(
    f"{analysis_bucket}analysis-interim-results/closed_questions/closed_questions_codes.parquet"
)


# %%
def get_selected_responses(response_row: pd.Series) -> int | None:
    """Finds the rank the response selected by user.

    Args:
        response_row: survey responses.

    Return:
        code_rank: int | None
    """
    column = "survey_assist_closed_question_option_"

    # save the response selected by the user
    response = response_row["survey_assist_closed_question_response"]

    # check only when closed question was asked
    if response is not None and response != "none of the above":

        # change options from closed questions to lower case (matching the selected response)
        for k in range(1, 7):
            column_name = column + str(k)
            if response_row[column_name] is not None:
                response_row[column_name] = response_row[column_name].lower()

        # find the order of the selected response
        code_rank = 1
        while code_rank < 7:
            column_name = column + str(code_rank)
            if response == response_row[column_name]:
                return code_rank
            code_rank += 1
    return None


# %%
selected_response = (
    data.apply(get_selected_responses, axis=1).dropna().astype(int).to_list()
)

# %%
# percentage of codes found using closed quesiton for all surveys

print(round(100 * len(selected_response) / len(data), 2))

# %%
selected_response_order = {}
for i in range(1, 7):
    picked_response = "option_" + str(i)
    selected_response_order[picked_response] = selected_response.count(i)

# %%
# order, which answer was seleceted
print(selected_response_order)

# order of the question selected
for i in range(1, 7):
    option_order = "option_" + str(i)
    print(
        f"Option in order {i} [%]: {round(100 * selected_response_order[option_order] / len(selected_response), 2)}"
    )

# %% [markdown]
# Note, "none of the above" is always presented at the bottom of the list, which means the 6th option never brings a code. Options 1 and 2 will always have possible codes.

# %%
# we get "None" when the closed question was not asked or the answer was "none of the above"
print(data[data["survey_assist_closed_question_response_code"] == "None"].shape[0])

# %%
none_of_the_above_answer = (
    data["survey_assist_closed_question_response"] == "none of the above"
).sum()

# %%
# asked closed question, but didn't get a code
print(round(100 * none_of_the_above_answer / len(selected_response), 2))

# %%
options_columns = []
for i in range(1, 6):
    options_columns.append(f"survey_assist_closed_question_option_{i}_code")


# %%
def get_alt_codes_count(response_row: pd.Series) -> int | None:
    """Get count of alternative codes presented to the user.

    Args:
        response_row (pd.Series): row with survey response

    Return:
        alt_count (int | None): alternative codes count
    """
    alt_count = 0
    for k in range(1, 6):
        if (
            response_row[f"survey_assist_closed_question_option_{k}_code"] is not None
            and (response_row["survey_assist_closed_question_response"]).lower()
            != "none of the above"
        ):
            alt_count += 1

    return alt_count if alt_count > 0 else None


# %%
alt_codes_count = data.apply(get_alt_codes_count, axis=1).dropna().astype(int).to_list()

# %%
len(alt_codes_count)

# %%
len(selected_response)

# %%
for i in range(7):
    print(alt_codes_count.count(i))

# %%
options_dict = {
    "selected_response": selected_response,
    "alt_codes_count": alt_codes_count,
}
df_options = pd.DataFrame(options_dict)

# %% [markdown]
# ## SA assigned code randomness

# %%
# selected code rank from the list


def get_code_rank(response_row: pd.Series) -> int | None:
    """Get the rank of the code selected by the user.

    Args:
        response_row (pd.Series): row with survey response

    Return:
        code_rank (int | None): rank of the code selected.
    """
    survey_assist_alt = "survey_assist_alt_candidate_code_"
    k = 1
    code_rank = 0
    while k < 6:
        sa_code = survey_assist_alt + str(k)
        if response_row["survey_assist_closed_question_response_code"] is None:
            k = 6

        elif (
            response_row[sa_code]
            == response_row["survey_assist_closed_question_response_code"]
        ):
            code_rank = k
            k = 6
        k += 1

    if k == 6:
        code_rank = k

    return code_rank if code_rank != 6 else None


# %%
sa_code_match = data.apply(get_code_rank, axis=1).to_list()

# %%
# alternative codes count


def get_alternative_codes_count(response_row: pd.Series) -> int:
    """Get count of alternative codes presented to the user.

    Args:
        response_row (pd.Series): row with survey response

    Return:
        alts (int): alternative codes count
    """
    k = 1
    alts = 0
    survey_assist_alt = "survey_assist_alt_candidate_code_"

    while k < 6:
        sa_code = survey_assist_alt + str(k)
        if response_row["survey_assist_closed_question_response_code"] is None:
            k = 6
        elif response_row[sa_code] is not None:
            alts += 1
        k += 1
    return alts


# %%
sa_alt_codes_count = data.apply(get_alternative_codes_count, axis=1).to_list()

# %%
for i in range(6):
    print(sa_code_match.count(i))
print(sa_code_match.count(None))

# %% [markdown]
# Note that 6th rank is either not presented or is a "none of the above", which is not a valid code.
# "None" answers are the count of surveys that were not presented with a closed question.

# %%
# count of alternative codes
for i in range(6):
    print(sa_alt_codes_count.count(i))

# %% [markdown]
# Zero alternative codes, when final code is found. No 1 alternative codes, as that means the final code is found.
# 5 alternative codes count is high - this includes rows, that selected final code.

# %% [markdown]
# #### Chi square: Tendency to favouring a specific rank option.
# Hypothesis: "Respondents don't favour the n-th option".

# %%
# create a DF for checking the chi square
sa_codes = {"selected_response": sa_code_match, "alt_codes_count": sa_alt_codes_count}
df_sa_codes = pd.DataFrame(sa_codes)


# %%
# check for primacy effect
def check_primacy(df: pd.DataFrame):

    for k in range(2, 6):
        df_grouped_sa = df[df["alt_codes_count"] == k]
        observed = df_grouped_sa["selected_response"].value_counts().values
        expected = observed.sum() / k
        residual = round((observed[0] - expected) / (expected**0.5), 3)
        print(
            f"Presented codes count: {k}, primacy effect using standardised residual: {residual}"
        )


# %%
check_primacy(df_options)

# %% [markdown]
# For all number of options presented, the residual values are within -1.96 < residual < 1.96, which means there is not enough evidence that respondents favour the first option presented (primacy effect).

# %%
# remove all rows that didn't get the closed question asked
df_sa_codes_no_none = df_sa_codes[df_sa_codes["selected_response"] != "None"]

# %%
# check if any of the options was selected more often than others. If p-values are > 0.05, then there is no significant difference in the responses seleted.
# Goodness-of-fit

for i in range(2, 6):
    df_group_sa = df_options[df_options["alt_codes_count"] == i]

    observed_count = df_group_sa["selected_response"].value_counts().values
    chi_pvalue = chisquare(f_obs=observed_count).pvalue

    print(f"Options presented: {i}")
    print(f"Surveys count: {df_group_sa.shape[0]}")
    print(f"p-value: {chi_pvalue}\nGreater than alpha 0.05 {chi_pvalue > 0.05}\n")

# %%
# for the whole group, using weighted expected frequencies

obs_all = df_options["selected_response"].value_counts().sort_index().values
expected_prob = []
for rank in range(1, 6):
    prob = (
        1 / df_options[df_options["alt_codes_count"] >= rank]["alt_codes_count"]
    ).sum()
    expected_prob.append(prob)
exp_all = np.array(expected_prob)

# %%
chi_pvalue_all = chisquare(f_obs=obs_all, f_exp=exp_all).pvalue
print(chi_pvalue_all)

# %% [markdown]
# Using weighted expected frequency, the p-value for the whole data collected is 0.59, which is >0.05. There is no significant deviation - the position of the options did not influence the respondents choice.

# %% [markdown]
# ### Check LLM's assumption regarding most likely code.
#
# Use the ordered list, to check if the first option was favoured - this is good, because this will means that the respondent selects what SA thinks is most likely.
# When comapring options that were selected by respondents with the order of those options determined by the LLM, we expect the first option to be most popular, as LLM decide it has the highest likelihood.
# Null hypothesis: "Respondents don't favour the first option".

# %%
for i in range(2, 6):
    df_group_sa = df_sa_codes_no_none[df_sa_codes_no_none["alt_codes_count"] == i]

    observed_count = df_group_sa["selected_response"].value_counts().values
    chi_pvalue = chisquare(f_obs=observed_count).pvalue

    print(f"Options presented: {i}")
    print(f"Surveys count: {df_group_sa.shape[0]}")
    print(f"p-value: {chi_pvalue}\nGreater than alpha 0.05 {chi_pvalue > 0.05}\n")

# %% [markdown]
# When 2 and 4 options were presented, we do not reject the null hypothesis (pvalue > 0.05).
# When 3 and 5 options were presented, we can reject the null hypothesis (pvalue > 0.05), suggesting that one of the options was selected more often.
#
# (possibly not noticable, because 2 and 4 options were presented only in small number of surveys, unlike 3 and 5)
#
# Check for primacy effect (we want that).

# %%
check_primacy(df_sa_codes_no_none)

# %% [markdown]
# The residual value for first option being selected confirms that it was selected more often in cases, when 3 and 5 options were presented.For all options, the first option was selected more often than expected. This suggests that LLM was generally correct with its hierarchical codes selection.

# %% [markdown]
# Null hypothesis: "Respondents do not favour one of the options over other options".
# All p-values, regardles of the count of options presented, are above alpha=0.05. Therefore, the null hypothesis cannot be rejected, and it is possible that respondents don't favour any option.
#
# Order of the options doesn't seem to influence the number of times, the option was selected.

# %% [markdown]
# ## None of the above (NOTA)
#
# Find reasons why respondent selected "None of the above" when presented with closed question.

# %%
full_data = pd.read_parquet(
    f"{analysis_bucket}analysis-interim-results/evaluation_df_with_sa_clean_codes.parquet"
)

# %%
data["sa_initial_codes"] = full_data["sa_initial_codes"]

# %%
data_nota = data[data["survey_assist_closed_question_response"] == "none of the above"]
data_not = data[data["survey_assist_closed_question_response"] != "none of the above"]
data_not_nota = data_not[data_not["survey_assist_closed_question_response"].notna()]

# %%
# check against sections


def possible_sections(response_row: pd.Series):
    """Checks the count of options presented to the user. Counts the number of unique sections of the presented options, and adds to the dataframe.

    Args:
        response_row (pd.Series): a row containing survey results.
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

    unique_section_list_by_row = list(get_clean_n_digit_codes(row_codes, n=0)[0])

    return unique_section_list_by_row, codes


# %%
# add columns 'unique_sections' and 'codes_count'

data_not_nota[["unique_sections", "codes_count"]] = pd.DataFrame(
    data_not_nota.apply(possible_sections, axis=1).to_list(), index=data_not_nota.index
)
data_nota[["unique_sections", "codes_count"]] = pd.DataFrame(
    data_nota.apply(possible_sections, axis=1).to_list(), index=data_nota.index
)

# %%
df_check = data_not_nota
# df_check = data_nota

sections = 0
count_codes = 0
for data_row in range(len(df_check)):
    sections += len(df_check["unique_sections"].iloc[data_row])
    count_codes += df_check["codes_count"].iloc[data_row]
print(sections / count_codes)

# %%
# Average number of codes shown

avg_codes_count = (
    data_not_nota["codes_count"].sum() + data_nota["codes_count"].sum()
) / (len(data_not_nota) + len(data_nota))
avg_codes_count_nota = data_nota["codes_count"].sum() / len(data_nota)
avg_codes_count_not_nota = data_not_nota["codes_count"].sum() / len(data_not_nota)

# %%
print(f"Average count of codes presented to the respondent {avg_codes_count}")
print(
    f"Average count of codes presented to the respondent when NOTA was selected {avg_codes_count_nota}"
)
print(
    f"Average count of codes presented to the respondent when NOTA was not selected {avg_codes_count_not_nota}"
)

# %%
# check statistical signifcance of those averages

pvalue_average_options_count = ttest_ind(
    data_nota["codes_count"], data_not_nota["codes_count"], equal_var=False
).pvalue

# %%
print(f"p-value for average options count: {pvalue_average_options_count}")

# %% [markdown]
# This suggests that the number of optoins isn't a significant reason for NOTA to be selected.

# %% [markdown]
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
data_nota["options_with_other"] = pd.DataFrame(
    data_nota.apply(count_others, axis=1).to_list(), index=data_nota.index
)

data_not_nota["options_with_other"] = pd.DataFrame(
    data_not_nota.apply(count_others, axis=1).to_list(), index=data_not_nota.index
)

# %%
data_nota["other_ratio"] = data_nota["options_with_other"] / data_nota["codes_count"]

data_not_nota["other_ratio"] = (
    data_not_nota["options_with_other"] / data_not_nota["codes_count"]
)

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
    data_nota["other_ratio"], data_not_nota["other_ratio"], alternative="two-sided"
).pvalue

# %%
print(p_value_whitney_other)

# %% [markdown]
# Again, the p-value obtained from Mann-Whitney test (0.043) is close but below to 0.05. This allows to reject the hypothesis - there is a difference.

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
#
# Calculate the **odds ratio** now.
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
nota_many_sections = (nota_sections_count > 1).sum()

not_nota_one_section = (not_nota_sections_count == 1).sum()
not_nota_many_sections = (not_nota_sections_count > 1).sum()

# %%
row1 = [nota_one_section, not_nota_one_section]
row2 = [nota_many_sections, not_nota_many_sections]

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
multiple_sections = not_nota_many_sections / (
    nota_many_sections + not_nota_many_sections
)

# %%
print(f"Success rate when options presented from the same section: {same_section}")
print(
    f"Success rate when options presented from multiple sections: {multiple_sections}"
)

# %%
contingency_table = np.array(
    [
        [not_nota_one_section, nota_one_section],
        [not_nota_many_sections, nota_many_sections],
    ]
)

p_value_success_rate = chi2_contingency(contingency_table).pvalue

# %%
print(p_value_success_rate)

# %% [markdown]
# The p-value for success rate is 0.15. There is no statistical significance between number of secctions the presented options came from.

# %% [markdown]
# ### Time it took the respondent to answer the survey
#
# Check if respondents just went with NOTA, to finish the survey sooner. Use surveys that didn't get asked closed question as a baseline.

# %%
time_data_nota = full_data[
    full_data["survey_assist_closed_question_response"] == "none of the above"
]
time_data_not = full_data[
    full_data["survey_assist_closed_question_response"] != "none of the above"
]
time_data_not_nota = time_data_not[
    time_data_not["survey_assist_closed_question_response"].notna()
]
time_data_no_closed_question = full_data[
    full_data["survey_assist_closed_question_response"].isna()
]

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
def get_duration_minutes(df: pd.DataFrame) -> pd.Series:
    start = df["time_start"]
    end = df["time_end"]
    return (end - start).dt.total_seconds()  # time in seconds


# %%
# get response time for nota / not nota / no question
time_data_nota["response_time"] = get_duration_minutes(time_data_nota)
time_data_not_nota["response_time"] = get_duration_minutes(time_data_not_nota)
time_data_no_closed_question["response_time"] = get_duration_minutes(
    time_data_no_closed_question
)

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
# Analyse the duration time vs number of sections.

# %%
data_not_none = full_data[full_data["survey_assist_closed_question_response"].notna()]

# %%
data_not_none[["unique_sections", "codes_count"]] = pd.DataFrame(
    data_not_none.apply(possible_sections, axis=1).to_list(), index=data_not_none.index
)

# %%
# calculate time vs section

list_length = data_not_none["unique_sections"].str.len() == 1
data_not_none_one_section = data_not_none[list_length]
data_not_none_multi_section = data_not_none[~list_length]

# %%
data_not_none_one_section["response_time"] = get_duration_minutes(
    data_not_none_one_section
)
data_not_none_multi_section["response_time"] = get_duration_minutes(
    data_not_none_multi_section
)

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

# %% [markdown]
# There is 16 seconds difference between the median time it takes to select any answer (one associated with a code, or NOTA). When options are from multiple sections, the time it takes to make a decision is longer (152s), than when all options are from one section (136s).

# %% [markdown]
# #### Results
#
# Respondents selected NOTA less often when they are presented with options form the same section. The odds of failing (respondent selecting NOTA) are getting lower by 37% when the user is presented with options from the same section.
#
# Odds ratio: 0.63
#
# Same section:
# - Success rate: 84.5%
#
# Multiple sections:
# - Success rate: 77.3%
#
# With options presented from multiple sections, the number of NOTA answers increases, as well as the time spent on the survey (median 136s for same section increases to median 152s for multiple sections, jump of 16 seconds), suggesting that respondent can be more confused with the options presented [respondent needs to "work harder" to pick one option, doesn't findanything that matches (or findstwo or more options that work equally well), and selects NOTA].
