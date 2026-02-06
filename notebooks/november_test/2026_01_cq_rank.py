# %%
"""Work in progess.

Initial analysis of survey responses, focusing on ranks in Closed Follow up questions.

Create .env file with bucket variables, such as
PREPROD_DATA_BUCKET = "gs://<bucket-name>/<folder>/".
"""

# %%
# pylint: disable=C0103, C0301, C0114, R0801
# ruff: noqa: PLR2004

# %%
import dotenv
import numpy as np
import pandas as pd
from scipy.stats import (
    chisquare,
)

# %%
preprod_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET")
if not preprod_bucket:
    raise ValueError("PREPROD_DATA_BUCKET not found in .env file. Please set it.")

# %%
data = pd.read_parquet(
    f"{preprod_bucket}analysis-interim-results/closed_questions/closed_questions_codes.parquet"
)

# %%
# ranks selected by repsondent


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
for i in range(7):
    print(
        f"{i} options: {alt_codes_count.count(i)} ({round(alt_codes_count.count(i)/ len(alt_codes_count) * 100,1)}%)"
    )

# %%
options_dict = {
    "selected_response": selected_response,
    "alt_codes_count": alt_codes_count,
}
df_options = pd.DataFrame(options_dict)

# %% [markdown]
# ## SA assigned code randomness
#
# Check if the order of the options presented was random

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
for i in range(1, 7):
    print(f"code rank {i}: {sa_code_match.count(i)}")
print(f"None: {int(np.isnan(sa_code_match).sum())}")

# %% [markdown]
# Note that 6th rank is either not presented or is a "none of the above", which is not a valid code.
# "None" answers are the count of surveys that were not presented with a closed question.

# %%
# count of alternative codes
for i in range(6):
    print(f"{i} altetnative codes found: {sa_alt_codes_count.count(i)}")

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
    """Checks if there is a selection bias in the subset, based on the number of options presented.

    Args:
        df (pd.DataFrame): a dataframe with 'alt_codes_count' and 'selected_response' columns.
    """
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
df_sa_codes_no_none = df_sa_codes[~df_sa_codes["selected_response"].isna()]

# %%
# check if any of the options was selected more often than others. If p-values are > 0.05, then there is no significant difference in the responses seleted.
# Goodness-of-fit

for i in range(2, 6):
    df_group_sa = df_options[df_options["alt_codes_count"] == i]

    observed_count = df_group_sa["selected_response"].value_counts().values
    chi_pvalue = chisquare(f_obs=observed_count).pvalue

    print(f"Options presented: {i}")
    print(f"Surveys count: {df_group_sa.shape[0]}")
    print(
        f"p-value: {round(chi_pvalue, 2)}\nGreater than alpha 0.05: {chi_pvalue > 0.05}\n"
    )

# %% [markdown]
# Null hypothesis: "Respondents do not favour one of the options over other options".
# All p-values, regardles of the count of options presented, are above alpha=0.05. Therefore, the null hypothesis cannot be rejected, and it is possible that respondents don't favour any option.
#
# Order of the options doesn't seem to influence the number of times, the option was selected.

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
    df_sa_codes_no_none = df_sa_codes_no_none[
        ~df_sa_codes_no_none["selected_response"].isna()
    ]
    df_group_sa = df_sa_codes_no_none[df_sa_codes_no_none["alt_codes_count"] == i]

    observed_count = df_group_sa["selected_response"].value_counts().values
    chi_pvalue = chisquare(f_obs=observed_count).pvalue

    print(f"Options presented: {i}")
    print(f"Surveys count: {df_group_sa.shape[0]}")
    print(f"p-value: {chi_pvalue}\nGreater than alpha 0.05:\n  {chi_pvalue > 0.05}\n")

# %% [markdown]
# When 2 and 4 options were presented, we do not reject the null hypothesis (pvalue > 0.05).
# When 3 and 5 options were presented, we can reject the null hypothesis (pvalue > 0.05), suggesting that one of the options was selected more often.
#
# (possibly not noticable, because 2 and 4 options were presented only in small number of surveys, unlike 3 and 5)
#
# Check for if the LLM shortlist aligns with the respondents selection.

# %%
check_primacy(df_sa_codes_no_none)

# %% [markdown]
# The residual value for first option being selected confirms that it was selected more often in cases, when 3 and 5 options were presented.For all options, the first option was selected more often than expected. This suggests that LLM was generally correct with its hierarchical codes selection.
