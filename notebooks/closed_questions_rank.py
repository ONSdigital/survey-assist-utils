# %%
"""Work in progess.

Initial analysis of survey responses, focusing on Closed Follow up questions.

Create .env file with bucket variables, such as EVALUATION_BUCKET = "gs://<bucket-name>/<folder>/",
and ANALYSIS_BUCKET similarly.
"""

# %%
# pylint: disable=C0103, C0116, C0301, C0114, R0801
# ruff: noqa: PLR2004

import re

# %%
import dotenv
import pandas as pd
from industrial_classification.hierarchy.sic_hierarchy import load_hierarchy
from industrial_classification_utils.embed.embedding import get_config
from industrial_classification_utils.utils.sic_data_access import (
    load_sic_index,
    load_sic_structure,
)
from scipy import stats

# %%
evaluation_bucket = dotenv.get_key(".env", "EVALUATION_BUCKET")
analysis_bucket = dotenv.get_key(".env", "ANALYSIS_BUCKET")
if not evaluation_bucket:
    raise ValueError("EVALUATION_BUCKET not found in .env file. Please set it.")
if not analysis_bucket:
    raise ValueError("ANALYSIS_BUCKET not found in .env file. Please set it.")

# %%
data = pd.read_parquet(
    f"{analysis_bucket}closed_questions/closed_questions_codes.parquet"
)

# %%
column = "survey_assist_closed_question_option_"
selected_response = []

# iterate through all rows of the data
for data_row in range(len(data)):
    # save the response selected by the user
    response = data["survey_assist_closed_question_response"][data_row]

    # check only when closed question was asked
    if response is not None and response != "none of the above":

        # change options from closed questions to lower case (matching the selected response)
        for option_order in range(1, 7):
            col_name = column + str(option_order)
            if data[col_name][data_row] is not None:
                data.loc[data_row, col_name] = data[col_name][data_row].lower()

        # find the order of the selected response
        j = 1
        while j < 7:
            col_name = column + str(j)
            if response == data[col_name][data_row]:
                selected_response.append(j)
                j = 7
            j += 1

# %%
# percentage of codes found using closed quesiton for all surveys
print(round(100 * len(selected_response) / len(data), 2))

# %%
selected_response_order = {}
for option_order in range(1, 7):
    picked_response = "option_" + str(option_order)
    selected_response_order[picked_response] = selected_response.count(option_order)

# %%
# order, which answer was seleceted
print(selected_response_order)
# order of the question selected
print(
    "1st rank [%]:",
    round(100 * selected_response_order["option_1"] / len(selected_response), 2),
)
print(
    "2nd rank [%]:",
    round(100 * selected_response_order["option_2"] / len(selected_response), 2),
)
print(
    "3rd rank [%]:",
    round(100 * selected_response_order["option_3"] / len(selected_response), 2),
)
print(
    "4th rank [%]:",
    round(100 * selected_response_order["option_4"] / len(selected_response), 2),
)
print(
    "5th rank [%]:",
    round(100 * selected_response_order["option_5"] / len(selected_response), 2),
)
print(
    "6th rank [%]:",
    round(100 * selected_response_order["option_6"] / len(selected_response), 2),
)

# %% [markdown]
# Note, "none of the above" is always presented at the bottom of the list, which means the 6th option never brings a code. Options 1 and 2 will always have possible codes.

# %%
sic_rephrased = pd.read_csv(
    f"{evaluation_bucket}sic_rephrased_descriptions_2025_02_03.csv", dtype=str
)


# %%
def convert_to_dict(dict_string):
    clean_string = dict_string.strip().strip("'")

    code = re.search(r"{Code: \s*(.*?),\s*Title: ", clean_string, re.DOTALL)
    code_value = code.group(1).strip() if code else ""

    title = re.search(
        r"Title: \s*(.*?),\s*Example activities: ", clean_string, re.DOTALL
    )
    title_value = title.group(1).strip() if title else ""
    title_value = title_value.lower()

    activities = re.search(r"Example activities: \s*(.*?)\s*}", clean_string, re.DOTALL)
    activities_value = activities.group(1).strip() if activities else ""

    result_dictionary = {
        "Code": code_value,
        "Title": title_value,
        "Example activities": activities_value,
    }
    return result_dictionary


# %%
sic_rephrased["reviewed_description"] = sic_rephrased[
    "reviewed_description"
].str.lower()
sic_rephrased["llm_rephrased_description"] = sic_rephrased[
    "llm_rephrased_description"
].str.lower()


# %%
def get_code_by_title(dictionary, title):
    for item in dictionary:
        if item.get("Title") == title:
            return item.get("Code")
    return "XXXXX"


# %%
sic_dictionary = sic_rephrased["input_description"].apply(convert_to_dict)
none_of_the_above = 0

options: dict[str, list[str | None]] = {"1": [], "2": [], "3": [], "4": [], "5": []}

for data_row in range(len(data)):
    # if the closed question response is None, it means the question was not asked, and there's no codes.
    if data["survey_assist_closed_question_response"][data_row] is not None:

        option_order = 1
        while option_order < 6:
            current_row = data[f"survey_assist_closed_question_option_{option_order}"][
                data_row
            ].lower()
            if current_row == "none of the above":
                none_of_the_above += 1
                while option_order < 6:
                    options[f"{option_order}"].append(None)
                    option_order += 1

            else:
                if current_row in list(sic_rephrased["reviewed_description"]):
                    sic_code = sic_rephrased[
                        sic_rephrased["reviewed_description"] == current_row
                    ]["input_code"].item()
                    options[f"{option_order}"].append(str(sic_code))

                else:
                    sic_code = get_code_by_title(sic_dictionary, current_row)
                    options[f"{option_order}"].append(str(sic_code))

                option_order += 1
    else:
        options["1"].append(None)
        options["2"].append(None)
        options["3"].append(None)
        options["4"].append(None)
        options["5"].append(None)

# %%
# we get "None" when the closed question was not asked or the answer was "none of the above"
print(data[data["survey_assist_closed_question_response_code"] == "None"].shape[0])

# %%
# asked closed question, but didn't get a code
round(100 * none_of_the_above / len(selected_response), 2)

# %%
options.keys()

# %%
options_df = pd.DataFrame(options)

# %%
alt_codes_count = []
for option_order in range(len(data)):
    responses = int(options_df.iloc[option_order].count())
    if (
        responses != 0
        and data["survey_assist_closed_question_response"][option_order]
        != "none of the above"
    ):
        alt_codes_count.append(responses)

# %%
print(alt_codes_count.count(0))
print(alt_codes_count.count(1))
print(alt_codes_count.count(2))
print(alt_codes_count.count(3))
print(alt_codes_count.count(4))
print(alt_codes_count.count(5))
print(alt_codes_count.count(6))

# %%
options_dict = {
    "selected_response": selected_response,
    "alt_codes_count": alt_codes_count,
}
df_options = pd.DataFrame(options_dict)


# %%
def chi2_test(df, response_column: str, alt_codes_column: str, n: int = 1):
    """Args:
    df: DataFrame
    response_column: a column with the respondends choice recorded as a rank
    alt_codes_column: count of alternative codes
    n: n-th option to be tested.
    """
    # The n-th option was selected, where 'none of the above' was NOT selected
    N_P1 = df[df[response_column] == n].shape[0]
    print(N_P1)

    # Total number of surveys asked, where 'none of the above' was NOT selected
    N_Surveys = df.shape[0]

    # Total number of potential answers (other than 'none of the above') was presented
    N_Opportunities = df[alt_codes_column].sum()

    # null hypothesis: P_Expected = N_Surveys / N_Opportunities; how many 1st options are we expecting
    P_Expected = N_Surveys / N_Opportunities

    # Expected first responses
    E_P1 = P_Expected * N_Surveys

    # Expected other responses than first
    E_P2_P5 = N_Surveys - E_P1

    # Observed values
    O_P1 = N_P1
    O_P2_P5 = N_Surveys - N_P1

    # Chi-square stats
    chi2_st = (O_P1 - E_P1) ** 2 / E_P1 + (O_P2_P5 - E_P2_P5) ** 2 / E_P2_P5
    p_value = stats.chi2.sf(chi2_st, 1)

    return p_value


# %%
df_grouped = df_options[df_options["alt_codes_count"] == 2]
print(chi2_test(df_grouped, "selected_response", "alt_codes_count"))
# print(chi2_test(df_options, 'selected_response', 'alt_codes_count'))

# %%
df_grouped.sample()

# %% [markdown]
# ## SA assigned code randomness

# %%
# selected code rank from the list
sa_code_match: list[int | str] = []
survey_assist_alt = "survey_assist_alt_candidate_code_"
for data_row in range(len(data)):
    if data["survey_assist_closed_question_response"].iloc[data_row] is None:
        pass
    else:
        option_order = 1
        while option_order < 6:
            sa_code = survey_assist_alt + str(option_order)
            if data["survey_assist_closed_question_response_code"][data_row] is None:
                # sa_code_match.append(0)
                option_order = 6
            elif (
                data[sa_code][data_row]
                == data["survey_assist_closed_question_response_code"][data_row]
            ):
                sa_code_match.append(option_order)
                option_order = 6
            option_order += 1
        if option_order == 6:
            sa_code_match.append("None")

# %%
# alternative codes count
sa_alt_codes_count = []
survey_assist_alt = "survey_assist_alt_candidate_code_"
for data_row in range(len(data)):
    if data["survey_assist_closed_question_response"].iloc[data_row] is None:
        pass
    else:
        alts = 0
        option_order = 1
        while option_order < 6:
            sa_code = survey_assist_alt + str(option_order)
            if data["survey_assist_closed_question_response_code"][data_row] is None:
                option_order = 6
            elif data[sa_code][data_row] is not None:
                alts += 1
            option_order += 1
        sa_alt_codes_count.append(alts)

# %%
# rank of the selected code
print("1:", sa_code_match.count(1))
print("2:", sa_code_match.count(2))
print("3:", sa_code_match.count(3))
print("4:", sa_code_match.count(4))
print("5:", sa_code_match.count(5))
print("6:", sa_code_match.count(6))
print("None:", sa_code_match.count("None"))

# %% [markdown]
# Note that 6th rank is either not presented or is a "none of the above", which is not a valid code.
# "None" answers are the count of surveys that were not presented with a closed question.

# %%
# count of alternative codes
print("0:", sa_alt_codes_count.count(0))
print("1:", sa_alt_codes_count.count(1))
print("2:", sa_alt_codes_count.count(2))
print("3:", sa_alt_codes_count.count(3))
print("4:", sa_alt_codes_count.count(4))
print("5:", sa_alt_codes_count.count(5))

# %% [markdown]
# Zero alternative codes, when final code is found. No 1 alternative codes, as that means the final code is found.
# 5 alternative codes count is high - this includes rows, that selected final code.

# %% [markdown]
# #### Chi square: Tendency to favouring a specific rank option.
# Hypothesis: "Respondents tend to favour the n-th option".

# %%
# create a DF for checking the chi square
sa_codes = {"selected_code": sa_code_match, "alt_codes_count": sa_alt_codes_count}
df_sa_codes = pd.DataFrame(sa_codes)

# %%
print("1:", chi2_test(df_sa_codes, "selected_code", "alt_codes_count", 1))
print("2:", chi2_test(df_sa_codes, "selected_code", "alt_codes_count", 2))
print("3:", chi2_test(df_sa_codes, "selected_code", "alt_codes_count", 3))
print("4:", chi2_test(df_sa_codes, "selected_code", "alt_codes_count", 4))
print("5:", chi2_test(df_sa_codes, "selected_code", "alt_codes_count", 5))

# %% [markdown]
# For all number of optins presented, the p-value is belov 0.5, which means the null hypothesis cannot be rejected - there is a sign of bias when respondent select specific rank option. For every rank it was selected it significantly differs from what we would expect by random chance (1/k). This might be due to comparing lists of different length (not all respondents were presented with the same number of options).

# %%
# remove all rows that didn't get the closed question asked
df_sa_codes_no_none = df_sa_codes[df_sa_codes["selected_code"] != "None"]

# %%
# get p-values for each group of number of options presented vs the rank of the option selected

print("Alt codes count x rank of selected response\n")
df_group_sa = df_sa_codes_no_none[df_sa_codes_no_none["alt_codes_count"] == 2]
print("number of respondents with 2 options presented:", df_group_sa.shape[0])
print("2x1:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 1))
print("2x2:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 2), "\n")

df_group_sa = df_sa_codes_no_none[df_sa_codes_no_none["alt_codes_count"] == 3]
print("number of respondents with 3 options presented:", df_group_sa.shape[0])
print("3x1:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 1))
print("3x2:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 2))
print("3x3:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 3), "\n")

df_group_sa = df_sa_codes_no_none[df_sa_codes_no_none["alt_codes_count"] == 4]
print("number of respondents with 4 options presented:", df_group_sa.shape[0])
print("4x1:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 1))
print("4x2:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 2))
print("4x3:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 3))
print("4x4:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 4), "\n")

df_group_sa = df_sa_codes_no_none[df_sa_codes_no_none["alt_codes_count"] == 5]
print("number of respondents with 5 options presented:", df_group_sa.shape[0])
print("5x1:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 1))
print("5x2:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 2))
print("5x3:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 3))
print("5x4:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 4))
print("5x5:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 5))

# %% [markdown]
# #### 2 options presented
# Both p-values are above 0.5, which suggests the null hypothesis can be rejected. The share of answers is randomly selected.
#
# #### 3 options presented
# For rank 2, the p-value is equal to 1. This is because the number of times this rank was selected is equal to the expected value (26).
# For ranks 1 and 3, p-value < 0.05 - there is some bias. Further test will need to be run.
#
# #### 4 options presented
# All values are > 0.05. There is no bias for 4 options presented.
#
# #### 5 options presented
# 4 of 5 p-values are low. One of the p-values is 0.49 Similar case as with 3 options presented.

# %% [markdown]
# ### Goodnes-of-fit
#
# For further investingation, run goodness-of-fit test with null hypothesis "the answers selected follow the expected distribution (1/k)" wih k being the number of surveys. This will be considered for groups, where the data will be split based on the number of options presented (not including NOTA answers). Alternative hypothesis: the number of answers selected deviates from the expected 1/k.


# %%
def chi2_gof(df: pd.DataFrame, n: int) -> float:
    """Calculates chi-square goodness-of-fit for a specific subset of data with n options presented.

    Args:
        df (pd.DataFrame): DataFrame with columns
        n (int): number of options presented.

    Returns:
        p_value (float): the p-value for the goodness-of-fit test.

    """
    group_df = df[df["alt_codes_count"] == n]
    # n_total = len(group_df)

    observed_counts = group_df["selected_code"].value_counts().values

    # expected = n_total / n

    p_value = stats.chisquare(f_obs=observed_counts).pvalue

    return p_value


# %%
# goodness-of-fit for each number of presented options
print("2:", chi2_gof(df_sa_codes_no_none, 2))
print("3:", chi2_gof(df_sa_codes_no_none, 3))
print("4:", chi2_gof(df_sa_codes_no_none, 4))
print("5:", chi2_gof(df_sa_codes_no_none, 5))

# %% [markdown]
# ### goodness-of-fit for 2 options presented
# The p-value is greater than 0.05, which means there is evidence to reject the null hypothesis. The differences between observed and expected values are due to noise.
#
# ### goodness-of-fit for 3 options presented
# The p-value is lower than 0.05, which means the null hypothesis cannot be rejected. There is a primacy bias.
#
# ### goodness-of-fit for 4 options presented
# Same as for 2 options.
#
# ### goodness-of-fit for 5 options presented
# Same as for 3 options.
#
#
# Overall, it is uncertain if there is bias throughout the whole survey.

# %% [markdown]
# ## None of the above (NOTA)
#
# Find reasons why respondent selected "None of the above" when presented with closed question.

# %%
full_data = pd.read_parquet(
    f"{analysis_bucket}evaluation_df_with_sa_clean_codes.parquet"
)

# %%
data["sa_initial_codes"] = full_data["sa_initial_codes"]

# %%
data_nota = data[data["survey_assist_closed_question_response"] == "none of the above"]
data_not = data[data["survey_assist_closed_question_response"] != "none of the above"]
data_not_nota = data_not[data_not["survey_assist_closed_question_response"].notna()]


# %%
def count_others(df: pd.DataFrame) -> tuple:
    """Counts the number of option descriptions that contains word 'other'.

    Args:
        df (pd.DataFrame): a dataframe containing survey assist closed question options.

    Returns:
        return_df (pd.DataFrame): a DataFrame with the count of all options presented to the use and count of answers with a word'other'.
        other_percentage (float): percentage of options containing 'other' out of all options for a row.
    """
    column_placeholder = "survey_assist_closed_question_option_"
    options_presented = []
    others = []
    for row in range(len(df)):
        i = 1
        other_count = 0
        while i < 7:
            column_name = column_placeholder + str(i)
            row_string = df[column_name].iloc[row]
            if row_string.lower() == "none of the above":
                options_presented.append(i)
                others.append(other_count)
                i = 7
            else:
                if "other" in row_string.lower():
                    other_count += 1
                i += 1
    return_df = pd.DataFrame(
        {"options_count": options_presented, "other_count": others}
    )

    count = 0
    for row in range(len(return_df)):
        if return_df["other_count"].iloc[row] == 0:
            count += 1
    other_percentage = count / len(return_df)

    return return_df, other_percentage


nota_other_count_df, nota_perc = count_others(data_nota)
not_nota_other_count_df, not_nota_perc = count_others(data_not_nota)

# %%
print(f"percentage of 'other' in possible answers when nota: {round(nota_perc * 100)}%")
print(
    f"percentage of 'other' in possible answers when not nota: {round(not_nota_perc * 100)}%"
)

# %% [markdown]
# this suggests that the proportion of "other..." is not the reason

# %%
config = get_config()

# %%
sic_index_file = config["lookups"]["sic_index"]
sic_index_df = load_sic_index(sic_index_file)

sic_structure_file = config["lookups"]["sic_structure"]
sic_df = load_sic_structure(sic_structure_file)

sic = load_hierarchy(sic_df, sic_index_df)

# %%
# check against sections


def possible_sections(df: pd.DataFrame):
    """Checks the count of options presented to the user. Counts the number of unique sections of the presented options, and adds to the dataframe.

    Args:
        df (pd.DataFrame): a DataFrame containing survey results.
    """
    column_placeholder = "survey_assist_alt_candidate_code_"
    unique_section_list_by_row = []
    codes_count = []

    for row in range(len(df)):
        section_list = []
        codes = 0
        i = 1
        while i < 6:
            column_name = column_placeholder + str(i)
            row_string = df[column_name].iloc[row]
            if row_string is not None:
                codes += 1
                section = sic[row_string].sic_code.alpha_code[0]
                if section not in section_list:
                    section_list.append(section)
            i += 1
        unique_section_list_by_row.append(section_list)
        codes_count.append(codes)
    df["unique_sections"] = unique_section_list_by_row
    df["codes_count"] = codes_count


# %%
possible_sections(data_not_nota)
possible_sections(data_nota)

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

# %% [markdown]
# This suggests that the number of optoins isn't a significant reason for NOTA to be selected.
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
a = 0
for data_row in range(len(data_nota)):
    if len(data_nota["unique_sections"].iloc[data_row]) == 1:
        a += 1

b = 0
for data_row in range(len(data_not_nota)):
    if len(data_not_nota["unique_sections"].iloc[data_row]) == 1:
        b += 1

c = 0
for data_row in range(len(data_nota)):
    if len(data_nota["unique_sections"].iloc[data_row]) != 1:
        c += 1

d = 0
for data_row in range(len(data_not_nota)):
    if len(data_not_nota["unique_sections"].iloc[data_row]) != 1:
        d += 1

# %%
print(a, b)
print(c, d)

# %%
OR = (a * d) / (b * c)

# %%
print(OR)

# %% [markdown]
# OR < 1, which means there is a negative association between homogeneity and failure (NOTA answer), i.e. when potential answers shown to the respondent are from the same section, the respondent is more likely to select an answer associated with a code, not a NOTA.

# %%
# Success rate

same_section = b / (a + b)
multiple_sections = d / (c + d)

# %%
print(f"Success rate when options presented from the same section: {same_section}")
print(
    f"Success rate when options presented from multiple sections: {multiple_sections}"
)

# %% [markdown]
# same section > multiple sections, which suggests that the codes prsented are 84% successful when from the same section, and 77% success when from multiple sections.
#
# All above, suggest that sections are **not** the reason for respondent selecting NOTA.

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
for option_order in range(len(time_data_nota)):
    time_nota += (
        time_data_nota["time_end"].iloc[option_order]
        - time_data_nota["time_start"].iloc[option_order]
    )
print(
    f"Average time for finishing a survey when NOTA is selected {time_nota / len(time_data_nota)}"
)

# %%
time_not_nota = pd.Timedelta(0)
for option_order in range(len(time_data_not_nota)):
    time_not_nota += (
        time_data_not_nota["time_end"].iloc[option_order]
        - time_data_not_nota["time_start"].iloc[option_order]
    )
print(
    f"Average time for finishing a survey when NOTA is not selected {time_not_nota / len(time_data_not_nota)}"
)

# %%
time_no_closed_question = pd.Timedelta(0)
for option_order in range(len(time_data_no_closed_question)):
    time_no_closed_question += (
        time_data_no_closed_question["time_end"].iloc[option_order]
        - time_data_no_closed_question["time_start"].iloc[option_order]
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
# The median response time when the closed question was asked for both, NOTA and not NOTA answers were very similar (4 seconds difference), suggesting that respondents spent the same amount of time reading and considering possible answers; neither NOTA answers came from respondents skipping through, nor the quality of the options was a problem (no options lead to respondents needing to think too much on the answers).

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
possible_sections(data_not_none)

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
