# %%
"""Work in progess.

Initial analysis of survey responses, focusing on Clased Follow up questions.

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
from scipy import stats

# %%
evaluation_bucket = dotenv.get_key(".env", "EVALUATION_BUCKET")
analysis_bucket = dotenv.get_key(".env", "ANALYSIS_BUCKET")
if not evaluation_bucket:
    raise ValueError("EVALUATION_BUCKET not found in .env file. Please set it.")
if not analysis_bucket:
    raise ValueError("ANALYSIS_BUCKET not found in .env file. Please set it.")

# %%
data = pd.read_parquet(f"{analysis_bucket}evaluation_df_with_sa_clean_codes.parquet")

# %%
column = "survey_assist_closed_question_option_"
selected_response = []

# iterate through all rows of the data
for row in range(len(data)):
    # save the response selected by the user
    response = data["survey_assist_closed_question_response"][row]

    # check only when closed question was asked
    if response is not None and response != "none of the above":

        # change options from closed questions to lower case (matching the selected response)
        for i in range(1, 7):
            column_name = column + str(i)
            if data[column_name][row] is not None:
                data.loc[row, column_name] = data[column_name][row].lower()

        # find the order of the selected response
        j = 1
        while j < 7:
            column_name = column + str(j)
            if response == data[column_name][row]:
                selected_response.append(j)
                j = 7
            j += 1

# %%
# percentage of codes found using closed quesiton
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
print(
    "1st option [%]:",
    round(100 * selected_response_order["option_1"] / len(selected_response), 2),
)
print(
    "2nd option [%]:",
    round(100 * selected_response_order["option_2"] / len(selected_response), 2),
)
print(
    "3rd option [%]:",
    round(100 * selected_response_order["option_3"] / len(selected_response), 2),
)
print(
    "4th option [%]:",
    round(100 * selected_response_order["option_4"] / len(selected_response), 2),
)
print(
    "5th option [%]:",
    round(100 * selected_response_order["option_5"] / len(selected_response), 2),
)
print(
    "6th option [%]:",
    round(100 * selected_response_order["option_6"] / len(selected_response), 2),
)

# %% [markdown]
# Note, "none of the above" is always presented at the bottom of the list, so 6th option never brings a code. Options 1 and 2 will always have possible codes.

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
sic_rephrased.sample()

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
closed_q_data = data[
    [
        "unique_id",
        "user",
        "job_title",
        "job_description",
        "org_description",
        "survey_assist_closed_question_response",
        "survey_assist_closed_question_option_1",
        "survey_assist_closed_question_option_2",
        "survey_assist_closed_question_option_3",
        "survey_assist_closed_question_option_4",
        "survey_assist_closed_question_option_5",
        "survey_assist_closed_question_option_6",
        "survey_assist_alt_candidate_code_1",
        "survey_assist_alt_candidate_code_2",
        "survey_assist_alt_candidate_code_3",
        "survey_assist_alt_candidate_code_4",
        "survey_assist_alt_candidate_code_5",
    ]
]

# %%
sic_dictionary = sic_rephrased["input_description"].apply(convert_to_dict)
none_of_the_above = 0

options: dict[str, list[str | None]] = {"1": [], "2": [], "3": [], "4": [], "5": []}

for row in range(len(closed_q_data)):
    # if the closed question response is None, it means the question was not asked, and there's no codes.
    if closed_q_data["survey_assist_closed_question_response"][row] is not None:

        i = 1
        while i < 6:
            current_row = closed_q_data[f"survey_assist_closed_question_option_{i}"][
                row
            ].lower()
            if current_row == "none of the above":
                none_of_the_above += 1
                while i < 6:
                    options[f"{i}"].append(None)
                    i += 1

            else:
                if current_row in list(sic_rephrased["reviewed_description"]):
                    sic_code = sic_rephrased[
                        sic_rephrased["reviewed_description"] == current_row
                    ]["input_code"].item()
                    options[f"{i}"].append(str(sic_code))

                else:
                    sic_code = get_code_by_title(sic_dictionary, current_row)
                    options[f"{i}"].append(str(sic_code))

                i += 1
    else:
        options["1"].append(None)
        options["2"].append(None)
        options["3"].append(None)
        options["4"].append(None)
        options["5"].append(None)

# %%
for k in range(1, 6):
    closed_q_data[f"survey_assist_closed_question_option_{k}_code"] = options[f"{k}"]

# %%
response_codes = []
for row in range(len(closed_q_data)):
    response = closed_q_data["survey_assist_closed_question_response"][row]
    if response is None or response == "none of the above":
        response_codes.append("None")
    else:
        i = 1
        while i < 6:
            option = "survey_assist_closed_question_option_" + str(i)
            option_code = "survey_assist_closed_question_option_" + str(i) + "_code"
            if response == closed_q_data[option][row]:
                response_codes.append(closed_q_data[option_code][row])
                i = 6
            else:
                i += 1

# %%
# we get "None" when the closed question was not asked or the answer was "none of the above"
response_codes.count("None")

# %%
len(response_codes)

# %%
# asked closed question, but didn't get a code
round(100 * none_of_the_above / len(selected_response), 2)

# %%
closed_q_data["survey_assist_closed_question_response_code"] = response_codes

# %%
# closed_q_data.to_parquet(
#     f"{analysis_bucket}closed_questions/closed_questions_codes.parquet", index=False
# )

# %%
options.keys()

# %%
options_df = pd.DataFrame(options)

# %%
alt_codes_count = []
for i in range(len(closed_q_data)):
    responses = int(options_df.iloc[i].count())
    if (
        responses != 0
        and closed_q_data["survey_assist_closed_question_response"][i]
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
    """df: dataframe
    response_column: a column with the respondends choice recorded as a rank
    alt_codes_column: count of alternative codes
    n: n-th option to be tested.
    """
    # The first option was selected, where 'none of the above' was NOT selected
    N_P1 = df[df[response_column] == n].shape[0]

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
sa_code_match: list[int | str] = []
survey_assist_alt = "survey_assist_alt_candidate_code_"
for row in range(len(closed_q_data)):
    i = 1
    while i < 6:
        sa_code = survey_assist_alt + str(i)
        if closed_q_data["survey_assist_closed_question_response_code"][row] is None:
            # sa_code_match.append(0)
            i = 6
        elif (
            closed_q_data[sa_code][row]
            == closed_q_data["survey_assist_closed_question_response_code"][row]
        ):
            sa_code_match.append(i)
            i = 6
        i += 1
    if i == 6:
        sa_code_match.append("None")


# %%
sa_alt_codes_count = []
survey_assist_alt = "survey_assist_alt_candidate_code_"
for row in range(len(closed_q_data)):
    alts = 0
    i = 1
    while i < 6:
        sa_code = survey_assist_alt + str(i)
        if closed_q_data["survey_assist_closed_question_response_code"][row] is None:
            i = 6
        elif closed_q_data[sa_code][row] is not None:
            alts += 1
            # i = 6
        i += 1
    # if alts != 0:
    sa_alt_codes_count.append(alts)

# %%
len(sa_alt_codes_count)

# %%
print(sa_code_match.count(0))
print(sa_code_match.count(1))
print(sa_code_match.count(2))
print(sa_code_match.count(3))
print(sa_code_match.count(4))
print(sa_code_match.count(5))
print(sa_code_match.count("None"))

# %%
print(sa_alt_codes_count.count(0))
print(sa_alt_codes_count.count(1))
print(sa_alt_codes_count.count(2))
print(sa_alt_codes_count.count(3))
print(sa_alt_codes_count.count(4))
print(sa_alt_codes_count.count(5))

# %%
sa_codes = {"selected_code": sa_code_match, "alt_codes_count": sa_alt_codes_count}
df_sa_codes = pd.DataFrame(sa_codes)

# %%
print("1:", chi2_test(df_sa_codes, "selected_code", "alt_codes_count", 1))
print("2:", chi2_test(df_sa_codes, "selected_code", "alt_codes_count", 2))
print("3:", chi2_test(df_sa_codes, "selected_code", "alt_codes_count", 3))
print("4:", chi2_test(df_sa_codes, "selected_code", "alt_codes_count", 4))
print("5:", chi2_test(df_sa_codes, "selected_code", "alt_codes_count", 5))

# %%
df_sa_codes_no_none = df_sa_codes[df_sa_codes["selected_code"] != "None"]

# %%
print("alt codes count x rank of selected response\n")
df_group_sa = df_sa_codes_no_none[df_sa_codes_no_none["alt_codes_count"] == 2]
print(df_group_sa.shape[0])
print("2x1:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 1))
print("2x2:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 2), "\n")

df_group_sa = df_sa_codes_no_none[df_sa_codes_no_none["alt_codes_count"] == 3]
print(df_group_sa.shape[0])
print("3x1:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 1))
print("3x2:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 2))
print("3x3:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 3), "\n")

df_group_sa = df_sa_codes_no_none[df_sa_codes_no_none["alt_codes_count"] == 4]
print(df_group_sa.shape[0])
print("4x1:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 1))
print("4x2:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 2))
print("4x3:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 3))
print("4x4:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 4), "\n")

df_group_sa = df_sa_codes_no_none[df_sa_codes_no_none["alt_codes_count"] == 5]
print(df_group_sa.shape[0])
print("5x1:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 1))
print("5x2:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 2))
print("5x3:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 3))
print("5x4:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 4))
print("5x5:", chi2_test(df_group_sa, "selected_code", "alt_codes_count", 5))
