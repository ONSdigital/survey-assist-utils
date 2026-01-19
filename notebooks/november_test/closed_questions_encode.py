# %%
"""Notebook that converts options presented to (and selected by) respondents back into SIC codes.

Saves to the storage in a location specified in .env file (cell with that functionality
is currently commented out, at the very bottom of this notebook).

Before running the notebook, create .env file with bucket variables, such as
PREPROD_DATA_BUCKET = "gs://<bucket-name>/<folder>/", and EVALUATION_BUCKET similarly.

PREPROD_DATA_BUCKET - location of the input files.
EVALUATION_BUCKET - location where the file is to be saved.
"""

# %%
# pylint: disable=C0103, C0116, C0301, C0114, R0801
# ruff: noqa: PLR2004

import re

# %%
import dotenv
import pandas as pd

# %%
evaluation_bucket = dotenv.get_key(".env", "EVALUATION_BUCKET")
preprod_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET")
if not evaluation_bucket:
    raise ValueError("EVALUATION_BUCKET not found in .env file. Please set it.")
if not preprod_bucket:
    raise ValueError("PREPROD_DATA_BUCKET not found in .env file. Please set it.")

# %%
data = pd.read_parquet(
    f"{preprod_bucket}analysis-interim-results/evaluation_df_with_sa_clean_codes.parquet"
)

# %%
sic_rephrased = pd.read_csv(
    f"{evaluation_bucket}sic_rephrased_descriptions_2025_02_03.csv", dtype=str
)

# %%
# Majority of the sic descriptions come from 'reviewed_description' column, however some come from
# a dictionary, which is saved as 'input_description'. Because the file is saved as a CSV,
# the dictionary is converted into a string. This function allows unpacking sic codes and their
# descriptions from the string.


def convert_to_dict(dict_string):
    clean_string = dict_string.strip().strip("'")

    code = re.search(r"{Code: \s*(.*?),\s*Title: ", clean_string, re.DOTALL)
    code_value = code.group(1).strip() if code else ""

    title = re.search(
        r"Title: \s*(.*?),\s*Example activities: ", clean_string, re.DOTALL
    )
    title_value = title.group(1).strip() if title else ""
    title_value = title_value.lower()

    result_dictionary = {
        title_value: code_value,
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
def get_options_codes(response_row, lookup):

    options_list: dict[str, list[str | None]] = {
        "1": [],
        "2": [],
        "3": [],
        "4": [],
        "5": [],
    }

    response = response_row["survey_assist_closed_question_response"]

    if response is not None:

        k = 1
        while k < 6:
            current_row = response_row[
                f"survey_assist_closed_question_option_{k}"
            ].lower()

            if current_row == "none of the above":
                while k < 6:
                    options_list[f"{k}"].append(None)
                    k += 1

            else:
                sic_code = lookup.get(current_row)
                options_list[f"{k}"].append(str(sic_code))

                k += 1
    else:
        options_list["1"].append(None)
        options_list["2"].append(None)
        options_list["3"].append(None)
        options_list["4"].append(None)
        options_list["5"].append(None)
    return options_list


# %%
# create a lookup using sic description and rephrased sic description. Allow for 'none of the above'

sic_lookup = sic_rephrased["input_description"].apply(convert_to_dict)
sic_lookup = {key: value for d in sic_lookup for key, value in d.items()}
sic_lookup["none of the above"] = None

# %%
rephrased_lookup = sic_rephrased.set_index("reviewed_description")["sic_code"].to_dict()
sic_lookup.update(rephrased_lookup)

# %%
options = pd.DataFrame(
    closed_q_data.apply(get_options_codes, axis=1, args=(sic_lookup,)).to_list()
)
options = options.map(lambda x: x[0] if isinstance(x, list) else x)

# %%
codes_columns = options
codes_columns.columns = [
    f"survey_assist_closed_question_option_{k}_code" for k in codes_columns.columns
]
closed_q_data = pd.concat([closed_q_data, codes_columns], axis=1)


# %%
def get_response_codes(response_row: pd.Series):
    """Get the code corresponding to the option selected by the user.

    Args:
        response_row: row with survey response

    Return:
        responde code
    """
    response = response_row["survey_assist_closed_question_response"]
    if response is None or response.lower() == "none of the above":
        res_codes = "None"
    else:
        k = 1
        while k < 6:
            option = "survey_assist_closed_question_option_" + str(k)
            option_code = "survey_assist_closed_question_option_" + str(k) + "_code"
            if response == response_row[option].lower():
                res_codes = response_row[option_code]
                k = 6
            else:
                k += 1
    return res_codes


# %%
response_codes = closed_q_data.apply(get_response_codes, axis=1)

# %%
# we get "None" when the closed question was not asked or the answer was "none of the above"
print((response_codes.isna()).sum())

# %%
closed_q_data["survey_assist_closed_question_response_code"] = response_codes

# %%
# closed_q_data.to_parquet(
#     f"{preprod_bucket}closed_questions/closed_questions_codes.parquet", index=False
# )
