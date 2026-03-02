# %%
"""Additional checks for SAYT, SA and CC overlap and lack of overlap.

Disabled:
    - Line too long: print statements.
    - Constant name: naming style.
"""

# pylint: disable= C0103, C0301

# %%
import dotenv
import pandas as pd

# %%
preprod_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET")
if not preprod_bucket:
    raise ValueError("PREPROD_DATA_BUCKET not found in .env file. Please set it.")

# %%
data = pd.read_parquet(
    f"{preprod_bucket}analysis-interim-results/SAYT/sayt_sa_comparison_all_columns.parquet"
)

# %%
digits = [0, 2, 5]


# %%
def convert_to_set(row: pd.Series):
    """Converts array to set.

    Args:
        row: pd.Series: a row with values as array

    Return:
        set
    """
    return set(row.flatten())


# %%
columns_to_change = [
    "sayt_codes",
    "sa_initial_codes",
    "cc_initial_codes",
    "sa_final_codes_open_q",
    "cc_final_codes_open_q",
    "sa_final_codes_closed_q",
    "sayt_codes_0_digit",
    "sa_final_codes_closed_q_0_digit",
    "cc_codes_0_digit",
    "sayt_codes_2_digit",
    "sa_final_codes_closed_q_2_digit",
    "cc_codes_2_digit",
    "sayt_codes_5_digit",
    "sa_final_codes_closed_q_5_digit",
    "cc_codes_5_digit",
]

# %%
for column in columns_to_change:
    data.loc[:, column] = data[column].apply(convert_to_set)

# %%
len_data = len(data)

# %%
# SA and SAYT have same code at i-digits

for i in digits:
    msk = data[f"both_codable_{i}_digit"]

    sa_final = data[f"sa_final_codes_closed_q_{i}_digit"]
    sayt_final = data[f"sayt_codes_{i}_digit"]

    sayt_sa_codable = len(data[msk])
    sayt_sa_codable_agreement = len(data[msk & (sa_final == sayt_final)])

    agreement_percentage = sayt_sa_codable_agreement / sayt_sa_codable * 100

    print(
        f"SAYT and SA agreement at {i}-digits: {sayt_sa_codable_agreement} ({agreement_percentage:.1f}%)"
    )

# %%
# SA and SAYT have different codes at i-digits

for i in digits:
    msk = data[f"both_codable_{i}_digit"]

    sa_final = data[f"sa_final_codes_closed_q_{i}_digit"]
    sayt_final = data[f"sayt_codes_{i}_digit"]

    sayt_sa_codable = len(data[msk])
    sayt_sa_codable_disagreement = len(data[msk & (sa_final != sayt_final)])

    disagreement_percentage = sayt_sa_codable_disagreement / sayt_sa_codable * 100

    print(
        f"SAYT and SA disagreement at {i}-digits: {sayt_sa_codable_disagreement} ({disagreement_percentage:.1f}%)"
    )

# %%
# SA codable, SAYT has multiple codes at 5 digit level (i.e. respondent selected division)
msk = data["sayt_codes_5_digit"].str.len() > 1
sa_final = data["sa_final_codes_closed_q_5_digit"].str.len() == 1

sa_codable_sayt_not_codable = len(data[msk & sa_final])

sa_codable_sayt_not_codable_percentage = (
    sa_codable_sayt_not_codable / sayt_sa_codable * 100
)

print(f"SA codable, SAYT not codable at 5-digits: {sa_codable_sayt_not_codable}")

# %%
# SAYT and closed match (same code or both not codable)

for i in digits:
    column = f"sayt_closed_match_{i}_digit"
    sayt_respondent_agreement = data[column].value_counts()[True]
    print(
        f"SAYT and SA matches at {i}-digit level {sayt_respondent_agreement} ({sayt_respondent_agreement / len_data * 100:.1f}% of all responses)"
    )

# %%
# SAYT and cc agreement and codable

for i in digits:
    column_cc = f"cc_codes_{i}_digit"
    column_sayt = f"sayt_codes_{i}_digit"

    msk = data[column_sayt].str.len() > 0

    cc_sayt_agreement = len(data[msk & (data[column_cc] == data[column_sayt])])
    print(
        f"SAYT and CC agreement at {i}-digit level: {cc_sayt_agreement} ({cc_sayt_agreement/ len_data * 100:.1f}%)"
    )

# %%
# SA and cc agreement and codable

for i in digits:
    column_cc = f"cc_codes_{i}_digit"
    column_respondent = f"sa_final_codes_closed_q_{i}_digit"

    msk = data[column_respondent].str.len() > 0

    cc_respondent_agreement = len(
        data[msk & (data[column_cc] == data[column_respondent])]
    )

    print(
        f"Respondent (SA) and CC agreement on {i}-digit level: {cc_respondent_agreement} ({cc_respondent_agreement/ len_data * 100:.1f}%)"
    )

# %%
# SA in CC shortlist

for i in digits:
    column_closed_in_cc = f"closed_in_cc_{i}_digit"
    closed_in_cc = data[column_closed_in_cc].value_counts()[True]
    print(
        f"Respondent (SA) selection within CC shortlist at {i}-digit level: {closed_in_cc} ({closed_in_cc / len_data * 100:.1f}%)"
    )

# %%
# SAYT in CC shortlist

for i in digits:
    column_sayt_in_cc = f"sayt_in_cc_{i}_digit"
    sayt_in_cc = data[column_sayt_in_cc].value_counts()[True]
    print(
        f"SAYT selection in CC shortlist at {i}-digit level: {sayt_in_cc} ({sayt_in_cc / len_data * 100:.1f}%)"
    )

# %%
# SAYT, SA and CC codable to the same code (unambiguously)

for i in digits:
    sayt_column = f"sayt_codes_{i}_digit"
    respondent_column = f"sa_final_codes_closed_q_{i}_digit"
    cc_column = f"cc_codes_{i}_digit"

    msk = data[sayt_column].str.len() > 0

    sayt_respondednt_agreement = data[sayt_column] == data[respondent_column]
    sayt_cc_agreement = data[sayt_column] == data[cc_column]

    sayt_cc_respondent_agreemnt = len(
        data[(msk) & (sayt_respondednt_agreement) & (sayt_cc_agreement)]
    )

    print(
        f"SAYT, CC and Respondent (SA) agreement at {i}-digit level: {sayt_cc_respondent_agreemnt} ({sayt_cc_respondent_agreemnt / len_data * 100:.1f}%)"
    )

# %%
# Neither SA or SAYT returned any codes
for i in digits:
    sa_not_codable = data[f"sa_final_codes_closed_q_{i}_digit"] == set()
    sayt_not_codable = data[f"sayt_codes_{i}_digit"] == set()
    msk_not_codable = data[sa_not_codable & sayt_not_codable]
    not_codable_count = len(msk_not_codable)
    print(
        f"Respondent (SA) did not selected a code, nor SAYT: {not_codable_count} ({round(not_codable_count/len_data * 100,1)}%)"
    )

# %%
# SA respondent is not codable, SAYT is codable

for i in digits:
    sa_not_codable = data[f"sa_final_codes_closed_q_{i}_digit"] == set()
    sayt_codable = data[f"sayt_codes_{i}_digit"].str.len() == 1
    msk_sa_not_codable = data[sa_not_codable & sayt_codable]
    sa_not_codable = len(msk_sa_not_codable)
    print(
        f"Respondent (SA) did not selected a code, but SAYT: {sa_not_codable} ({round(sa_not_codable/len_data * 100,1)}%)"
    )

# %%
# SA respondent selected a code, SAYT is not codable

for i in digits:
    sa_codable = data[f"sa_final_codes_closed_q_{i}_digit"].str.len() == 1
    sayt_not_codable = data[f"sayt_codes_{i}_digit"] == set()
    msk_sayt_not_codable = data[sa_codable & sayt_not_codable]
    sayt_not_codable = len(msk_sayt_not_codable)
    print(
        f"Respondent selected a code, but not SAYT: {sayt_not_codable} ({round(sayt_not_codable/len_data * 100,1)}%)"
    )

# %%
# Most likely SIC section for those that SA and SAYT are uncodable

sa_not_codable = data["sa_final_codes_closed_q_5_digit"] == set()
sayt_not_codable = data["sayt_codes_5_digit"].str.len() != 1

sections_not_codable = data[sa_not_codable & sayt_not_codable][
    "most_likely_sic_section"
]

print(
    f"Most likely SIC section when SA and SAYT not codable at 5-digit\n{sections_not_codable.value_counts()}"
)


# %%
def sa_sayt_codable(row: pd.Series):
    """Check if is codable to a single code.

    Args:
        row (pd.Series): row with set of codes selected.

    Return:
        bool: True when codable, False when not codable to single code.
    """
    return row != set()


# %%
# Prepare data for creating a confusion matrix for the final codability of SA and SAYT

data.loc[:, "sa_final_5_digit_codable"] = data["sa_final_codes_closed_q_5_digit"].apply(
    sa_sayt_codable
)
data.loc[:, "sayt_final_5_digit_codable"] = data["sayt_codes_5_digit"].apply(
    sa_sayt_codable
)

# %%
final_sa_sayt_confusion_matrix = pd.crosstab(
    data["sayt_final_5_digit_codable"],
    data["sa_final_5_digit_codable"],
    rownames=["SAYT"],
    colnames=["SA"],
)

# %%
print(f"Final codability\n{final_sa_sayt_confusion_matrix}")

# %% [markdown]
# In 13 cases both SA ans SAYT are not codable to final code.
#
# 184 are codable: This includes selecting different codes and SAYT codable to multiple codes.
#
# 26 respondents selected one of the titles (including 2- and 5-digit codes), but selected "None of the above " in Closed Quesiton.
#
# 28 respondents were able to find final 5-digit code in Closde Question, but didn't select any in SAYT.
