"""Notebook to ingest clerical coded data from November test.

Loads clerical coding excel files from preprocessed data bucket,
cleans and processes the clerical codes, assigns codability levels and
calculates codability gain.

Expects environment variable PREPROD_DATA_BUCKET to be set.

Disabled check for too long lines (f strings) and variables names (uppercase for constants)
"""

# pylint: disable=C0301,C0103,R0801
# %%
import dotenv
import pandas as pd

from survey_assist_utils.data_cleaning.prep_data import prep_clerical_codes
from survey_assist_utils.data_cleaning.sic_codes import (
    asses_codability_gain,
    get_codability_level,
)

# %%
data_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET") or ""
work_dir = data_bucket + "analysis-interim-results/clerically-coded/"


# %%
def excel_columns_clean(in_df: pd.DataFrame) -> pd.DataFrame:
    """Clean excel columns by stripping whitespace from column names.

    Args:
        in_df: DataFrame with columns to be cleaned.

    Returns:
        DataFrame with cleaned column names.
    """
    in_df.columns = in_df.columns.str.strip().str.lower().str.replace(" ", "_")
    # drop empty columns
    in_df = in_df.loc[:, ~in_df.isna().all(axis=0)]
    return in_df


# %%
file_names = {
    "initial": [
        "SA_Clerical_coding_batch1_file_1_2025-11-27-public-beta.xlsx",
        "SA_Clerical_coding_batch2_file_1_2025-12-02-public-beta.xlsx",
        "SA_Clerical_coding_batch3_file_1_2025-12-09-public-beta.xlsx",
    ],
    "final": [
        "SA_Clerical_coding_batch1_file_2_2025-11-27-public-beta.xlsx",
        "SA_Clerical_coding_batch2_file_2_2025-12-02-public-beta.xlsx",
        "SA_Clerical_coding_batch3_file_2_2025-12-09-public-beta.xlsx",
    ],
}
df_list: dict[str, list[pd.DataFrame]] = {"initial": [], "final": []}
for batch in range(3):
    print(f"Batch {batch+1} clerical codes:")
    for key, file_name in file_names.items():
        try:
            df = excel_columns_clean(
                pd.read_excel(
                    work_dir + file_name[batch],
                    dtype=str,
                )
            ).rename(
                columns={
                    "clerical_code": f"clerical_code_{key}",
                    "qa": f"qa_{key}",
                    "comments": f"comments_{key}",
                    "qa_comments": f"qa_comments_{key}",
                }
            )
            print(f"- {key} codes read for {len(df)} records.")
            df["batch_num"] = batch + 1
            df_list[key].append(df)
        except FileNotFoundError:
            print(f"- {key} file for batch {batch+1} not found.")
            df = pd.DataFrame()

initial_cc_df = pd.concat(df_list["initial"], axis=0, ignore_index=True)
final_cc_df = pd.concat(df_list["final"], axis=0, ignore_index=True).drop(
    columns=["user", "job_title", "job_description", "org_description"], errors="ignore"
)

clerical_codes_df = initial_cc_df.merge(
    final_cc_df,
    on=["unique_id", "batch_num"],
    how="outer",
    suffixes=("_initial", "_final"),
)
print(f"Total records with clerical codes: {len(clerical_codes_df)}")

# %%
# assign and clean initial codes
clerical_codes_df["sic_ind_occ1"] = clerical_codes_df["clerical_code_initial"]
clerical_codes_df = clerical_codes_df.merge(
    prep_clerical_codes(clerical_codes_df, out_col="cc_initial_codes")
)

print(clerical_codes_df[clerical_codes_df["cc_initial_codes_invalid"].apply(len) > 0])

# %%
# deal with invalid codes manually
replace_dict = {
    "62xxx, 63xxx, 95100": "62xxx, 63xxx, 951xx",
    "86xx": "86xxx",
    "71229": "71129",
}
clerical_codes_df["sic_ind_occ1"] = clerical_codes_df["sic_ind_occ1"].replace(
    replace_dict
)

clerical_codes_df["cc_initial_codes"] = prep_clerical_codes(
    clerical_codes_df, out_col="cc_initial_codes"
)["cc_initial_codes"]
clerical_codes_df["cc_initial_codability_level"] = clerical_codes_df[
    "cc_initial_codes"
].apply(get_codability_level)

# %%
# assign and clean final codes
clerical_codes_df["sic_ind_occ1"] = clerical_codes_df["clerical_code_final"]
clerical_codes_df = clerical_codes_df.merge(
    prep_clerical_codes(clerical_codes_df, out_col="cc_final_codes_open_q")
)

print(
    clerical_codes_df[clerical_codes_df["cc_final_codes_open_q_invalid"].apply(len) > 0]
)

# %%
# deal with invalid codes manually
replace_dict = {
    "52301, 52302, 52290": "53201, 53202, 52290",
    "84xx": "84xxx",
}
clerical_codes_df["sic_ind_occ1"] = clerical_codes_df["sic_ind_occ1"].replace(
    replace_dict
)
clerical_codes_df["cc_final_codes_open_q"] = prep_clerical_codes(
    clerical_codes_df, out_col="cc_final_codes_open_q"
)["cc_final_codes_open_q"]
clerical_codes_df.drop(columns=["sic_ind_occ1"], inplace=True)

# %%
# final codes for those without open question coding
mask_no_follow_up = clerical_codes_df["survey_assist_open_question"].isna()

clerical_codes_df.loc[mask_no_follow_up, "cc_final_codes_open_q"] = (
    clerical_codes_df.loc[mask_no_follow_up, "cc_initial_codes"]
)

clerical_codes_df["cc_final_codability_level_open_q"] = clerical_codes_df[
    "cc_final_codes_open_q"
].apply(get_codability_level)
clerical_codes_df["cc_codability_gain_open_q"] = clerical_codes_df.apply(
    asses_codability_gain,
    axis=1,
    initial_level_col="cc_initial_codability_level",
    final_level_col="cc_final_codability_level_open_q",
)

# %%
clerical_codes_df.to_parquet(work_dir + "clerical_df_with_cc_clean_codes.parquet")

# %%
