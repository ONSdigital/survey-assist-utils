"""Notebook to visualise the codability gain/loss using a Sankey diagram."""

# pylint: disable=C0301,C0103,R0801
# %%
import dotenv
import pandas as pd

from survey_assist_utils.data_cleaning.prep_data import prep_clerical_codes

data_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET") or ""

# %%
work_dir = data_bucket + "analysis-interim-results/clerically-coded/"

# %%
file_names = [
    "SA_Clerical_coding_batch1_file_1_2025-11-27-public-beta.xlsx",
    "SA_Clerical_coding_batch2_file_1_2025-12-02-public-beta.xlsx",
]
df_list = []
for file_name in file_names:
    df_list.append(pd.read_excel(work_dir + file_name, dtype=str))
initial_cc_df = pd.concat(df_list, axis=0, ignore_index=True)
initial_cc_df["sic_ind_occ1"] = initial_cc_df["Clerical code"]

clerical_codes_df = prep_clerical_codes(initial_cc_df)
# %%
# deal with invalid codes manually
replace_dict = {
    "62xxx, 63xxx, 95100": "62xxx, 63xxx, 951xx",
    "86xx": "86xxx",
    "71229": "71129",
}
initial_cc_df["sic_ind_occ1"] = initial_cc_df["sic_ind_occ1"].replace(replace_dict)
clerical_codes_df["clerical_codes"] = prep_clerical_codes(initial_cc_df)[
    "clerical_codes"
]

# %%
initial_cc_df.merge(clerical_codes_df).to_parquet(
    work_dir + "clerical_codes_df_cleaned_initial_codes.parquet"
)

# %%
