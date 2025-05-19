# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: survey-assist-utils-PWI-TvqZ-py3.12
#     language: python
#     name: python3
# ---

# %%

"""Summary analysis of the TLFS_evaluation_data_IT2 dataset,
    prepared via `prepare_evaluation_data_for_analysis`.

Key outputs include:
- Coverage of 2-digit SIC groups
- Distribution of possible SIC codes per record
- Codeability and unambiguous coding stats at 2-digit and 5-digit levels
- Analysis of derived SIC from `sic_ind_occ1`
- Identification of uncodeable records and those requiring follow-up

Developed in a notebook and converted to script for version control and reproducibility.
"""


import pandas as pd

cols_wanted = [
    "unique_id",
    "sic_ind_occ1",
    "sic_ind_occ2",
    "sic_ind_occ3",
    "sic_ind_occ_flag",
    "Not_Codeable",
    "Four_Or_More",
    "SIC_Division",
    "num_answers",
    "All_Clerical_codes",
    "Match_5_digits",
    "Match_3_digits",
    "Match_2_digits",
    "Unambiguous",
]

# Load your DataFrame
eval_data = pd.read_csv(
    "../data/analysis_outputs/TLFS_evaluation_data_IT2_output.csv",
    usecols=cols_wanted,
    dtype={"SIC_Division": str},
)
