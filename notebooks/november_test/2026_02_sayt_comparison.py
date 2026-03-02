# %%
"""Notebook to compare SAYT vs SurveyAssist on the small sample overlap.

Loads preprocessed data with both clerical and SA codings,
calculates various metrics and visualises them.
Expects environment variable PREPROD_DATA_BUCKET to be set.

Disabled check for too long lines (f strings) and variables names (uppercase for constants)
"""

# pylint: disable=C0301,C0103,R0801,W0104,W0106, C0121
# flake8: noqa: B023, E712

# %%
import os

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from google import genai
from helper_load_data import load_data
from matplotlib_venn import venn3
from sklearn.metrics.pairwise import cosine_similarity

from survey_assist_utils.data_cleaning.sic_codes import (
    get_clean_n_digit_codes,
    get_codability_level,
)

# %%
data_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET") or ""
project_id = dotenv.get_key(".env", "PROJECT_ID") or ""

work_dir = data_bucket + "analysis-interim-results"
out_dir = (
    "data/figures/"  # needs local folder unfortunately, set to None to skip saving
)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

sa_combined_df = load_data(work_dir)

# %%
# get sheet names
id1_df = pd.read_excel(
    work_dir + "/SAYT/PFR-Crossover IDs.xlsx", sheet_name="Matched Crossover IDs"
)
id1_df["user"] = id1_df["ONS ID (Industry) "] + "-01"
sa_cross_df = sa_combined_df[sa_combined_df["user"].isin(id1_df["user"])].reset_index(
    drop=True
)

# %%
sayt_df = pd.read_excel(work_dir + "/SAYT/SAYT_and_SA_crossover_respondent_data.xlsx")
id2_df = pd.read_excel(work_dir + "/SAYT/matched_serials.xlsx")
lookup = (
    id1_df.rename(columns={"SAYT ID": "UAC"})
    .merge(id2_df, on="UAC", how="left")
    .reset_index(drop=True)[["user", "serial_number"]]
    .rename(columns={"serial_number": "UAC"})
)
sayt_df = sayt_df.merge(lookup, on="UAC", how="outer", indicator=True)
print(sayt_df[sayt_df["_merge"] != "both"])

# %%
sayt_cross_df = sayt_df[sayt_df["_merge"] == "both"].reset_index(drop=True)
combined_df = sayt_cross_df.merge(
    sa_cross_df,
    on="user",
    how="inner",
)

print(combined_df.shape)

# %%
#####################
# initial the lines were misaligned so the linkage was wrong, and we used semantic best matching to find the best pairs,
# and then check how many of them are correct based on the true pairs from the combined_df.
######################

# %%
vectoriser = genai.Client(vertexai=True, project=project_id, location="europe-west1")


def _embed_by_chunks(contents: list[str], chunk_size: int = 150):
    embeddings: list[genai.types.ContentEmbedding] = []
    for i in range(0, len(contents), chunk_size):
        chunk = contents[i : i + chunk_size]
        embed_response = vectoriser.models.embed_content(
            model="text-embedding-005",
            contents=chunk,  # type: ignore[arg-type]
            config=genai.types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        embeddings.extend(embed_response.embeddings)  # type: ignore[arg-type]
    return embeddings


def get_str_dist_matrix(col1: pd.Series, col2: pd.Series):
    """Get distance matrix between two string columns using embeddings cosine similarity.

    Args:
        col1: first string column
        col2: second string column
    Return:
        distance matrix
    """
    col1_embed = _embed_by_chunks(col1.tolist())
    col2_embed = _embed_by_chunks(col2.tolist())
    col1_vecs = np.array([e.values for e in col1_embed])
    col2_vecs = np.array([e.values for e in col2_embed])
    dist_mat = cosine_similarity(col1_vecs, col2_vecs)
    return dist_mat


# %%
dist_matrix = get_str_dist_matrix(
    sa_cross_df["job_title"], sayt_cross_df["SOC_2020_pt1"]
) + 0.5 * get_str_dist_matrix(
    sa_cross_df["job_description"], sayt_cross_df["SOC_2020_pt2"]
)


# %%
def best_pairwise_matching(dist_mat):
    """Get best pairwise matching between two sets based on distance matrix.

    Args:
        dist_mat: distance matrix between two sets
    Return:
        list of pairs of indices
    """
    sa_indices = set(range(dist_mat.shape[0]))
    sayt_indices = set(range(dist_mat.shape[1]))
    pairs = []
    while sa_indices and sayt_indices:
        max_idx = np.unravel_index(np.argmax(dist_mat, axis=None), dist_mat.shape)
        pairs.append((max_idx[0], max_idx[1], dist_mat[max_idx]))
        sa_indices.remove(max_idx[0])
        sayt_indices.remove(max_idx[1])
        dist_mat[max_idx[0], :] = -1
        dist_mat[:, max_idx[1]] = -1
    return pairs


# %%
best_matching = best_pairwise_matching(dist_matrix.copy())
matched_sa_indices = [p[0] for p in best_matching]
matched_sayt_indices = [p[1] for p in best_matching]
matched_sa_df = sa_cross_df.iloc[matched_sa_indices].reset_index(drop=True)
matched_sayt_df = sayt_cross_df.iloc[matched_sayt_indices].reset_index(drop=True)
matched_df = pd.concat(
    [matched_sayt_df.rename(columns={"user": "user_sayt"}), matched_sa_df], axis=1
)

# %%
true_pairs = [(row["user"], row["UAC"]) for _, row in combined_df.iterrows()]
my_pairs = [
    (sa_cross_df.iloc[x[0]]["user"], int(sayt_cross_df.iloc[x[1]]["UAC"]))
    for x in best_matching
]
sum(pair in true_pairs for pair in my_pairs) / len(true_pairs)
matched_df["is_true_pair"] = [pair in true_pairs for pair in my_pairs]
matched_df["matching_score"] = [p[2] for p in best_matching]

# %%
fig = px.scatter(
    matched_df,
    y="matching_score",
    color="is_true_pair",
    title="Best pairwise matching scores between SAYT and SA job titles/descriptions",
    template="plotly_white",
    hover_data=[
        "user",
        "UAC",
        "job_title",
        "SOC_2020_pt1",
        "job_description",
        "SOC_2020_pt2",
    ],
)
fig.show()

# print few mismatches for inspection
matched_df.loc[
    matched_df.is_true_pair == False,
    ["job_title", "SOC_2020_pt1", "job_description", "SOC_2020_pt2", "matching_score"],
].head()

# %%
###############
# after update, we got the corrected matching IDs, so use the from now on
###############


# %%
def get_sayt_codes(input_str):
    """Get SAYT codes from input string."""
    if pd.isna(input_str):
        return set()
    stripped = input_str.strip().replace(".", "")

    for i in range(5):
        good, bad = get_clean_n_digit_codes(stripped[: (6 - i)] + "x" * i, n=5)
        if len(bad) == 0 and len(good) > 0:
            return good
    return {}


combined_df["sayt_codes"] = combined_df["SAYT_code"].apply(get_sayt_codes)

# %%
combined_df["SAYT_code"].apply(lambda x: len(x) if pd.notna(x) else 0).value_counts()
combined_df["sayt_codability_level"] = combined_df["sayt_codes"].apply(
    get_codability_level
)
combined_df["sayt_codability_level"].value_counts()

# %%
combined_df["sa_final_codability_level_closed_q"].value_counts()

# %%
combined_df.groupby(
    ["sa_final_codability_level_closed_q", "sayt_codability_level"]
).size().unstack(fill_value=0)

# %%
# calculate agreement for different SIC levels
for digits in [0, 2, 5]:
    combined_df[f"sayt_codes_{digits}_digit"] = combined_df["sayt_codes"].apply(
        lambda x, n=digits: get_clean_n_digit_codes(x, n=n)[0]
    )
    combined_df[f"sa_final_codes_closed_q_{digits}_digit"] = combined_df[
        "sa_final_codes_closed_q"
    ].apply(lambda x, n=digits: get_clean_n_digit_codes(x, n=n)[0])
    combined_df[f"both_codable_{digits}_digit"] = combined_df[
        f"sayt_codes_{digits}_digit"
    ].apply(lambda x: len(x) == 1) & combined_df[
        f"sa_final_codes_closed_q_{digits}_digit"
    ].apply(
        lambda x: len(x) == 1
    )
    combined_df[f"cc_codes_{digits}_digit"] = combined_df["cc_initial_codes"].apply(
        lambda x: get_clean_n_digit_codes(x, n=digits)[0]
    )
    combined_df[f"sayt_closed_match_{digits}_digit"] = (
        combined_df[f"sayt_codes_{digits}_digit"]
        == combined_df[f"sa_final_codes_closed_q_{digits}_digit"]
    )
    combined_df[f"sayt_in_cc_{digits}_digit"] = combined_df.apply(
        lambda row: len(
            row[f"sayt_codes_{digits}_digit"].intersection(
                row[f"cc_codes_{digits}_digit"]
            )
        )
        > 0,
        axis=1,
    )
    combined_df[f"closed_in_cc_{digits}_digit"] = combined_df.apply(
        lambda row: len(
            row[f"sa_final_codes_closed_q_{digits}_digit"].intersection(
                row[f"cc_codes_{digits}_digit"]
            )
        )
        > 0,
        axis=1,
    )

    combined_df[combined_df[f"both_codable_{digits}_digit"]].groupby(
        [
            f"sayt_closed_match_{digits}_digit",
            f"closed_in_cc_{digits}_digit",
            f"sayt_in_cc_{digits}_digit",
        ]
    ).size()

# %%
for digits in [0, 2, 5]:
    msk = combined_df[f"both_codable_{digits}_digit"]
    subsets = {}
    subsets["111"] = sum(
        msk
        & (combined_df[f"sayt_closed_match_{digits}_digit"] == True)
        & (combined_df[f"closed_in_cc_{digits}_digit"] == True)
    )
    subsets["110"] = sum(
        msk
        & (combined_df[f"sayt_closed_match_{digits}_digit"] == True)
        & (combined_df[f"closed_in_cc_{digits}_digit"] == False)
    )
    subsets["101"] = sum(
        msk
        & (combined_df[f"sayt_closed_match_{digits}_digit"] == False)
        & (combined_df[f"sayt_in_cc_{digits}_digit"] == True)
    )
    subsets["011"] = sum(
        msk
        & (combined_df[f"sayt_closed_match_{digits}_digit"] == False)
        & (combined_df[f"closed_in_cc_{digits}_digit"] == True)
    )
    subsets["100"] = sum(
        msk
        & (combined_df[f"sayt_closed_match_{digits}_digit"] == False)
        & (combined_df[f"sayt_in_cc_{digits}_digit"] == False)
    )
    subsets["010"] = sum(
        msk
        & (combined_df[f"sayt_closed_match_{digits}_digit"] == False)
        & (combined_df[f"closed_in_cc_{digits}_digit"] == False)
    )
    subsets["001"] = sum(msk) - subsets["111"] - subsets["101"] - subsets["011"]
    venn3(
        subsets=subsets,
        set_labels=(
            f"SAYT (codable at {digits} digits)",
            "SA (closed q codable)",
            "CC (standard tlfs, OM plausible)",
        ),
    )

    plt.show()

# %%
# most frequent 2-digit disagreements between SA and SAYT
digits = 2
msk = (
    combined_df[f"both_codable_{digits}_digit"]
    & ~combined_df[f"sayt_closed_match_{digits}_digit"]
)
disagree_2_digit = combined_df[msk].copy()
disagree_2_digit["sayt_str"] = disagree_2_digit[f"sayt_codes_{digits}_digit"].apply(
    lambda x: next(iter(x)) if len(x) == 1 else None
)
disagree_2_digit["sa_str"] = disagree_2_digit[
    f"sa_final_codes_closed_q_{digits}_digit"
].apply(lambda x: next(iter(x)) if len(x) == 1 else None)
aggr = (
    disagree_2_digit.groupby(["sayt_str", "sa_str"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
aggr.head(10)

# %%
examples = combined_df[
    (combined_df[f"sayt_codes_{digits}_digit"] == {aggr.iloc[0]["sayt_str"]})
    & (
        combined_df[f"sa_final_codes_closed_q_{digits}_digit"]
        == {aggr.iloc[0]["sa_str"]}
    )
]
examples[
    [
        "SOC_2020_pt1",
        "SOC_2020_pt2",
        "SAYT_selection",
        "sayt_codes",
        "job_title",
        "job_description",
        "org_description",
        "survey_assist_closed_question_response",
        "sa_final_codes_closed_q",
        "clerical_code_initial",
    ]
].head(5)

# %%
combined_df["SAYT_selection"] = combined_df["SAYT_selection"].astype(str)

# %%
# combined_df.to_parquet(work_dir + "/SAYT/sayt_sa_comparison_all_columns.parquet")

# %%
# combined_df[
#     [
#         "UAC",
#         "user",
#         "SAYT_code",
#         "sayt_codes",
#         "sa_initial_codes",
#         "cc_initial_codes",
#         "sa_final_codes_open_q",
#         "cc_final_codes_open_q",
#         "sa_final_codes_closed_q",
#         "sayt_codes_0_digit",
#         "sa_final_codes_closed_q_0_digit",
#         "both_codable_0_digit",
#         "cc_codes_0_digit",
#         "sayt_closed_match_0_digit",
#         "sayt_in_cc_0_digit",
#         "closed_in_cc_0_digit",
#         "sayt_codes_2_digit",
#         "sa_final_codes_closed_q_2_digit",
#         "both_codable_2_digit",
#         "cc_codes_2_digit",
#         "sayt_closed_match_2_digit",
#         "sayt_in_cc_2_digit",
#         "closed_in_cc_2_digit",
#         "sayt_codes_5_digit",
#         "sa_final_codes_closed_q_5_digit",
#         "both_codable_5_digit",
#         "cc_codes_5_digit",
#         "sayt_closed_match_5_digit",
#         "sayt_in_cc_5_digit",
#         "closed_in_cc_5_digit",
#     ]
# ].to_csv(work_dir + "/SAYT/sayt_sa_comparison.csv", index=False)
