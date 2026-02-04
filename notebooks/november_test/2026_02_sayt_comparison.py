"""Notebook to compare SAYT vs SurveyAssist on the small sample overlap.

Loads preprocessed data with both clerical and SA codings,
calculates various metrics and visualises them.
Expects environment variable PREPROD_DATA_BUCKET to be set.

Disabled check for too long lines (f strings) and variables names (uppercase for constants)
"""

# pylint: disable=C0301,C0103,R0801,W0104,W0106

# %%
import os

import dotenv
import numpy as np
import pandas as pd
import plotly.express as px
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

from notebooks.november_test.helper_load_data import load_data

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
id1_df = pd.read_excel(work_dir + "/SAYT/PFR-Crossover IDs - UPDATED 26.01.26.xlsx")
id1_df["user"] = id1_df["ONS Participant ID"] + "-01"
sa_cross_df = sa_combined_df[sa_combined_df["user"].isin(id1_df["user"])].reset_index(
    drop=True
)

# %%
sayt_df = pd.read_excel(work_dir + "/SAYT/SAYT_and_SA_crossover_respondent_data.xlsx")
id2_df = pd.read_excel(work_dir + "/SAYT/matched_serials.xlsx")
lookup = (
    id1_df.rename(columns={"SAYT ID ": "UAC"})
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
match_term = "university professor"
# match_term = 'director'
combined_df.loc[
    (combined_df["job_title"].str.lower() == match_term)
    | (combined_df["SOC_2020_pt1"].str.lower() == match_term),
    ["job_title", "SOC_2020_pt1", "job_description", "SOC_2020_pt2"],
]

#####################
# so we see same people buton different rows - so the linkage is wrong, lets do it ourselves
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
matched_combined_df = pd.concat([matched_sayt_df, matched_sa_df], axis=1)

matched_combined_df.loc[
    100:220, ["job_title", "SOC_2020_pt1", "job_description", "SOC_2020_pt2"]
]

# %%
fig = px.line(
    [p[2] for p in best_matching],
    title="Best pairwise matching scores between SAYT and SA job titles/descriptions",
    template="plotly_white",
)
fig.show()

# %%
