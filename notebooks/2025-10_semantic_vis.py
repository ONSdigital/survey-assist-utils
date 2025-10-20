"""Work in progress notebook to visualize metrics for different embeddings.

It loads specific clerical coding data and model outputs from bucket.
The bucket name and folder (on line 32) can be manually entered or it is read from
the .env file, where it should be stored as BUCKET_PREFIX variable, i.e.:
BUCKET_PREFIX = "gs://<bucket-name>/<folder>/"

Disabled check for too long lines (f strings) and variables names (uppercase for constants)
"""

# pylint: disable=C0301,C0103,R0801

# %%
import logging
import os

import dotenv
import pandas as pd
import plotly.express as px

from survey_assist_utils.data_cleaning.prep_data import (
    prep_clerical_codes,
    prep_model_codes,
)
from survey_assist_utils.data_cleaning.sic_codes import get_clean_n_digit_one_code
from survey_assist_utils.evaluation.metrics import (
    calc_simple_metrics,
)

# %%
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

bucket_prefix = dotenv.get_key(".env", "BUCKET_PREFIX")

if not bucket_prefix:
    raise ValueError("BUCKET_PREFIX not found in .env file. Please set it.")

output_folder = "data/temp/"  # set to None if no output saving is needed

if output_folder:
    os.makedirs(output_folder, exist_ok=True)

# %%
# load clerical data
clerical_it2_file = f"{bucket_prefix}original_datasets/DSC_Rep_Sample_IT2.csv"
clerical_it2_4plus_file = (
    f"{bucket_prefix}original_datasets/Codes_for_4_plus_DSC_Rep_Sample_IT2.csv"
)
cc_it2_df = pd.read_csv(clerical_it2_file)
cc_it2_4plus_df = pd.read_csv(clerical_it2_4plus_file)


# %%
# load semantic outputs
semantic_files = {
    "minilm-l2": f"{bucket_prefix}semantic_stage/minilm-l6-v2-l2dist/STG1.parquet",
    "te005-clastask-l2": f"{bucket_prefix}semantic_stage/text-embedding-005-clas-l2dist/STG1.parquet",
    "te005-semtask-l2": f"{bucket_prefix}semantic_stage/text-embedding-005-sem-l2dist/STG1.parquet",
    "te005-doctask-l2": f"{bucket_prefix}semantic_stage/text-embedding-005-doc-l2dist/STG1.parquet",
    "te004-semtask-l2": f"{bucket_prefix}semantic_stage/text-embedding-004-sem-l2dist/STG1.parquet",
    "te004-clastask-l2": f"{bucket_prefix}semantic_stage/text-embedding-004-clas-l2dist/STG1.parquet",
    # cosine is almost identical with l2dist, so we skip it (too many traces otherwise)
    # "minilm-cos": f"{bucket_prefix}semantic_stage/minilm-l6-v2-cosine/STG1.parquet",
    # "te005-clastask-cos": f"{bucket_prefix}semantic_stage/text-embedding-005-clas-cosine/STG1.parquet",
}
semantic_dfs = {name: pd.read_parquet(path) for name, path in semantic_files.items()}


# %%
# convert semantic distance to confidence
def semantic_distance_to_confidence(
    sem_dist: list[dict],
    digits: int,
    shortlist_len: int | None = None,
) -> list[dict] | None:
    """Convert semantic distance to confidence score."""
    sem_dist = list(sem_dist)
    if len(sem_dist) == 0:
        return None
    if len(sem_dist) == 1:
        sem_dist[0]["likelihood"] = 1.0
        return sem_dist
    # normalize to [0, 1]
    eps = 0.00001
    pairs: dict = {}
    for item in sem_dist:
        code = item["code"][: (digits if digits > 0 else 2)]
        if code not in pairs or item["distance"] < pairs[code]:
            pairs[code] = item["distance"]
    pruned = sorted(
        [{"code": k, "distance": v} for k, v in pairs.items()],
        key=lambda x: x["distance"],
    )

    top_two = sum(x["distance"] for x in pruned[:2]) + eps

    # this likelihood transformation is useful to decide on a threshold, but not
    # directly used in the OM/MM metrics now
    for ind, item in enumerate(pruned):
        item["likelihood"] = max(
            0, 1 - item["distance"] / top_two - item["distance"] * 0.4 - ind * 0.1
        )

    # if pruned[0]['distance']<0.1:  # Clasiffai threshold implementation
    #    return pruned[:1]

    if shortlist_len:
        pruned = pruned[:shortlist_len]

    return pruned  # keep top 5 candidates (we limit the number of candidates in SurveyAssist)


# top = prompt2_df["semantic_candidates"].apply(lambda x: x[0].get("likelihood") if x else None)
# px.histogram(top)


# %%
# calculate metrics
eval_metrics = {}
logger.info(
    """Starting metrics calculation for semantic search methods...
    Note that we are not using final code, so this will be reported as missing during metrics calculation."""
)

for DIGITS in [5, 4, 3, 2, 1, 0]:
    logger.info("--- Evaluating %d-digit match ---", DIGITS)

    # clerical coding (2nd iteration, ground truth):
    clerical_codes_it2 = prep_clerical_codes(cc_it2_df, cc_it2_4plus_df, digits=DIGITS)

    # semantic search models (for different length of candidate shortlist):
    for shortlist_size in [5, 10, None]:
        for sem_name, sem_df in semantic_dfs.items():
            semantic_dfs[sem_name]["semantic_candidates"] = sem_df[
                "semantic_search_results"
            ].apply(
                lambda x, shortlist_size=shortlist_size, digits=DIGITS: semantic_distance_to_confidence(
                    x, digits, shortlist_size
                )
            )
            model_semantic = prep_model_codes(
                sem_df,
                digits=DIGITS,
                out_col="sa_initial_codes",
                codes_col=None,
                alt_codes_col="semantic_candidates",
                threshold=0,
            )
            combined_dataframe_sem = model_semantic.merge(
                clerical_codes_it2, on="unique_id", how="inner"
            )
            eval_metrics[(DIGITS, sem_name, shortlist_size)] = calc_simple_metrics(
                combined_dataframe_sem
            )


# %%
plot_df_accu = pd.DataFrame(
    [
        {
            "digits": k[0],
            "method": k[1],
            "shortlist_size": k[2] if k[2] is not None else "All",
            "OM Accuracy": v.initial_accuracy_metrics.accuracy_om_unambiguous,
            "MM Accuracy": v.initial_accuracy_metrics.accuracy_mm_total,
        }
        for k, v in eval_metrics.items()
    ]
)

# melt for easier plotting
plot_df_accu = plot_df_accu.melt(
    id_vars=["digits", "method", "shortlist_size"],
    value_vars=["OM Accuracy", "MM Accuracy"],
    var_name="metrics",
    value_name="value",
)
fig = px.line(
    plot_df_accu,
    x="digits",
    y="value",
    color="method",
    line_dash="shortlist_size",
    facet_col="metrics",
    title="Initial Classification Accuracy Metrics by Number of Digits and Method",
    markers=True,
    template="simple_white",
)
# drop first part of facet annotation
for i in fig.layout.annotations:
    i.text = i.text.split("=")[1]
# display y axes as percentages and remove axis title
fig.update_yaxes(tickformat=".0%", title_text="", showgrid=True, gridcolor="lightgrey")
fig.update_xaxes(
    tickvals=plot_df_accu["digits"].unique(),
    ticktext=["S" if i == 0 else str(i) for i in plot_df_accu["digits"].unique()],
)

# add text to footnote
fig.update_layout(margin={"b": 100})
fig.add_annotation(
    text=(
        """
OM: One-to-Many match on a subset where the true label is not ambiguous. (Is the true label in the model's shortlist?)<br>
MM: Many-to-Many match on the full set. (Is there any overlap between the true label's and model's shortlists?)
"""
    ),
    align="left",
    xref="paper",
    yref="paper",
    x=-0.08,
    y=-0.3,
    showarrow=False,
    font={"size": 10},
)
fig.update_layout(height=500, width=770)

fig.show()
if output_folder:
    fig.write_html(f"{output_folder}/2025-09_metrics_accuracy_semantic_methods.html")


# %%
# top candidate performance by threshold on distance
top_match_metrics = {}
cc_codable = {}
for DIGITS in [5, 0]:
    logger.info("--- Evaluating %d-digit match ---", DIGITS)

    # clerical coding (2nd iteration, ground truth):
    clerical_codes_it2 = prep_clerical_codes(cc_it2_df, cc_it2_4plus_df, digits=DIGITS)

    # semantic search model (with thresholding on the likelihood calculated from distance):
    for sem_name, sem_df in semantic_dfs.items():
        combined_dataframe_sem = sem_df.merge(
            clerical_codes_it2, on="unique_id", how="inner"
        ).copy()

        # get top candidate with its distance and whether it is in clerical shortlist
        combined_dataframe_sem["top_distance"] = combined_dataframe_sem[
            "semantic_search_results"
        ].apply(lambda x: x[0]["distance"] if len(x) > 0 else None)
        combined_dataframe_sem["top_candidate"] = combined_dataframe_sem[
            "semantic_search_results"
        ].apply(
            lambda x, digits=DIGITS: (
                get_clean_n_digit_one_code(x[0]["code"], digits) if len(x) > 0 else None
            )
        )
        combined_dataframe_sem["top_in_cc"] = combined_dataframe_sem.apply(
            lambda row: next(iter(row["top_candidate"])) in row["clerical_codes"],
            axis=1,
        )

        combined_dataframe_sem = combined_dataframe_sem.sort_values(
            by="top_distance", ascending=True
        ).reset_index(drop=True)

        # subset to CC unambiguous cases
        unambig_msk = combined_dataframe_sem["clerical_codes"].apply(
            lambda x: len(x) == 1
        )
        unambig_df = combined_dataframe_sem[unambig_msk].reset_index(drop=True).copy()
        cc_codable[DIGITS] = len(unambig_df) / clerical_codes_it2.shape[0]

        # calculate proportion and accuracy at different thresholds on distance
        for df in [combined_dataframe_sem, unambig_df]:
            df["subset_total"] = range(1, len(df) + 1)
            df["codability"] = df["subset_total"] / len(df)
            df["match_count"] = df["top_in_cc"].cumsum()
            df["accuracy"] = df["match_count"] / df["subset_total"]

        combined_dataframe_sem = combined_dataframe_sem.drop_duplicates(
            subset=["top_distance"], keep="last"
        )
        unambig_df = unambig_df.drop_duplicates(subset=["top_distance"], keep="last")

        # store for plotting
        top_match_metrics[(DIGITS, sem_name, "OO")] = unambig_df.copy()
        top_match_metrics[(DIGITS, sem_name, "MO")] = combined_dataframe_sem.copy()

# %%
# get survey assist model metrics for comparison (one point, not a curve)
model_df = pd.read_parquet(
    f"{bucket_prefix}two_prompt_pipeline/2025_09_full_2k_gemini25/STG5.parquet"
)
logger.info(
    """Starting metrics calculation for current SurveyAssist model...
    Note that we are not using final code, so this will be reported as missing during metrics calculation."""
)
sa_df = pd.DataFrame()
for DIGITS in [0, 5]:
    clerical_codes_it2 = prep_clerical_codes(cc_it2_df, cc_it2_4plus_df, digits=DIGITS)
    model_prompt2 = prep_model_codes(
        model_df,
        digits=DIGITS,
        out_col="sa_initial_codes",
        threshold=0.7,
    )
    combined_dataframe_m2 = model_prompt2.merge(
        clerical_codes_it2, on="unique_id", how="inner"
    )
    eval_metr = calc_simple_metrics(combined_dataframe_m2)
    sa_df = pd.concat(
        [
            sa_df,
            pd.DataFrame(
                {
                    "digits": [DIGITS] * 2,
                    "method": ["SurveyAssist"] * 2,
                    "match_type": ["OO", "MO"],
                    "accuracy": [
                        eval_metr.initial_accuracy_metrics.accuracy_oo_unambiguous,
                        eval_metr.initial_accuracy_metrics.accuracy_mo_unambiguous,
                    ],
                    "codability": [eval_metr.codability_metrics.initial_codable_prop]
                    * 2,
                }
            ),
        ],
        ignore_index=True,
    ).reset_index(drop=True)


# %%
# prepare data for plotting codability vs accuracy curves
plot_df = pd.DataFrame(
    [
        {
            "digits": key[0],
            "method": key[1],
            "match_type": key[2],
            "codability": row["codability"],
            "distance_threshold": round(row.top_distance, 3),
            "accuracy": row["accuracy"],
            "match_count": row["match_count"],
            "subset_total": row["subset_total"],
        }
        for key, df in top_match_metrics.items()
        for _, row in df.iterrows()
    ]
)

for DIGITS in [0, 5]:
    fig = px.line(
        plot_df[
            (plot_df["codability"] > 1 / 10) & (plot_df["digits"] == DIGITS)
        ],  # remove initial small sample variation
        x="codability",
        y="accuracy",
        color="method",
        facet_col="match_type",
        title=f"""Codability vs Accuracy of top candidates <br>(above parametrised threshold, {
            str(DIGITS)+"-digits" if DIGITS>0 else "Section level"} match)""",
        template="simple_white",
        hover_data={
            "distance_threshold": True,
            "match_count": True,
            "subset_total": True,
        },
    )
    # Add vline to all subplots/facets
    fig.add_vline(
        x=cc_codable[DIGITS],
        line={"color": "navy", "width": 2},
        line_dash="dot",
        annotation_text="Clerical codability",
        annotation_position="bottom right",
        annotation_font_size=10,
    )

    # add dots for SA model by facets
    for fac, match_type in enumerate(["OO", "MO"]):
        msk = (sa_df["digits"] == DIGITS) & (sa_df["match_type"] == match_type)
        fig.add_scatter(
            x=sa_df[msk]["codability"],
            y=sa_df[msk]["accuracy"],
            mode="markers",
            marker={"size": 10, "color": "navy", "symbol": "x"},
            name="SurveyAssist",
            col=fac + 1,
            row=1,
            showlegend=fac == 0,
        )

    # display y axes as percentages and remove axis title
    fig.update_yaxes(tickformat=".0%", showgrid=True, gridcolor="lightgrey")
    fig.update_xaxes(tickformat=".0%", title_text="Codability (prop. above threshold)")

    # add text to footnote
    fig.update_layout(margin={"b": 100})
    fig.add_annotation(
        text=(
            """
    OO: One-to-One match on a subset where the true label as well as the model's label are not ambiguous.<br>
    MO: Many-to-One match on a subset where the model is not ambiguous. (Is the model's label in the true label shortlist?)<br>
    """
        ),
        align="left",
        xref="paper",
        yref="paper",
        x=-0.08,
        y=-0.34,
        showarrow=False,
        font={"size": 10},
    )
    fig.update_layout(height=500, width=770)
    fig.show()

    if output_folder:
        fig.to_html(
            f"{output_folder}/2025-10_semantic_top_candidate_accuracy_codability_{DIGITS}-digits.html"
        )

# %%
