"""Work in progress notebook to visualize metrics for different models.

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
clerical_it1_file = f"{bucket_prefix}original_datasets/DSC_Rep_Sample.csv"
clerical_it1_4plus_file = (
    f"{bucket_prefix}original_datasets/Codes_for_4_plus_DSC_Rep_Sample.csv"
)
clerical_it2_file = f"{bucket_prefix}original_datasets/DSC_Rep_Sample_IT2.csv"
clerical_it2_4plus_file = (
    f"{bucket_prefix}original_datasets/Codes_for_4_plus_DSC_Rep_Sample_IT2.csv"
)
cc_it1_df = pd.read_csv(clerical_it1_file)
cc_it1_4plus_df = pd.read_csv(clerical_it1_4plus_file)
cc_it2_df = pd.read_csv(clerical_it2_file)
cc_it2_4plus_df = pd.read_csv(clerical_it2_4plus_file)


# %%
# load model outputs
latest_model_name = "m_2p_g2.5"
model_files = {
    # one-prompt survey assist outputs
    "m_1p_g2.5": f"{bucket_prefix}one_prompt_pipeline/2025_10_full_2k_gemini25/STG5.parquet",
    "m_1p_g2.0": f"{bucket_prefix}one_prompt_pipeline/2025_10_full_2k_gemini20/STG5.parquet",
    # two-prompt survey assist outputs
    "m_2p_g2.0": f"{bucket_prefix}two_prompt_pipeline/2025_08_full_2k_gemini20/STG5.parquet",
    "m_2p_g2.5": f"{bucket_prefix}two_prompt_pipeline/2025_09_full_2k_gemini25/STG5.parquet",
}

model_dfs = {name: pd.read_parquet(path) for name, path in model_files.items()}


# %%
# calculate metrics
eval_metrics = {}
logger.info(
    """Starting metrics calculation...
    Note that we are not using final code, so this will be reported as missing during metrics calculation."""
)
for DIGITS in [0, 2, 3, 4, 5]:
    logger.info("--- Evaluating %d-digit match ---", DIGITS)

    # clerical coding (2nd iteration, ground truth):
    clerical_codes_it2 = prep_clerical_codes(cc_it2_df, cc_it2_4plus_df, digits=DIGITS)

    # initial clerical codes for comparison with ground truth (it2):
    clerical_codes_it1 = prep_clerical_codes(
        cc_it1_df, cc_it1_4plus_df, digits=DIGITS
    ).rename(columns={"clerical_codes": "sa_initial_codes"})
    combined_dataframe_cc = clerical_codes_it1.merge(
        clerical_codes_it2, on="unique_id", how="inner"
    )
    eval_metrics[(DIGITS, "cc_it1")] = calc_simple_metrics(combined_dataframe_cc)

    # standard model outputs (2 prompt):
    for model_name, df in model_dfs.items():
        model_prompt2 = prep_model_codes(
            df,
            digits=DIGITS,
            out_col="sa_initial_codes",
            threshold=0.7,
        )
        combined_dataframe_m2 = model_prompt2.merge(
            clerical_codes_it2, on="unique_id", how="inner"
        )
        eval_metrics[(DIGITS, model_name)] = calc_simple_metrics(combined_dataframe_m2)


# %%
plot_df_f1 = pd.DataFrame(
    [
        {
            "digits": str(k[0]) if k[0] > 0 else "S",
            "method": k[1],
            "codability": v.codability_metrics.initial_codable_prop,
            "f1": v.ambiguity_metrics.f1,
            "precision": v.ambiguity_metrics.precision,
            "recall": v.ambiguity_metrics.recall,
            "accuracy": v.ambiguity_metrics.accuracy,
        }
        for k, v in eval_metrics.items()
    ]
)
true_codablity = pd.DataFrame(
    [
        {
            "digits": str(k[0]) if k[0] > 0 else "S",
            "method": "true_it2",
            "codability": (v.ambiguity_metrics.TN + v.ambiguity_metrics.FP)
            / v.initial_accuracy_metrics.total_records,
        }
        for k, v in eval_metrics.items()
    ]
).drop_duplicates()
plot_df_f1 = pd.concat([plot_df_f1, true_codablity], ignore_index=True)

# melt for easier plotting
plot_df_f1 = plot_df_f1.melt(
    id_vars=["digits", "method"],
    value_vars=["codability", "precision", "recall", "f1", "accuracy"],
    var_name="metrics",
    value_name="value",
)

# add wald CI for codability
n = cc_it2_df.shape[0]
plot_df_f1["ci"] = 1.96 * (plot_df_f1["value"] * (1 - plot_df_f1["value"]) / n).pow(0.5)
plot_df_f1.loc[~plot_df_f1["metrics"].isin(["codability", "accuracy"]), "ci"] = None

fig = px.line(
    plot_df_f1,
    x="digits",
    y="value",
    color="method",
    facet_col="metrics",
    title="Ambiguity Decision Metrics by Number of Digits and Method",
    markers=True,
    template="simple_white",
    # error_y="ci",
)
# drop first part of facet annotation
for i in fig.layout.annotations:
    i.text = i.text.split("=")[-1].capitalize()
# display y axes as percentages and remove axis title
fig.update_yaxes(tickformat=".0%", title_text="", showgrid=True, gridcolor="lightgrey")

# add text to footnote
fig.update_layout(margin={"b": 130})
fig.add_annotation(
    text=(
        """
Codability: Percentage of records identified as unambiguous by either the model or clerical coders.<br>
Precision: Among cases flagged as ambiguous by the model, the percentage that are truly ambiguous.<br>
Recall: Among all truly ambiguous cases, the percentage correctly identified by the model.<br>
F1: The harmonic mean of precision and recall.<br>
Accuracy: Overall percentage of correct codability/ambiguity decisions.
"""
    ),
    align="left",
    xref="paper",
    yref="paper",
    x=-0.08,
    y=-0.45,
    showarrow=False,
    font={"size": 10},
)
fig.update_layout(height=500, width=1000)

fig.show()

if output_folder:
    fig.write_html(f"{output_folder}/2025-09_metrics_ambiguity_all_methods.html")


# %%
plot_df_accu = pd.DataFrame(
    [
        {
            "digits": str(k[0]) if k[0] > 0 else "S",
            "method": k[1],
            "OO Accuracy": v.initial_accuracy_metrics.accuracy_oo_unambiguous,
            "OM Accuracy": v.initial_accuracy_metrics.accuracy_om_unambiguous,
            "MO Accuracy": v.initial_accuracy_metrics.accuracy_mo_unambiguous,
            "MM Accuracy": v.initial_accuracy_metrics.accuracy_mm_total,
        }
        for k, v in eval_metrics.items()
    ]
)

# melt for easier plotting
plot_df_accu = plot_df_accu.melt(
    id_vars=["digits", "method"],
    value_vars=["OO Accuracy", "OM Accuracy", "MO Accuracy", "MM Accuracy"],
    var_name="metrics",
    value_name="value",
)
fig = px.line(
    plot_df_accu,
    x="digits",
    y="value",
    color="method",
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

# add text to footnote
fig.update_layout(margin={"b": 125})
fig.add_annotation(
    text=(
        """
OO: One-to-One match on a subset where the true label as well as the model's label are not ambiguous.<br>
OM: One-to-Many match on a subset where the true label is not ambiguous. (Is the true label in the model's shortlist?)<br>
MO: Many-to-One match on a subset where the model is not ambiguous. (Is the model's label in the true label shortlist?)<br>
MM: Many-to-Many match on the full set. (Is there any overlap between the true label's and model's shortlists?)
"""
    ),
    align="left",
    xref="paper",
    yref="paper",
    x=-0.08,
    y=-0.42,
    showarrow=False,
    font={"size": 10},
)
fig.update_layout(height=500, width=770)

fig.show()
if output_folder:
    fig.write_html(f"{output_folder}/2025-09_metrics_accuracy_all_methods.html")

# %%
# create confusion matrix for section (0-digit) and subset of 5-digit
col1 = "clerical_codes"
col2 = "sa_initial_codes"

for DIGITS in [0, 5]:
    clerical_codes_it2 = prep_clerical_codes(cc_it2_df, cc_it2_4plus_df, digits=DIGITS)
    model_prompt2 = prep_model_codes(
        model_dfs[latest_model_name], digits=DIGITS, out_col="sa_initial_codes"
    )
    df = model_prompt2.merge(clerical_codes_it2, on="unique_id", how="inner")

    subset = {}
    subset["Unambiguously coded cases only"] = (df[col1].map(len) == 1) & (
        df[col2].map(len) == 1
    )
    # for semi-unambiguous, keep only cases where there is small set on either side
    n = 3
    subset["Subset of ambiguous cases with only two candidates"] = (
        (df[col1].map(len) < n) & (df[col2].map(len) < n) & ~next(iter(subset.values()))
    )

    for lab, msk in subset.items():
        df2 = df[msk].copy().explode(col1).explode(col2)
        if DIGITS > 1:
            # find the most frequent off diagonal entries in plot_df
            df3 = (
                df2[df2[col1] != df2[col2]]
                .groupby([col1, col2])
                .size()
                .sort_values(ascending=False)
            )
            cutoff = df3.iloc[min(10, len(df3) - 1)]
            df3 = df3[df3 > cutoff].reset_index()
            labels = sorted(set(df3[col1]).union(df3[col2]))
            plot_df = (
                df2[df2[col1].isin(labels) & df2[col2].isin(labels)]
                .groupby([col1, col2])
                .size()
                .unstack(fill_value="")
            )
        else:
            labels = sorted(df[col1].explode().dropna().unique())
            plot_df = df2.groupby([col1, col2]).size().unstack(fill_value="")

        fig = px.imshow(
            plot_df,
            text_auto=True,
            aspect="equal",
            color_continuous_scale="Blues",
            title=f"Confusion matrix for SIC section, Clerical vs SurveyAssist (2 prompt model)<br><b>{lab}</b>",
            template="simple_white",
        )
        # reorder x axis values
        fig.update_xaxes(
            title="Model Initial Code",
            categoryorder="array",
            categoryarray=labels,
            showgrid=True,
            gridcolor="lightgrey",
            ticks="outside",
            showline=True,
            mirror=True,
            zeroline=False,
            dtick=1,
            tickson="boundaries",  # show grid between ticks
        )
        fig.update_yaxes(
            title="Clerical Initial Code",
            categoryorder="array",
            categoryarray=labels,
            showgrid=True,
            gridcolor="lightgrey",
            ticks="outside",
            showline=True,
            mirror=True,
            zeroline=False,
            dtick=1,
            tickson="boundaries",
        )

        fig.update_layout(height=700, width=770)

        fig.show()

        if output_folder:
            fig.write_html(
                f"{output_folder}/2025-09_metrics_vis_prompt2_{lab.lower().replace('-', '_')}_confusion_matrix.html"
            )

# %%
# inspect sample of questions
sample_ids = cc_it2_df.sample(20, random_state=1).unique_id
for uid in sample_ids:
    print(f"Unique ID: {uid}")
    for model in model_dfs:
        question = model_dfs[model][
            model_dfs[model].unique_id == uid
        ].followup_question.iloc[0]
        print(f"{model}: {question}")

# %%
# histogram by section (digits=0)
df_section = {}
df_section["Admin labels"] = cc_it2_df[["unique_id", "sic_section"]].copy()
df_section["Clerical codes"] = prep_clerical_codes(cc_it2_df, cc_it2_4plus_df, digits=0)
df_section["Clerical codes"]["sic_section"] = df_section["Clerical codes"][
    "clerical_codes"
].map(lambda x: next(iter(x)) if len(x) == 1 else None)
for model_name in [latest_model_name]:
    model_df = prep_model_codes(
        model_dfs[model_name], digits=0, out_col="sa_initial_codes"
    )
    model_df["sic_section"] = model_df["sa_initial_codes"].map(
        lambda x: next(iter(x)) if len(x) == 1 else None
    )
    df_section["SurveyAssist"] = model_df.copy()
for key, df in df_section.items():
    df["source"] = key

plot_df_section = (
    pd.concat(df_section.values(), ignore_index=True)
    .dropna(subset=["sic_section"])
    .groupby(["sic_section", "source"])
    .size()
    .reset_index()
    .rename(columns={0: "count"})
)
plot_df_section["sample_size"] = plot_df_section.groupby("source")["count"].transform(
    "sum"
)
plot_df_section["Frequency"] = plot_df_section.groupby("source")["count"].transform(
    lambda x: x / x.sum()
)
plot_df_section["ci"] = 1.96 * (
    plot_df_section["Frequency"]
    * (1 - plot_df_section["Frequency"])
    / plot_df_section["sample_size"]
).pow(0.5)

fig = px.bar(
    plot_df_section,
    x="sic_section",
    y="Frequency",
    color="source",
    barmode="group",
    title="Distribution of unambiguously coded responses at SIC Section level",
    template="simple_white",
    error_y="ci",
    hover_data={"count": True, "sample_size": True},
)

fig.update_xaxes(
    title="SIC Section",
    categoryorder="category ascending",
    showgrid=True,
    gridcolor="lightgrey",
    ticks="outside",
    showline=True,
    mirror=True,
    zeroline=False,
    dtick=1,
    tickson="boundaries",
)

# make the ci lines thinner
fig.update_traces(error_y={"thickness": 1, "width": 2})
fig.update_yaxes(showgrid=True, gridcolor="lightgrey", tickformat=".0%")

# legend on top, no title
fig.update_layout(
    legend={
        "title_text": "",
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
    }
)

if output_folder:
    fig.write_html(f"{output_folder}/2025-09_metrics_vis_section_distribution.html")

fig.update_layout(height=500, width=1200)
fig.show()


# %%
