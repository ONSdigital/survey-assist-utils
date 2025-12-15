"""Work in progress notebook to visualize metrics for different models.

It loads specific clerical coding data and model outputs from bucket.
The bucket name and folder (on line 32) can be manually entered or it is read from
the .env file, where it should be stored as BUCKET_PREFIX variable, i.e.:
BUCKET_PREFIX = "gs://<bucket-name>/<folder>/"

Disabled check for too long lines (f strings) and variables names (uppercase for constants)
"""

# pylint: disable=C0301,C0103,R0801

# %%
import os

import dotenv
import pandas as pd
import plotly.express as px

from survey_assist_utils.data_cleaning.prep_data import get_clean_n_digit_codes
from survey_assist_utils.evaluation.metrics import (
    calc_simple_metrics,
)

# %%
data_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET") or ""
work_dir = data_bucket + "analysis-interim-results"
out_dir = (
    "data/figures/"  # needs local folder unfortunately, set to None to skip saving
)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)


# %% load data
cc_file = work_dir + "/clerically-coded/clerical_df_with_cc_clean_initial_codes.parquet"
sa_file = work_dir + "/evaluation_df_with_sa_clean_codes.parquet"
cc_df = pd.read_parquet(cc_file)
sa_df = pd.read_parquet(sa_file)

merged_subset = cc_df.merge(
    sa_df, on="unique_id", how="inner", suffixes=("_cc", "_sa")
).reset_index(drop=True)

print(
    f"Loaded data with {merged_subset.shape[0]} records (after merging clerical {cc_df.shape[0]} and model data {sa_df.shape[0]})."
)


# %% calculate metrics at different digit levels for different methods
eval_metrics = {}
for DIGITS in [0, 2, 3, 4, 5]:
    merged_subset[f"clerical_codes_to_{DIGITS}digits"] = merged_subset[
        "clerical_codes"
    ].apply(lambda x, n=DIGITS: get_clean_n_digit_codes(set(x), n=n)[0])
    merged_subset[f"sa_initial_codes_to_{DIGITS}digits"] = merged_subset[
        "sa_initial_codes"
    ].apply(lambda x, n=DIGITS: get_clean_n_digit_codes(set(x), n=n)[0])
    eval_metrics[(DIGITS, "Initial SA")] = calc_simple_metrics(
        merged_subset,
        truth_col=f"clerical_codes_to_{DIGITS}digits",
        initial_model_col=f"sa_initial_codes_to_{DIGITS}digits",
        final_model_col=None,
    )


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
            "method": "Initial CC",
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
n = merged_subset.shape[0]
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

if out_dir:
    fig.write_html(f"{out_dir}/cc_sa_initial_codes_ambiguity_decision.html")


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
if out_dir:
    fig.write_html(f"{out_dir}/cc_sa_initial_codes_accuracy_metrics.html")

# %%
# create confusion matrix for section (0-digit) and subset of 5-digit
df = merged_subset.copy()
for DIGITS in [5, 2]:
    col1 = f"clerical_codes_to_{DIGITS}digits"
    col2 = f"sa_initial_codes_to_{DIGITS}digits"
    subset = {}
    subset["Unambiguously coded cases only"] = (merged_subset[col1].map(len) == 1) & (
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

        if out_dir:
            fig.write_html(
                f"{out_dir}/cc_sa_initial_codes_{lab.lower().replace('-', '_')}_confusion_matrix.html"
            )

# %%
# histogram by section (digits=0)
df_section = {}
df_section["CC initial"] = merged_subset[["clerical_codes_to_0digits"]].copy()
df_section["CC initial"]["sic_section"] = df_section["CC initial"][
    "clerical_codes_to_0digits"
].map(lambda x: next(iter(x)) if len(x) == 1 else None)
df_section["SA initial"] = merged_subset[["sa_initial_codes_to_0digits"]].copy()
df_section["SA initial"]["sic_section"] = df_section["SA initial"][
    "sa_initial_codes_to_0digits"
].map(lambda x: next(iter(x)) if len(x) == 1 else None)

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

if out_dir:
    fig.write_html(f"{out_dir}/cc_sa_initial_codes_section_distribution.html")

fig.update_layout(height=500, width=1200)
fig.show()


# %%
