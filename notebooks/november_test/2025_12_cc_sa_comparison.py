"""Notebook to compare clerical coding vs SurveyAssist model coding performance.

Loads preprocessed data with both clerical and SA codings,
calculates various metrics and visualises them.
Expects environment variable PREPROD_DATA_BUCKET to be set.

Disabled check for too long lines (f strings) and variables names (uppercase for constants)
"""

# pylint: disable=C0301,C0103,R0801

# %%
import os

import dotenv
import pandas as pd
import plotly.express as px
from scipy.stats import binomtest

from survey_assist_utils.data_cleaning.prep_data import (
    get_clean_n_digit_codes,
    parse_numerical_code,
)
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

# %%
# load combined df with codability levels
sa_coded_df = pd.read_parquet(work_dir + "/evaluation_df_with_sa_clean_codes.parquet")
sa_closed_q = pd.read_parquet(
    work_dir + "/closed_questions/closed_questions_codes.parquet"
)
cc_coded_df = pd.read_parquet(
    work_dir + "/clerically-coded/clerical_df_with_cc_clean_codes.parquet"
)
repeated_cols = [
    "job_title",
    "job_description",
    "org_description",
    "survey_assist_open_question",
    "survey_assist_open_question_response",
]
combined_df = sa_coded_df.merge(
    sa_closed_q.drop(columns=repeated_cols[0:3]),
    on=["unique_id", "user"],
    how="outer",
).merge(
    cc_coded_df.drop(columns=repeated_cols),
    on=["unique_id", "user"],
    how="outer",
)

print(
    f"Loaded data with {combined_df.shape[0]} records. "
    f"Merging clerical ({cc_coded_df.shape[0]}) with model data ({sa_coded_df.shape[0]}) "
    f"and closed q data ({sa_closed_q.shape[0]})."
)

# %%
# parquet doesn't like sets it saves it as arrays, convert back
set_cols = [
    "sa_initial_codes",
    "sa_final_codes_open_q",
    "cc_initial_codes",
    "cc_final_codes_open_q",
]

for col in set_cols:
    msk = combined_df[col].notna()
    combined_df.loc[msk, col] = combined_df.loc[msk, col].apply(set)
    combined_df.loc[~msk, col] = [set() for _ in range(msk.sum(), combined_df.shape[0])]

# and convert closed q codes to set for consistency
combined_df["sa_final_codes_closed_q"] = combined_df[
    "survey_assist_closed_question_response_code"
].apply(lambda x: get_clean_n_digit_codes(parse_numerical_code(x), n=5)[0])


# %% calculate metrics at different digit levels for different methods
eval_metrics = {}
stage_cols = {
    "Initial": ("cc_initial_codes", "sa_initial_codes"),
    "Final Open Q": ("cc_final_codes_open_q", "sa_final_codes_open_q"),
    "Final Closed Q": ("cc_final_codes_open_q", "sa_final_codes_closed_q"),
}
for stage, col_names in stage_cols.items():
    # we don't have clerical for most batches so use temporary subset for now
    if stage == "Initial":
        msk = combined_df.batch_num.notna()
    else:
        msk = (
            combined_df.batch_num.isin([1])
            & combined_df["survey_assist_open_question"].notna()
        )
    for DIGITS in [0, 2, 3, 4, 5]:
        for col in col_names:
            print(f"Processing {stage} codes to {DIGITS} digits for column {col}...")
            combined_df.loc[msk, f"{col}_to_{DIGITS}digits"] = combined_df.loc[
                msk, col
            ].apply(lambda x, n=DIGITS: get_clean_n_digit_codes(x, n=n)[0])
        eval_metrics[(DIGITS, stage, "sa_cc")] = calc_simple_metrics(
            combined_df.loc[msk],
            truth_col=f"{col_names[0]}_to_{DIGITS}digits",
            initial_model_col=f"{col_names[1]}_to_{DIGITS}digits",
            final_model_col=None,
        )
        eval_metrics[(DIGITS, stage, "cc_cc")] = calc_simple_metrics(
            combined_df.loc[msk],
            truth_col=f"{col_names[0]}_to_{DIGITS}digits",
            initial_model_col=f"{col_names[0]}_to_{DIGITS}digits",
            final_model_col=None,
        )


# %%
plot_df_f1 = pd.DataFrame(
    [
        {
            "digits": str(k[0]) if k[0] > 0 else "S",
            "method": k[2][0:2].upper() + " " + k[1],
            "codability": v.codability_metrics.initial_codable_prop,
            "f1": v.ambiguity_metrics.f1 if k[2] == "sa_cc" else None,
            "precision": v.ambiguity_metrics.precision if k[2] == "sa_cc" else None,
            "recall": v.ambiguity_metrics.recall if k[2] == "sa_cc" else None,
            "accuracy": v.ambiguity_metrics.accuracy if k[2] == "sa_cc" else None,
        }
        for k, v in eval_metrics.items()
    ]
)

# melt for easier plotting
plot_df_f1 = plot_df_f1.melt(
    id_vars=["digits", "method"],
    value_vars=["codability", "precision", "recall", "f1", "accuracy"],
    var_name="metrics",
    value_name="value",
)

# add wald CI for codability
n = combined_df.shape[0]
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
    # fig.write_html(f"{out_dir}/cc_sa_initial_codes_ambiguity_decision.html")
    fig.write_image(f"{out_dir}/cc_sa_initial_codes_ambiguity_decision.png")


# %%
plot_df_accu = pd.DataFrame(
    [
        {
            "digits": str(k[0]) if k[0] > 0 else "S",
            "Stage": k[1],
            "OO Accuracy": v.initial_accuracy_metrics.accuracy_oo_unambiguous,
            "OM Accuracy": v.initial_accuracy_metrics.accuracy_om_unambiguous,
            "MO Accuracy": v.initial_accuracy_metrics.accuracy_mo_unambiguous,
            "MM Accuracy": v.initial_accuracy_metrics.accuracy_mm_total,
        }
        for k, v in eval_metrics.items()
        if k[2] == "sa_cc"
    ]
)

# melt for easier plotting
plot_df_accu = plot_df_accu.melt(
    id_vars=["digits", "Stage"],
    value_vars=["OO Accuracy", "OM Accuracy", "MO Accuracy", "MM Accuracy"],
    var_name="metrics",
    value_name="value",
)
fig = px.line(
    plot_df_accu,
    x="digits",
    y="value",
    color="Stage",
    facet_col="metrics",
    title="Classification Accuracy Metrics by Number of Digits and Stage",
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
    # fig.write_html(f"{out_dir}/cc_sa_initial_codes_accuracy_metrics.html")
    fig.write_image(f"{out_dir}/cc_sa_initial_codes_accuracy_metrics.png")


# %%
# create confusion matrix for section (0-digit) and subset of 5-digit
df = combined_df[combined_df.batch_num.notna()].copy()
for DIGITS in [5, 2]:
    col1 = f"cc_initial_codes_to_{DIGITS}digits"
    col2 = f"sa_initial_codes_to_{DIGITS}digits"
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

        if plot_df.shape[0] == 0 or plot_df.shape[1] == 0:
            print(
                f"Skipping confusion matrix for {DIGITS}-digit, {lab} due to no data."
            )
            continue

        fig = px.imshow(
            plot_df,
            text_auto=True,
            aspect="equal",
            color_continuous_scale="Blues",
            title=f"Confusion matrix for SIC section, Clerical vs SurveyAssist<br><b>{lab}</b>",
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
            # fig.write_html(f"{out_dir}/cc_sa_initial_codes_{lab.lower().replace('-', '_')}_confusion_matrix.html")
            fig.write_image(
                f"{out_dir}/cc_sa_initial_codes_{lab.lower().replace('-', '_')}_confusion_matrix_{DIGITS}digits.png"
            )


# %%
# histogram by section (digits=0)
df_section = {}
for i, (stage, cols) in enumerate(stage_cols.items()):
    for col in cols:
        if col in df_section:
            continue
        print(f"Processing section distribution for column {col}...")
        msk = combined_df[col + "_to_0digits"].notna()
        df_section[col] = (
            combined_df.loc[msk, col + "_to_0digits"]
            .map(lambda x: next(iter(x)) if len(x) == 1 else None)
            .dropna()
            .to_frame(name="sic_section")
        )
        df_section[col]["source"] = f"{col.split('_')[0].upper()} {stage}"
        df_section[col]["ind"] = i  # for ordering traces in plot


plot_df_section = (
    pd.concat(df_section.values(), ignore_index=True)
    .dropna(subset=["sic_section"])
    .groupby(
        [
            "sic_section",
            "ind",
            "source",
        ]
    )
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


def add_proportion_confint(prop: float, nobs: int, alpha=0.05):
    """Calculate confidence interval for a proportion."""
    ci = binomtest(int(prop * nobs), nobs).proportion_ci(confidence_level=1 - alpha)
    return prop - ci.low, ci.high - prop


plot_df_section[["error_y_minus", "error_y_plus"]] = plot_df_section.apply(
    lambda row: pd.Series(
        add_proportion_confint(prop=row["Frequency"], nobs=row["sample_size"])
    ),
    axis=1,
)

fig = px.bar(
    plot_df_section,
    x="sic_section",
    y="Frequency",
    color="source",
    barmode="group",
    title="Distribution of unambiguously coded responses at SIC Section level",
    template="simple_white",
    error_y="error_y_plus",
    error_y_minus="error_y_minus",
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

fig.update_layout(height=500, width=1400)
fig.show()

if out_dir:
    # fig.write_html(f"{out_dir}/cc_sa_initial_codes_section_distribution.html")
    fig.write_image(f"{out_dir}/cc_sa_initial_codes_section_distribution.png")


# %%
# get examples
mask = (combined_df["sa_initial_codes"] == {"85310"}) & (
    combined_df["cc_initial_codes"] == {"85600"}
)
examples = combined_df[mask][
    [
        "unique_id",
        "job_title",
        "job_description",
        "org_description",
        "clerical_code_initial",
        "survey_assist_assigned_code",
    ]
]
print(examples)
# %%
