"""Notebook to visualise the codability gain/loss using a Sankey diagram."""

# pylint: disable=C0301,C0103,R0801
# %%
import os

import dotenv
import pandas as pd
import plotly.express as px
import statsmodels.api as sm

from survey_assist_utils.data_cleaning.sic_codes import CODABILITY_LEVELS

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
cc_coded_df = pd.read_parquet(
    work_dir + "/clerically-coded/clerical_df_with_cc_clean_codes.parquet"
)
combined_df = pd.merge(
    sa_coded_df,
    cc_coded_df.drop(
        columns=[
            "job_title",
            "job_description",
            "org_description",
            "survey_assist_open_question",
            "survey_assist_open_question_response",
        ]
    ),
    on=["unique_id", "user"],
    how="outer",
)

# %%
# set the level of codability to consider
NUM_DIGITS = 5  # 2,0,..

labels_considered_coded = [y for x, y in CODABILITY_LEVELS if x >= NUM_DIGITS]
x_axis_title = f"Proportion of Responses Codable Unambiguously to {NUM_DIGITS}-digits"

# %%
coding_method = "cc"  #  'sa' or 'cc'

initial_code_col = f"{coding_method}_initial_codability_level"
final_code_col1 = f"{coding_method}_final_codability_level_open_q"
final_code_col2 = "sa_final_codability_level_closed_q"

initial_label = f"{coding_method.upper()} Initial Codability"
final_label1 = f"{coding_method.upper()} Final Codability (Open Q)"
final_label2 = "SA Final Codability (Closed Q)"

# %%
# create groups by which we want to visualise codability gain/loss
temp_df = combined_df.copy()
if coding_method == "cc":
    temp_df = temp_df[
        temp_df["batch_num"] == 1
    ]  # only consider rows with cc final code

group_col = "SIC Section"
section_sizes = temp_df.most_likely_sic_section.value_counts(dropna=False)
size_thr = 10
temp_df1 = temp_df.copy()
temp_df1[group_col] = temp_df1.most_likely_sic_section
too_small = sorted(section_sizes[section_sizes < size_thr].index.tolist())
msk = temp_df1.most_likely_sic_section.isin(too_small)
temp_df1.loc[msk, group_col] = "+".join(too_small)
temp_df2 = temp_df.copy()
temp_df2[group_col] = "Total"
temp_df = pd.concat([temp_df1, temp_df2], axis=0, ignore_index=True)

# aggregate - group size, percentage of each codability at desired level
plot_df = (
    temp_df.groupby([group_col])
    .agg(
        {
            "user": "count",
            initial_code_col: lambda x: (x.isin(labels_considered_coded)).mean(),
            final_code_col1: lambda x: (x.isin(labels_considered_coded)).mean(),
            final_code_col2: lambda x: (x.isin(labels_considered_coded)).mean(),
        }
    )
    .rename(
        columns={
            "user": "num_responses",
            initial_code_col: "code0",
            final_code_col1: "code1",
            final_code_col2: "code2",
        }
    )
    .sort_values(group_col, ascending=False)
    .reset_index()
)

plot_df_melted = (
    plot_df.melt(
        id_vars=[group_col, "num_responses"],
        value_vars=["code0", "code1", "code2"],
        var_name="Stage",
        value_name="prop",
    )
    .sort_values("Stage")
    .reset_index(drop=True)
)


# add confidence intervals for proportions
def proportion_confint(prop, nobs, alpha=0.05, method="wilson"):
    """Calculate confidence interval for a proportion."""
    ci_low, ci_upp = sm.stats.proportion_confint(
        int(prop * nobs), nobs, alpha=alpha, method=method
    )
    return ci_low, ci_upp, prop - ci_low, ci_upp - prop


plot_df_melted[["ci_low", "ci_upp", "ci_low_err", "ci_upp_err"]] = plot_df_melted.apply(
    lambda row: proportion_confint(prop=row["prop"], nobs=row["num_responses"]),
    axis=1,
    result_type="expand",
)

plot_df_melted_ci = plot_df_melted.melt(
    id_vars=[group_col, "Stage", "num_responses"],
    value_vars=["ci_low", "ci_upp", "prop"],
    var_name="Metric",
    value_name=x_axis_title,
)

plot_df_melted_ci["size"] = plot_df_melted_ci["num_responses"]  # for size mapping

plot_df_melted_ci = plot_df_melted_ci.sort_values(
    [group_col, "Stage"], ascending=[False, True]
).reset_index(drop=True)

# %%
# based on the plot_df I want a figure where each group will be horizontall, with two connected dots - one for inital codability, one for final codability
# Slightly offset each Stage vertically for better separation
stage_offsets = {
    initial_code_col: 0,
    final_code_col1: +0.05,
    final_code_col2: -0.05,
}
plot_df_melted_ci["y_offset"] = plot_df_melted_ci.apply(
    lambda row: plot_df[group_col].tolist().index(row[group_col])
    + stage_offsets.get(row["Stage"], 0),
    axis=1,
)

fig = px.line(
    plot_df_melted_ci[plot_df_melted_ci["Metric"] != "prop"],
    x=x_axis_title,
    y="y_offset",
    color="Stage",
    markers=True,
    line_group=group_col,
    template="plotly_white",
    title=f"{coding_method.upper()} Codability (to {NUM_DIGITS}-digits) by SIC Section",
)

# Update y-axis ticks to show group names at correct positions
fig.update_yaxes(
    tickvals=list(range(len(plot_df[group_col]))),
    ticktext=plot_df[group_col].tolist(),
    title_text=None,
)
# use percentage on x axis
fig.update_xaxes(tickformat=".0%", range=[-0.01, 1.045])

# make lines dashed and symbols just vertical line
fig.update_traces(marker={"symbol": "line-ns-open"}, line={"width": 1})  # dash='2',

fig.add_traces(
    px.scatter(
        plot_df_melted_ci[plot_df_melted_ci["Metric"] == "prop"],
        x=x_axis_title,
        y="y_offset",
        color="Stage",
        size="size",  # use opacity instead of alpha
    ).data
)
# increse size of markers
fig.update_traces(
    selector={"type": "scatter", "mode": "markers"}, marker={"size": 10, "opacity": 0.9}
)  # , symbol= 'cross'))

# add text above top categor saying Initial (in blue) Open Q (in red) Closed Q (in green) 0 use same colors as in the scatter
fig.add_annotation(
    x=0.15,
    y=len(plot_df[group_col]) + 0.5,
    text=initial_label,
    showarrow=False,
    font={"color": px.colors.qualitative.Plotly[0], "size": 14},
    xref="paper",
)
fig.add_annotation(
    x=0.5,
    y=len(plot_df[group_col]) + 0.5,
    text=final_label1,
    showarrow=False,
    font={"color": px.colors.qualitative.Plotly[1], "size": 14},
    xref="paper",
)
fig.add_annotation(
    x=0.93,
    y=len(plot_df[group_col]) + 0.5,
    text=final_label2,
    showarrow=False,
    font={"color": px.colors.qualitative.Plotly[2], "size": 14},
    xref="paper",
)
fig.add_annotation(
    x=0,
    y=len(plot_df[group_col]) + 0.5,
    text="SIC Section",
    showarrow=False,
    font={"color": "black", "size": 14},
    xanchor="right",
    xref="paper",
)
# add annoptation on the right hand side with total number of responses
fig.add_annotation(
    x=1.02,
    y=len(plot_df[group_col]) + 0.5,
    text="Count",
    showarrow=False,
    font={"color": "black", "size": 14},
    xanchor="right",
    xref="paper",
)

for i, row in plot_df.iterrows():
    fig.add_annotation(
        x=1,
        y=i,
        text=f"{row['num_responses']}",
        showarrow=False,
        font={"color": "black", "size": 10},
        xanchor="right",
        xref="paper",
    )

fig.update_layout(
    width=1000,
    height=600,
    showlegend=False,
)
fig.show()

if out_dir:
    fig.write_image(
        os.path.join(
            out_dir, f"{coding_method}_codability_{NUM_DIGITS}digits_by_sic_section.png"
        ),
        scale=2,
    )

# %%
