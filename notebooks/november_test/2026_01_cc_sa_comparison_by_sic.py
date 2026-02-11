"""Notebook to compare clerical coding vs SurveyAssist model coding performance by SIC sections.

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

from notebooks.november_test.helper_load_data import combine_small_groups, load_data
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

combined_df = load_data(work_dir)

# %%
group_df = combine_small_groups(
    combined_df, group_col="SIC Section", group_size_threshold=30, add_total=True
)
# %%
eval_metrics = {}
stage_cols = {
    "SA Initial": ("cc_initial_codes", "sa_initial_codes"),
    "SA Final Open Q": ("cc_final_codes_open_q", "sa_final_codes_open_q"),
    "SA Final Closed Q": ("cc_final_codes_open_q", "sa_final_codes_closed_q"),
}
for stage, col_names in stage_cols.items():
    for DIGITS in [2, 5]:
        for col in col_names:
            print(f"Processing {stage} codes to {DIGITS} digits for column {col}...")
            group_df[f"{col}_to_{DIGITS}digits"] = group_df[col].apply(
                lambda x, n=DIGITS: get_clean_n_digit_codes(x, n=n)[0]
            )
        for gr in group_df["SIC Section"].unique():
            msk = group_df["SIC Section"] == gr
            eval_metrics[(DIGITS, stage, gr)] = calc_simple_metrics(
                group_df[msk],
                truth_col=f"{col_names[0]}_to_{DIGITS}digits",
                initial_model_col=f"{col_names[1]}_to_{DIGITS}digits",
                final_model_col=None,
            )
# %%
plot_df = pd.DataFrame(
    [
        {
            "digits": str(k[0]) if k[0] > 0 else "S",
            "Stage": k[1],
            "Most Likely<br>SIC Section": k[2],
            "Codability": v.codability_metrics.initial_codable_prop,
            "F1": v.ambiguity_metrics.f1,
            "Precision": v.ambiguity_metrics.precision,
            "Recall": v.ambiguity_metrics.recall,
            "Accuracy": v.ambiguity_metrics.accuracy,
            "OO Accuracy": (
                v.initial_accuracy_metrics.accuracy_oo_unambiguous,
                v.initial_accuracy_metrics.matches_oo,
                v.ambiguity_metrics.TN,
            ),
            "OM Accuracy": (
                v.initial_accuracy_metrics.accuracy_om_unambiguous,
                v.initial_accuracy_metrics.matches_om,
                v.ambiguity_metrics.FP + v.ambiguity_metrics.TN,
            ),
            "MO Accuracy": (
                v.initial_accuracy_metrics.accuracy_mo_unambiguous,
                v.initial_accuracy_metrics.matches_mo,
                v.ambiguity_metrics.FN + v.ambiguity_metrics.TN,
            ),
            "MM Accuracy": (
                v.initial_accuracy_metrics.accuracy_mm_total,
                v.initial_accuracy_metrics.matches_mm,
                v.initial_accuracy_metrics.total_records,
            ),
        }
        for k, v in eval_metrics.items()
    ]
)
plot_df = plot_df.merge(
    group_df.groupby("SIC Section")
    .size()
    .reset_index()
    .rename(columns={0: "num_responses"}),
    left_on="Most Likely<br>SIC Section",
    right_on="SIC Section",
    how="left",
)


# %%
def create_codability_by_section_figure(
    input_df: pd.DataFrame,
    metrics: list[str],
    group_col: str = "Most Likely<br>SIC Section",
    title: str = "",
):
    """Create a figure visualising classification alignment by SIC sections.

    Args:
        input_df: DataFrame containing both SA and CC codings.
        metrics: List of metrics to visualise.
        group_col: Name of the column representing groups.
        title: Title of the plot.

    Returns:
        A Plotly Figure object representing the codability by SIC sections.
    """
    plot_df_melted = input_df.melt(
        id_vars=[group_col, "num_responses", "Stage"],
        value_vars=metrics,
        var_name="Metrics",
        value_name="accu_value",
    ).reset_index(drop=True)
    hoverdata: dict[str, str | bool] = {"accu_value": ":.2%"}
    if plot_df_melted["accu_value"].dtype == "object":
        plot_df_melted[["accu_value", "matches", "total"]] = pd.DataFrame(
            plot_df_melted["accu_value"].tolist(), index=plot_df_melted.index
        )
        hoverdata = {"matches": True, "total": True, "accu_value": ":.2%"}

    plot_df_melted["metrics_ord"] = plot_df_melted["Metrics"].apply(metrics.index)

    plot_df_melted = plot_df_melted.sort_values(
        [group_col, "metrics_ord"], ascending=[False, True]
    ).reset_index(drop=True)

    fig = px.scatter(
        plot_df_melted,
        x="accu_value",
        y=group_col,
        color="Stage",
        symbol="Stage",
        template="plotly_white",
        facet_col="Metrics",
        title=title,
        hover_data=hoverdata,
    )
    fig.update_traces(
        marker={"size": 10, "opacity": 0.8},
    )
    fig.update_yaxes(title_text="")
    fig.update_xaxes(tickformat=".0%", title_text="")
    for annotation in fig.layout.annotations:
        annotation.text = annotation.text.split("=", 1)[1]
        annotation.font.size = 15
        for trace in fig.data:
            if trace.name.startswith(annotation.text):
                annotation.font.color = trace.marker.color
            annotation.y -= 0.006

    fig.add_annotation(
        x=0,
        y=len(plot_df[group_col]) / 6 + 0.5,
        text=group_col,
        showarrow=False,
        font={"color": "black", "size": 14},
        xanchor="right",
        xref="paper",
    )
    # add annoptation on the right hand side with total number of responses
    fig.add_annotation(
        x=1.06,
        y=len(plot_df[group_col]) / 6 + 0.5,
        text="Count",
        showarrow=False,
        font={"color": "black", "size": 14},
        xanchor="right",
        xref="paper",
    )

    for _, row in input_df.iterrows():
        fig.add_annotation(
            x=1.03,
            y=row[group_col],
            text=f"{row['num_responses']}",
            showarrow=False,
            font={"color": "black", "size": 10},
            xanchor="right",
            xref="paper",
        )

    fig.update_layout(
        width=400 + 150 * len(metrics),
        height=600,
        legend={
            "title_text": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    return fig


# %%
metrics_list = [
    [
        "Codability",
        "Precision",
        "Recall",
        "F1",
        "Accuracy",
    ],
    [
        "OO Accuracy",
        "OM Accuracy",
        "MO Accuracy",
        "MM Accuracy",
    ],
]

for num_dig in [2, 5]:
    for metric in metrics_list:
        out_fig = create_codability_by_section_figure(
            plot_df[plot_df["digits"] == str(num_dig)].reset_index(drop=True),
            metrics=metric,
            group_col="Most Likely<br>SIC Section",
            title=f"CC-SA Agreement metrics to {num_dig}-digits by SIC Section",
        )

        out_fig.show()

        if out_dir:
            file_name = f"cc_sa_comparison_by_sic_{num_dig}digits_" + "_".join(
                [m.lower().replace(" ", "_") for m in metric]
            )
            out_fig.write_image(
                os.path.join(
                    out_dir,
                    file_name + ".png",
                ),
                scale=2,
            )
            out_fig.write_html(
                os.path.join(out_dir, file_name + ".html"),
            )

# %%
