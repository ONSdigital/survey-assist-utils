"""Notebook to visualise the codability gain/loss by SIC sections (or other groups).

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

from notebooks.november_test.helper_load_data import combine_small_groups
from survey_assist_utils.data_cleaning.sic_codes import (
    CODABILITY_LEVELS,
    get_clean_n_digit_codes,
    parse_numerical_code,
)

# %%
data_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET") or ""
work_dir = data_bucket + "analysis-interim-results"
out_dir = (
    "data/figures/"  # needs local folder unfortunately, set to None to skip saving
)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

# load combined df with codability levels
sa_coded_df = pd.read_parquet(work_dir + "/evaluation_df_with_sa_clean_codes.parquet")
sa_closed_q = pd.read_parquet(
    work_dir + "/closed_questions/closed_questions_codes.parquet"
)
cc_coded_df = pd.read_parquet(
    work_dir + "/clerically-coded/clerical_df_with_cc_clean_codes.parquet"
)

combined_df = sa_coded_df.merge(
    sa_closed_q.drop(
        columns=sa_closed_q.columns.intersection(sa_coded_df.columns).difference(
            ["unique_id", "user"]
        )
    ),
    on=["unique_id", "user"],
    how="outer",
).merge(
    cc_coded_df.drop(
        columns=cc_coded_df.columns.intersection(sa_coded_df.columns).difference(
            ["unique_id", "user"]
        )
    ),
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


# %%
# update sic_section column based on final clerical codes
def extract_sic_section(row):
    """Extract SIC section (0-digit) from a set of codes."""
    cc_final = get_clean_n_digit_codes(row["cc_final_codes_open_q"], n=0)[0]
    if len(cc_final) == 1:
        return next(iter(cc_final))
    sa_closed = get_clean_n_digit_codes(row["sa_final_codes_closed_q"], n=0)[0]
    sa_open = get_clean_n_digit_codes(row["sa_final_codes_open_q"], n=0)[0]
    if (len(sa_closed.intersection(cc_final)) == 1) | (
        len(sa_closed.intersection(sa_open)) == 1
    ):
        return next(iter(sa_closed))
    if len(sa_open.intersection(cc_final)) == 1:
        return next(iter(sa_open.intersection(cc_final)))

    cc_initial = get_clean_n_digit_codes(row["cc_initial_codes"], n=0)[0]
    sa_initial = get_clean_n_digit_codes(row["sa_initial_codes"], n=0)[0]
    # find most frequent section among all codes
    codes = (
        list(cc_final)
        + list(sa_closed)
        + list(sa_open)
        + list(cc_initial)
        + list(sa_initial)
    )
    if not codes:
        return None
    section_counts = pd.Series(codes).value_counts()
    freq_sections = section_counts[section_counts == section_counts.max()].index
    if len(freq_sections) == 1:
        return freq_sections[0]
    # print(row['most_likely_sic_section'], cc_final, sa_closed, sa_open, cc_initial, sa_initial, section_counts)
    return row["most_likely_sic_section"]


combined_df["SIC Section"] = combined_df.apply(extract_sic_section, axis=1)

# sa_coded_df.merge(combined_df[["unique_id", "SIC Section"]]).to_parquet(
#     work_dir + "/evaluation_df_with_sa_clean_codes_and_sic_section.parquet"
# )


# %%
def add_proportion_confint(prop: float, nobs: int, alpha=0.05):
    """Calculate confidence interval for a proportion."""
    ci = binomtest(int(prop * nobs), nobs).proportion_ci(confidence_level=1 - alpha)
    return ci.low, ci.high, prop - ci.low, ci.high - prop


def create_codability_by_section_figure(
    input_df: pd.DataFrame,
    coding_method: str,
    num_digits: int,
    group_size_threshold: int = 30,
    group_col: str = "SIC Section",
):
    """Create a figure visualising codability by SIC sections.

    Args:
        input_df: DataFrame containing both SA and CC codings.
        coding_method: 'sa' or 'cc' to indicate which coding method to use.
        num_digits: Number of digits to consider for codability.
        group_size_threshold: Minimum size of group to be shown separately. Groups smaller than this will be combined.
        group_col: Name of the column representing groups.

    Returns:
        A Plotly Figure object representing the codability by SIC sections.
    """
    labels_considered_coded = [y for x, y in CODABILITY_LEVELS if x >= num_digits]
    x_axis_title = (
        f"Proportion of Responses Codable Unambiguously to {num_digits}-digits"
    )

    # create groups by which we want to visualise codability gain/loss
    temp_df = input_df.copy().rename(
        columns={
            f"{coding_method}_initial_codability_level": "code0",
            f"{coding_method}_final_codability_level_open_q": "code1",
            "sa_final_codability_level_closed_q": "code2",
        }
    )

    temp_df = combine_small_groups(
        temp_df,
        group_col=group_col,
        group_size_threshold=group_size_threshold,
        add_total=True,
    )

    # aggregate - group size, percentage of each codability at desired level
    plot_df = (
        temp_df.groupby([group_col])
        .agg(
            {
                "user": "count",
                "code0": lambda x: (x.isin(labels_considered_coded)).mean(),
                "code1": lambda x: (x.isin(labels_considered_coded)).mean(),
                "code2": lambda x: (x.isin(labels_considered_coded)).mean(),
            }
        )
        .rename(
            columns={
                "user": "num_responses",
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

    plot_df_melted[["ci_low", "ci_upp", "ci_low_err", "ci_upp_err"]] = (
        plot_df_melted.apply(
            lambda row: add_proportion_confint(
                prop=row["prop"], nobs=row["num_responses"]
            ),
            axis=1,
            result_type="expand",
        )
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

    # Slightly offset each Stage vertically for better separation
    stage_offsets = {
        "code0": 0,
        "code1": +0.05,
        "code2": -0.05,
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
        title=f"{coding_method.upper()} Codability (to {num_digits}-digits) by SIC Section",
    )

    # Update y-axis ticks to show group names at correct positions
    fig.update_yaxes(
        tickvals=list(range(len(plot_df[group_col]))),
        ticktext=plot_df[group_col].tolist(),
        title_text=None,
    )
    # use percentage on x axis
    fig.update_xaxes(tickformat=".0%", range=[-0.01, 1.045])
    fig.update_traces(marker={"symbol": "line-ns-open"}, line={"width": 1})  # dash='2',

    fig.add_traces(
        px.scatter(
            plot_df_melted_ci[plot_df_melted_ci["Metric"] == "prop"],
            x=x_axis_title,
            y="y_offset",
            color="Stage",
            size="size",
        ).data
    )
    fig.update_traces(
        selector={"type": "scatter", "mode": "markers"},
        marker={"size": 10, "opacity": 0.9},
    )

    # add text above top category based on the stage of classification
    fig.add_annotation(
        x=0.15,
        y=len(plot_df[group_col]) + 0.5,
        text=f"{coding_method.upper()} Initial Codability",
        showarrow=False,
        font={"color": px.colors.qualitative.Plotly[0], "size": 14},
        xref="paper",
    )
    fig.add_annotation(
        x=0.5,
        y=len(plot_df[group_col]) + 0.5,
        text=f"{coding_method.upper()} Final Codability (Open Q)",
        showarrow=False,
        font={"color": px.colors.qualitative.Plotly[1], "size": 14},
        xref="paper",
    )
    fig.add_annotation(
        x=0.93,
        y=len(plot_df[group_col]) + 0.5,
        text="SA Final Codability (Closed Q)",
        showarrow=False,
        font={"color": px.colors.qualitative.Plotly[2], "size": 14},
        xref="paper",
    )
    fig.add_annotation(
        x=0,
        y=len(plot_df[group_col]) + 0.5,
        text=group_col,
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

    return fig


# %%
for classification_method in ["sa", "cc"]:
    for num_dig in [2, 5]:
        out_fig = create_codability_by_section_figure(
            input_df=combined_df.rename(
                columns={"SIC Section": "Most Likely<br>SIC Section"}
            ),
            group_col="Most Likely<br>SIC Section",
            coding_method=classification_method,
            group_size_threshold=30,
            num_digits=num_dig,
        )

        out_fig.show()

        if out_dir:
            out_fig.write_image(
                os.path.join(
                    out_dir,
                    f"{classification_method}_codability_{num_dig}digits_by_sic_section.png",
                ),
                scale=2,
            )
            out_fig.write_html(
                os.path.join(
                    out_dir,
                    f"{classification_method}_codability_{num_dig}digits_by_sic_section.html",
                )
            )

# %%
