"""Notebook to visualise the codability gain/loss using a Sankey diagram.

Loads preprocessed data with both clerical and SA codings,
calculates various metrics and visualises them.
Expects environment variable PREPROD_DATA_BUCKET to be set.

Disabled check for too long lines (f strings) and variables names (uppercase for constants)
"""

# pylint: disable=C0301,C0103,R0801
# %%
import os
import re

import dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%
data_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET") or ""
work_dir = data_bucket + "analysis-interim-results"
out_dir = (
    "data/figures/"  # needs local folder unfortunately, set to None to skip saving
)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

# %%
sa_coded_df = pd.read_parquet(work_dir + "/evaluation_df_with_sa_clean_codes.parquet")
cc_coded_df = pd.read_parquet(
    work_dir + "/clerically-coded/clerical_df_with_cc_clean_codes.parquet"
)


# %%
def create_sankey_codability_gain_loss(
    input_df: pd.DataFrame,
    left_col: str = "SA Initial Codes",
    right_col: str = "SA Final Codes",
    gain_col: str | None = None,
    title_suffix: str = "",
) -> go.Figure:
    """Create a Sankey diagram to visualise codability gain/loss.

    Args:
        input_df: DataFrame containing initial and final codability levels.
        left_col: Name of the column representing initial codability levels.
        right_col: Name of the column representing final codability levels.
        gain_col: Name of the column indicating codability gain (True/False) used for color.
            If None, all links will be grey.
        title_suffix: Suffix to add to the title of the figure, e.g. section name.

    Return:
        A Plotly Figure object representing the Sankey diagram, or None if columns not found.
    """
    required_columns = [left_col, right_col]
    if gain_col:
        required_columns.append(gain_col)

    if not all(col in input_df.columns for col in required_columns):
        raise ValueError(
            f"Columns {left_col} or {right_col} not found in input DataFrame."
        )

    sankey_df = input_df.groupby(required_columns).size().reset_index()
    if not gain_col:
        gain_col = "gain_temp_col"
        sankey_df[gain_col] = False

    label_list = list(pd.unique(sankey_df[[left_col, right_col]].values.ravel("K")))
    # sort the list by value of number contained in the string
    label_list.sort(key=lambda x: -int(re.sub(r"\D", "", "0" + x)))

    # add proportion to label list
    label_list2 = (
        [
            lab
            + f""": {100 * sankey_df[sankey_df[left_col] == lab][0].sum() / sankey_df[0].sum():.1f}% ({
    sankey_df[sankey_df[left_col] == lab][0].sum()
        })"""
            for lab in label_list
        ]
        + [
            lab
            + f""": {100 * sankey_df[sankey_df[right_col] == lab][0].sum() / sankey_df[0].sum():.1f}% ({
            sankey_df[sankey_df[right_col] == lab][0].sum()
        })"""
            for lab in label_list
        ]
    )
    label_colors = ["#1a9641"] + ["#a6d96a"] * (len(label_list) - 2) + ["#fdae61"]
    link = {
        "source": sankey_df[left_col].apply(label_list.index).tolist(),
        "target": sankey_df[right_col]
        .apply(lambda x, label_list=label_list: label_list.index(x) + len(label_list))
        .tolist(),
        "color": sankey_df[gain_col]
        .apply(
            lambda x: (
                "rgba(166,217,106,0.3)"
                if x > 0
                else ("rgba(180,180,180,0.3)" if x == 0 else "rgba(253,174,97,0.3)")
            )
        )
        .tolist(),
        "value": sankey_df[0].tolist(),
    }
    link["hovertemplate"] = "Count: %{value}<extra></extra>"

    sankey_fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 15,
                    "thickness": 20,
                    "line": {"color": "black", "width": 0.5},
                    "color": label_colors * 2,
                    "label": label_list2,
                    "hovertemplate": "Count %{value}<extra></extra>",
                },
                link=link,
            )
        ]
    )
    # label the left and right sides
    sankey_fig.add_annotation(
        x=-0.05, y=1.05, text=left_col, showarrow=False, font={"size": 12}
    )
    sankey_fig.add_annotation(
        x=1.05, y=1.05, text=right_col, showarrow=False, font={"size": 12}
    )

    sankey_fig.update_layout(
        title_text=f"Impact of SurveyAssist Follow-up Q/A on Codability Levels {title_suffix}",
        font_size=10,
        height=600,
        width=600,
    )
    return sankey_fig


# %%
# create sankey diagram for CC codes
subset_msk = {
    "": cc_coded_df["batch_num"].isin([1, 2, 3]),  # clerical coding now finished
    "(follow-up subset)": cc_coded_df["survey_assist_open_question"].notna(),
}
for subset_name, msk in subset_msk.items():
    temp_df = (
        cc_coded_df[msk][
            [
                "cc_initial_codability_level",
                "cc_final_codability_level_open_q",
                "cc_codability_gain_open_q",
            ]
        ]
        .copy()
        .rename(
            columns={
                "cc_initial_codability_level": "Clerical Initial Codes",
                "cc_final_codability_level_open_q": "Clerical Final Codes - Open Question",
                "cc_codability_gain_open_q": "Codability Gain",
            }
        )
    )
    fig = create_sankey_codability_gain_loss(
        temp_df,
        left_col="Clerical Initial Codes",
        right_col="Clerical Final Codes - Open Question",
        gain_col="Codability Gain",
        title_suffix=f" {subset_name}",
    )
    fig.show()
    if out_dir:
        fig.write_image(f"{out_dir}/cc_codability_gain_sankey_{subset_name}.png")
        fig.write_html(f"{out_dir}/cc_codability_gain_sankey_{subset_name}.html")

# %%
# create sankey diagram for SA codes
for question_type in ["open", "closed"]:
    temp_df = (
        sa_coded_df[
            [
                "sa_initial_codability_level",
                f"sa_final_codability_level_{question_type}_q",
                f"sa_codability_gain_{question_type}_q",
            ]
        ]
        .copy()
        .rename(
            columns={
                "sa_initial_codability_level": "SA Initial Codes",
                f"sa_final_codability_level_{question_type}_q": "SA Final Codes - "
                + question_type.capitalize()
                + " Question",
                f"sa_codability_gain_{question_type}_q": "Codability Gain",
            }
        )
    )
    fig = create_sankey_codability_gain_loss(
        temp_df,
        right_col="SA Final Codes - " + question_type.capitalize() + " Question",
        gain_col="Codability Gain",
    )
    fig.show()
    if out_dir:
        fig.write_image(f"{out_dir}/sa_codability_gain_sankey_{question_type}_q.png")
        fig.write_html(f"{out_dir}/sa_codability_gain_sankey_{question_type}_q.html")


# %%
# same figures but for large sections only
section_sizes = sa_coded_df.most_likely_sic_section.value_counts(dropna=False)
size_thr = 100
large_sections = {x: [x] for x in section_sizes[section_sizes >= size_thr].index}
large_sections["Other"] = section_sizes[section_sizes < size_thr].index.tolist()

for section_name, sections in large_sections.items():
    large_sections_df = sa_coded_df[sa_coded_df.most_likely_sic_section.isin(sections)]
    print(
        f"Sankey diagrams for section {section_name} with {len(large_sections_df)} entries:"
    )
    for question_type in ["open", "closed"]:
        temp_df = large_sections_df[
            [
                "sa_initial_codability_level",
                f"sa_final_codability_level_{question_type}_q",
                f"sa_codability_gain_{question_type}_q",
            ]
        ].rename(
            columns={
                "sa_initial_codability_level": "SA Initial Codes",
                f"sa_final_codability_level_{question_type}_q": "SA Final Codes - "
                + question_type.capitalize()
                + " Question",
                f"sa_codability_gain_{question_type}_q": "Codability Gain",
            }
        )
        create_sankey_codability_gain_loss(
            temp_df,
            right_col="SA Final Codes - " + question_type.capitalize() + " Question",
            gain_col="Codability Gain",
            title_suffix=f" - Section {section_name}",
        ).show()


# %%
# plot user numbers vs time start
if "time_start" in sa_coded_df.columns:

    sa_coded_df["response_start_time"] = pd.to_datetime(sa_coded_df["time_start"])
    sa_coded_df["user_num"] = sa_coded_df["user"].map(lambda x: int(x[3:8]))

    # Prepare data
    sa_coded_df = sa_coded_df.sort_values("response_start_time")
    sa_coded_df["cum_user_num"] = range(1, len(sa_coded_df) + 1)

    # Create subplots with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Scatter plot for user_num
    scatter = px.scatter(
        sa_coded_df, x="response_start_time", y="user_num", template="simple_white"
    )
    fig.add_trace(scatter.data[0], secondary_y=False)

    # Line plot for cumulative user numbers
    line = px.line(
        sa_coded_df,
        x="response_start_time",
        y="cum_user_num",
        color_discrete_sequence=["orange"],
        template="simple_white",
    )
    fig.add_trace(line.data[0], secondary_y=True)

    # Update layout
    fig.update_layout(
        title="User Numbers vs Response Start Time",
        xaxis_title="Response Start Time",
        yaxis_title="User ID Number",
        yaxis2_title="Cumulative User Count",
        template="simple_white",
    )
    fig.show()

    if out_dir:
        fig.write_image(f"{out_dir}/user_numbers_vs_time_start.png")
        fig.write_html(f"{out_dir}/user_numbers_vs_time_start.html")


# %%
# extract specific proportions for the report
msk = sa_coded_df["survey_assist_open_question"].notna()
diff_df = sa_coded_df[msk].merge(cc_coded_df, on=["user"]).reset_index()

diff_df["cc_initial_coded"] = diff_df["cc_initial_codes"].apply(len) == 1
diff_df["cc_changed"] = diff_df["cc_initial_codes"].map(set) != diff_df[
    "cc_final_codes_open_q"
].map(set)
diff_df["sa_changed"] = diff_df["sa_initial_codes"].map(set) != diff_df[
    "sa_final_codes_open_q"
].map(set)

print("Overall SA codes change stats:")
print(diff_df["sa_changed"].aggregate(["mean", "sum", "count"]))

print(
    "SA codability gain > 0 proportion on whole dataset: ",
    (sa_coded_df["sa_codability_gain_open_q"] > 0).aggregate(["mean", "sum", "count"]),
)
print(
    "SA codability gain > 0 proportion on open q answered dataset: ",
    (diff_df["sa_codability_gain_open_q"] > 0).aggregate(["mean", "sum", "count"]),
)

print("\n Overall clerical codes change stats:")
print(diff_df["cc_changed"].aggregate(["mean", "sum", "count"]))
print(
    diff_df.groupby("cc_initial_coded")["cc_changed"].aggregate(
        ["mean", "sum", "count"]
    )
)

print(
    "CC codability gain > 0 proportion on whole dataset: ",
    (cc_coded_df["cc_codability_gain_open_q"] > 0).aggregate(["mean", "sum", "count"]),
)
print(
    "CC codability gain > 0 proportion on open q answered dataset: ",
    (diff_df["cc_codability_gain_open_q"] > 0).aggregate(["mean", "sum", "count"]),
)

# %%
