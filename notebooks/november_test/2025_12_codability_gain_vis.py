"""Notebook to visualise the codability gain/loss using a Sankey diagram."""

# pylint: disable=C0301,C0103,R0801
# %%
import os
import re

import dotenv
import pandas as pd
import plotly.graph_objects as go

data_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET") or ""

# %%
folder = data_bucket + "analysis-interim-results"
out_dir = (
    "data/figures/"  # needs local folder unfortunately, set to None to skip saving
)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

combined_df = pd.read_parquet(folder + "/evaluation_df_with_sa_clean_codes.parquet")


# %%
def create_sankey_codability_gain_loss(
    input_df: pd.DataFrame,
    left_col="SA Initial Codes",
    right_col="SA Final Codes",
    title_suffix: str = "",
) -> go.Figure:
    """Create a Sankey diagram to visualise codability gain/loss.

    Args:
        input_df: DataFrame containing initial and final codability levels.
        left_col: Name of the column representing initial codability levels.
        right_col: Name of the column representing final codability levels.
        out_dir: Directory to save the figure, if provided.
        title_suffix: Suffix to add to the title of the figure, e.g. section name.

    Return:
        A Plotly Figure object representing the Sankey diagram, or None if columns not found.
    """
    if left_col not in input_df.columns or right_col not in input_df.columns:
        raise ValueError(
            f"Columns {left_col} or {right_col} not found in input DataFrame."
        )

    sankey_df = input_df.groupby([left_col, right_col]).size().reset_index()

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
        "value": sankey_df[0].tolist(),
    }
    link["color"] = [
        (
            "rgba(253,174,97,0.3)"
            if (link["target"][i] - len(label_list) > link["source"][i])
            else (
                "rgba(166,217,106,0.3)"
                if (link["target"][i] - len(label_list) < link["source"][i])
                else "rgba(180,180,180,0.3)"
            )
        )
        for i in range(len(link["value"]))
    ]
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
# create sankey diagram
for question_type in ["open", "closed"]:
    temp_df = combined_df[
        ["sa_initial_codability_level", f"sa_final_codability_level_{question_type}_q"]
    ].rename(
        columns={
            "sa_initial_codability_level": "SA Initial Codes",
            f"sa_final_codability_level_{question_type}_q": "SA Final Codes - "
            + question_type.capitalize()
            + " Question",
        }
    )
    fig = create_sankey_codability_gain_loss(
        temp_df,
        right_col="SA Final Codes - " + question_type.capitalize() + " Question",
    )
    fig.show()
    if out_dir:
        fig.write_image(
            f"{out_dir}/sankey_codability_gain_loss_sa_followup_{question_type}_q.png"
        )

# %%
# same figures but for large sections only
section_sizes = combined_df.most_likely_sic_section.value_counts(dropna=False)
size_thr = 100
large_sections = {x: [x] for x in section_sizes[section_sizes >= size_thr].index}
large_sections["Other"] = section_sizes[section_sizes < size_thr].index.tolist()

for section_name, sections in large_sections.items():
    large_sections_df = combined_df[combined_df.most_likely_sic_section.isin(sections)]
    print(
        f"Sankey diagrams for section {section_name} with {len(large_sections_df)} entries:"
    )
    for question_type in ["open", "closed"]:
        temp_df = large_sections_df[
            [
                "sa_initial_codability_level",
                f"sa_final_codability_level_{question_type}_q",
            ]
        ].rename(
            columns={
                "sa_initial_codability_level": "SA Initial Codes",
                f"sa_final_codability_level_{question_type}_q": "SA Final Codes - "
                + question_type.capitalize()
                + " Question",
            }
        )
        create_sankey_codability_gain_loss(
            temp_df,
            right_col="SA Final Codes - " + question_type.capitalize() + " Question",
            title_suffix=f" - Section {section_name}",
        ).show()

# %%
