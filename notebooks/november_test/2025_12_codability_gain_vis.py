"""Notebook to visualise the codability gain/loss using a Sankey diagram."""

# pylint: disable=C0301,C0103,R0801
# %%
import re

import dotenv
import pandas as pd
import plotly.graph_objects as go

data_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET") or ""

dotenv.load_dotenv("../../.env")

# %%
folder = data_bucket + "data/2025-12-05-export"
out_dir = folder + "/figures/"  # set to None to skip saving

combined_df = pd.read_parquet(folder + "/evaluation_df_with_sa_clean_codes.parquet")

# %%
# create sankey diagram
for question_type in ["closed", "open"]:
    left_col = "SA Initial Codes"
    right_col = (
        f"SA Final Codes - {question_type.capitalize()} Question"  # "clerical_codes"
    )
    sankey_df = combined_df.copy()
    sankey_df[left_col] = sankey_df["sa_initial_codability_level"]
    sankey_df[right_col] = sankey_df[f"sa_final_codability_level_{question_type}_q"]

    sankey_df = sankey_df.groupby([left_col, right_col]).size().reset_index()

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
        title_text="Impact of SurveyAssist Follow-up Q/A on Codability Levels",
        font_size=10,
        height=600,
        width=600,
    )
    sankey_fig.show()
    if out_dir:
        sankey_fig.write_image(
            f"{out_dir}/sankey_codability_gain_loss_sa_followup_{question_type}_q.png"
        )

# %%
