"""Notebook for visualizing impact of Surveyassist on codability levels using a Sankey diagram."""

# pylint: disable=C0301,C0103,R0801
# %%
import logging
import os
import re

import dotenv
import pandas as pd
import plotly.graph_objects as go

from survey_assist_utils.data_cleaning.prep_data import prep_model_codes
from survey_assist_utils.data_cleaning.sic_codes import get_codability_level

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
# load data and preprocess
model_file = f"{bucket_prefix}two_prompt_pipeline/2025_09_full_2k_gemini25/STG5.parquet"
model_df = pd.read_parquet(model_file)

DIGITS = 5
initial_codes = prep_model_codes(
    model_df,
    digits=DIGITS,
    codes_col="initial_code",
    alt_codes_col="alt_sic_candidates",
    out_col="sa_initial_codes",
)
final_codes = prep_model_codes(
    model_df,
    digits=DIGITS,
    codes_col="final_sic",
    alt_codes_col="higher_level_final_sic",
    out_col="sa_final_codes",
)

combined_df = initial_codes.merge(final_codes, on="unique_id", how="inner")

# %%

left_col = "SA Initial Codes"
right_col = "SA Final Codes"  # "clerical_codes"
combined_df[left_col] = combined_df["sa_initial_codes"].apply(get_codability_level)
combined_df[right_col] = combined_df["sa_final_codes"].apply(get_codability_level)

sankey_df = combined_df.groupby([left_col, right_col]).size().reset_index()

# %%
# create sankey diagram
label_list = list(pd.unique(sankey_df[[left_col, right_col]].values.ravel("K")))
# sort the list by value of number contained in the string
label_list.sort(key=lambda x: -int(re.sub(r"\D", "", "0" + x)))

# add proportion to label list
label_list2 = [
    lab
    + f" {100 * sankey_df[sankey_df[left_col] == lab][0].sum() / sankey_df[0].sum():.1f}%"
    for lab in label_list
] + [
    lab
    + f" {100 * sankey_df[sankey_df[right_col] == lab][0].sum() / sankey_df[0].sum():.1f}%"
    for lab in label_list
]

label_colors = ["#1a9641"] + ["#a6d96a"] * (len(label_list) - 2) + ["#fdae61"]
link = {
    "source": sankey_df[left_col].apply(label_list.index).tolist(),
    "target": sankey_df[right_col]
    .apply(lambda x: label_list.index(x) + len(label_list))
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

# %%
