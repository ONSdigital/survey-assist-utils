"""Notebook to quantitative feedback analysis."""

# pylint: disable=C0301,C0103,R0801,C0121

# %%
import os

import dotenv
import pandas as pd
import plotly.express as px
from scipy.stats import kendalltau, kruskal, mannwhitneyu, spearmanr

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
sa_coded_df = pd.read_parquet(
    work_dir + "/evaluation_df_with_sa_clean_codes_and_sic_section.parquet"
)
sa_coded_df = sa_coded_df[sa_coded_df["feedback_survey_ease"] != ""].reset_index(
    drop=True
)
# %%
# flags to test association woth feedback
sa_coded_df["dynamic_q"] = sa_coded_df["survey_assist_open_question"].notna()
sa_coded_df["nota"] = (
    sa_coded_df["survey_assist_closed_question_response"] == "none of the above"
)
sa_coded_df.loc[~sa_coded_df["dynamic_q"], "nota"] = None
sa_coded_df["generic_open_q"] = sa_coded_df[
    "survey_assist_open_question"
].str.startswith("What is your employer's main business activity?")
groups = [
    "feedback_age_range",
    "dynamic_q",
    "SIC Section",
    "nota",
    "generic_open_q",
]

# %%
feedback_cols = [
    "feedback_survey_ease",
    "feedback_survey_relevance",
    "feedback_survey_comfort",
]

encode_dict = {
    "very easy": 5,
    "easy": 4,
    "neither easy or difficult": 3,
    "difficult": 2,
    "very difficult": 1,
    "very relevant": 5,
    "relevant": 4,
    "neither relevant or irrelevant": 3,
    "irrelevant": 2,
    "very irrelevant": 1,
    "very comfortable": 5,
    "comfortable": 4,
    "neither comfortable or uncomfortable": 3,
    "uncomfortable": 2,
    "very uncomfortable": 1,
    "": None,
    "16-24": 1,
    "25-34": 2,
    "35-49": 3,
    "50-64": 4,
    "65-plus": 5,
}

for col in [*feedback_cols, "feedback_age_range"]:
    sa_coded_df[col + "_encoded"] = sa_coded_df[col].map(encode_dict)

# %%
word_dict = {
    5: "very positive",
    4: "positive",
    3: "neutral",
    2: "negative",
    1: "very negative",
}
titles = {
    "nota": '"None of the above" response to closed question',
    "dynamic_q": "Presented with dynamic question",
    "feedback_age_range": "Age groups",
    "SIC Section": "SIC sections",
}
for gr in groups:
    df_list = []
    for col in feedback_cols:
        tmp_df = (
            sa_coded_df.fillna("NA")
            .groupby([gr, col + "_encoded"])["user"]
            .count()
            .reset_index()
            .rename(columns={col + "_encoded": "value_num", "user": "count"})
        )
        tmp_df["percent"] = tmp_df.groupby(gr)["count"].transform(
            lambda x: x / x.sum() * 100
        )
        tmp_df["feedback_type"] = col
        tmp_df["value"] = tmp_df["value_num"].map(word_dict)
        df_list.append(tmp_df)
    plot_df = (
        pd.concat(df_list).sort_values(by=["value_num", gr]).reset_index(drop=True)
    )
    # plot_df.loc[plot_df[gr].isna(), gr] = 'NA'
    fig = px.bar(
        plot_df,
        x=gr,
        y="percent",
        color="value",
        title=f"Quantitative Feedback by {gr}",
        hover_data={"count": True, "percent": ":.2f"},
        facet_col="feedback_type",
        facet_col_spacing=0.04,
        barmode="stack",
        template="plotly_white",
    )
    for annotation in fig.layout.annotations:
        if "feedback_type=" in annotation.text:
            annotation.text = annotation.text.split("=feedback_survey_")[1].strip()
    fig.update_xaxes(title_text="")
    fig.update_xaxes(title_text=titles.get(gr, gr), row=1, col=2)
    fig.update_layout(
        legend={"traceorder": "reversed"},
        width=550 + plot_df[gr].nunique() * 40,
        height=400,
    )
    fig.show()


# %%
# non-parametric tests for association
bonferroni_correction = len(groups) * len(feedback_cols)
print(
    f"Confidence threshold with Bonferroni correction: {0.05/bonferroni_correction:.4f}"
)

for gr in groups:
    print(f"\n=== Feedback by {gr} ===")
    for feedback_col in feedback_cols:
        print(f" - {feedback_col}:")
        if sa_coded_df[gr].nunique() == 2:  # noqa:PLR2004
            print(
                mannwhitneyu(
                    sa_coded_df.loc[
                        sa_coded_df[gr] == True, feedback_col + "_encoded"  # noqa:E712
                    ],
                    sa_coded_df.loc[
                        sa_coded_df[gr] == False, feedback_col + "_encoded"  # noqa:E712
                    ],
                    alternative="two-sided",
                )
            )
            msk = sa_coded_df[gr].notna()
            print(
                kendalltau(
                    sa_coded_df.loc[msk, feedback_col + "_encoded"],
                    sa_coded_df.loc[msk, gr].astype(int),
                )
            )
            print(
                spearmanr(
                    sa_coded_df.loc[msk, feedback_col + "_encoded"],
                    sa_coded_df.loc[msk, gr].astype(int),
                )
            )
        else:
            print(
                kruskal(
                    *[
                        sa_coded_df.loc[
                            sa_coded_df[gr] == cat, feedback_col + "_encoded"
                        ]
                        for cat in sa_coded_df[gr].unique()
                    ]
                )
            )
            if gr + "_encoded" in sa_coded_df.columns:
                print(
                    kendalltau(
                        sa_coded_df[feedback_col + "_encoded"],
                        sa_coded_df[gr + "_encoded"],
                    )
                )
                print(
                    spearmanr(
                        sa_coded_df[feedback_col + "_encoded"],
                        sa_coded_df[gr + "_encoded"],
                    )
                )
# %%
