"""Notebook to compare clerical coding vs SurveyAssist closed_q responses.

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
from plotly.subplots import make_subplots
from scipy.stats import binomtest, chisquare

from survey_assist_utils.data_cleaning.prep_data import (
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


# %%
# is the answer to the closed question in CC initial/final codes?
mask_closed_q_selected = combined_df["sa_final_codes_closed_q"].apply(len) == 1
cc_coded_batches = {"initial": (1, 2), "final": (1, 2)}
for method in ["sa", "cc"]:
    for stage, batches in cc_coded_batches.items():
        out_col_name = f"closed_q_in_{method}_{stage}_codes"
        cc_col_name = (
            f"{method}_initial_codes"
            if stage == "initial"
            else f"{method}_final_codes_open_q"
        )
        closed_df = combined_df[
            mask_closed_q_selected & (combined_df["batch_num"].isin(batches))
        ].copy()
        closed_df[out_col_name] = closed_df.apply(
            lambda row, cc_col_name=cc_col_name: row[
                "sa_final_codes_closed_q"
            ].issubset(row[cc_col_name]),
            axis=1,
        )
        prop_in_codes = closed_df[out_col_name].mean()
        print(
            f"Proportion of closed question codes found in {method} {stage} codes "
            f"for batches {batches}: {prop_in_codes:.3f} "
            f"({closed_df.shape[0]} records considered)"
        )

# %%
# get rank distributions

closed_df = combined_df[mask_closed_q_selected].copy()

closed_df["closed_q_num_options"] = closed_df["sa_initial_codes"].apply(len)
print(closed_df["closed_q_num_options"].value_counts())

closed_df["closed_q_rank_disp"] = closed_df.apply(
    lambda row: min(
        i + 1
        for i in range(5)
        if row["survey_assist_closed_question_response_code"]
        == row[f"survey_assist_closed_question_option_{i+1}_code"]
    ),
    axis=1,
)
print(closed_df["closed_q_rank_disp"].value_counts())

closed_df["closed_q_rank_llm"] = closed_df.apply(
    lambda row: min(
        i + 1
        for i in range(5)
        if next(iter(row["sa_final_codes_closed_q"]))
        == row[f"survey_assist_alt_candidate_code_{i+1}"]
    ),
    axis=1,
)
print(closed_df["closed_q_rank_llm"].value_counts())


# %%
# chi-square tests for rank distributions by option size
for option_size in sorted(closed_df["closed_q_num_options"].unique()):
    sub_df = closed_df[closed_df["closed_q_num_options"] == option_size]
    chisq_disp = chisquare(sub_df["closed_q_rank_disp"].value_counts().sort_index())
    chisq_llm = chisquare(sub_df["closed_q_rank_llm"].value_counts().sort_index())
    print(
        f"Chi-square test for option size {option_size} ({sub_df.shape[0]} records): "
    )
    print(
        f"  Display rank: statistic={chisq_disp.statistic:.2f}, p-value={chisq_disp.pvalue:.4f} "
    )
    print(
        f"  LLM rank: statistic={chisq_llm.statistic:.2f}, p-value={chisq_llm.pvalue:.4f} "
    )

all_groups = (
    (
        closed_df.groupby("closed_q_num_options")["closed_q_rank_disp"]
        .value_counts()
        .reset_index()
        .rename(columns={"count": "count_disp", "closed_q_rank_disp": "rank"})
    )
    .merge(
        closed_df.groupby("closed_q_num_options")["closed_q_rank_llm"]
        .value_counts()
        .reset_index()
        .rename(columns={"count": "count_llm", "closed_q_rank_llm": "rank"})
    )
    .merge(
        closed_df.groupby("closed_q_num_options")
        .size()
        .reset_index()
        .rename(columns={0: "count_group"})
    )
    .sort_values(by=["closed_q_num_options", "rank"], ascending=[False, True])
    .reset_index(drop=True)
)
all_groups["count_expected"] = (
    all_groups["count_group"] / all_groups["closed_q_num_options"]
)

# overall chi-square test
chisq_disp = chisquare(all_groups["count_disp"], f_exp=all_groups["count_expected"])
chisq_llm = chisquare(all_groups["count_llm"], f_exp=all_groups["count_expected"])
print("Chi-square test across all option sizes: ")
print(
    f"  Display rank: statistic={chisq_disp.statistic:.2f}, p-value={chisq_disp.pvalue:.4f} "
)
print(
    f"  LLM rank: statistic={chisq_llm.statistic:.2f}, p-value={chisq_llm.pvalue:.4f} "
)

# %%
# visualise rank distributions
plot_df = all_groups.rename(
    columns={
        "count_disp": "(Randomised) Display Ordering",
        "count_llm": "SurveyAssist Ordering",
    }
).melt(
    id_vars=["closed_q_num_options", "rank"],
    value_vars=[
        "(Randomised) Display Ordering",
        "SurveyAssist Ordering",
    ],
    var_name="method",
    value_name="count",
)


def compute_binom_ci(row, alpha=0.05, num_trials=5 + 4 + 3 + 2):
    """Compute binomial confidence interval for expected counts under uniform distribution."""
    ci = binomtest(int(row["count_expected"]), int(row["count_group"])).proportion_ci(
        confidence_level=1 - alpha / num_trials  # bonferroni correction
    )
    return pd.Series(
        {
            "closed_q_num_options": row["closed_q_num_options"],
            "ci_low": ci.low * row["count_group"],
            "ci_upp": ci.high * row["count_group"],
        }
    )


all_groups_ci = (
    all_groups.apply(compute_binom_ci, axis=1)
    .drop_duplicates()
    .set_index("closed_q_num_options")
)

# %%
option_sizes = [5, 4, 3, 2]
col_widths = [size / sum(option_sizes) for size in option_sizes]

fig = make_subplots(
    rows=1,
    cols=len(option_sizes),
    shared_yaxes=True,
    horizontal_spacing=0.1,
    column_widths=col_widths,
)
for i, option_size in enumerate(option_sizes):
    y_min = all_groups_ci.loc[option_size, "ci_low"]
    y_max = all_groups_ci.loc[option_size, "ci_upp"]
    fig.add_shape(
        type="rect",
        x0=0.2,
        x1=option_size + 0.8,
        y0=y_min,
        y1=y_max,
        fillcolor=px.colors.qualitative.D3[2],
        opacity=0.15,
        line_width=0,
        row=1,
        col=i + 1,
        layer="below",
        name="Confidence Interval for Uniform Distribution",
        showlegend=i == 0,
    )
    sub_df = plot_df[plot_df["closed_q_num_options"] == option_size]
    bar_fig = px.bar(
        sub_df,
        x="rank",
        y="count",
        color="method",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.D3,
    )
    for trace in bar_fig.data:
        trace.showlegend = i == 0
        fig.add_trace(trace, row=1, col=i + 1)
        fig.update_xaxes(
            title_text=f"Rank out of {option_size} options", dtick=1, row=1, col=i + 1
        )

    fig.update_layout(
        title_text="Closed Question Response Rank Distributions",
        template="plotly_white",
        width=900,
        height=500,
        legend={"x": 0.64, "y": 1.21},
    )
    fig.update_yaxes(title_text="Count", row=1, col=1)

fig.show()

if out_dir:
    fig.write_image(out_dir + "/closed_q_rank_distributions.png")

# %%
