"""Notebook to compare clerical coding vs SurveyAssist closed_q responses.

Loads preprocessed data with both clerical and SA codings,
calculates various metrics and visualises them.
Expects environment variable PREPROD_DATA_BUCKET to be set.

Disabled check for too long lines (f strings) and variables names (uppercase for constants)
"""

# pylint: disable=C0301,C0103,R0801,W0106

# %%
import os

import dotenv
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import binomtest, chisquare

from notebooks.november_test.helper_load_data import load_data

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
# is the answer to the closed question in CC initial/final codes?
mask_closed_q_selected = (
    combined_df["sa_final_codes_closed_q"].apply(len) == 1
) & combined_df["survey_assist_open_question"].notna()
for method in ["sa", "cc"]:
    for stage in ["initial", "final"]:
        out_col_name = f"closed_q_in_{method}_{stage}_codes"
        cc_col_name = (
            f"{method}_initial_codes"
            if stage == "initial"
            else f"{method}_final_codes_open_q"
        )
        closed_df = combined_df[mask_closed_q_selected].copy()
        closed_df[out_col_name] = closed_df.apply(
            lambda row, cc_col_name=cc_col_name: row[
                "sa_final_codes_closed_q"
            ].issubset(row[cc_col_name]),
            axis=1,
        )
        prop_in_codes = closed_df[out_col_name].mean()
        print(
            f"Proportion of closed question codes found in {method} {stage} codes: "
            f"{prop_in_codes:.3f} ({closed_df.shape[0]} records considered)"
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

tmp_df = combined_df[combined_df["survey_assist_open_question"].notna()].copy()
tmp_df["closed_q_num_options"] = tmp_df["sa_initial_codes"].apply(len)
tmp_df["sa_closed_q_responded"] = tmp_df["sa_final_codes_closed_q"].apply(len) > 0
print(tmp_df.groupby(tmp_df["closed_q_num_options"])["sa_closed_q_responded"].mean())

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

# chi-square for how likely they are to select none of the above by option size
obs_counts = tmp_df.groupby(tmp_df["closed_q_num_options"])[
    "sa_closed_q_responded"
].sum()
exp_counts = (
    tmp_df.groupby(tmp_df["closed_q_num_options"]).size()
    * tmp_df["sa_closed_q_responded"].mean()
)
chisq_nota = chisquare(obs_counts, f_exp=exp_counts)
print("Chi-square test for selecting none of the above by option size: ")
print(f"  statistic={chisq_nota.statistic:.2f}, p-value={chisq_nota.pvalue:.4f} ")

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
    rows=2,
    cols=len(option_sizes),
    shared_yaxes=True,
    horizontal_spacing=0.1,
    column_widths=col_widths,
)
for i, option_size in enumerate(option_sizes):
    y_min = all_groups_ci.loc[option_size, "ci_low"]
    y_max = all_groups_ci.loc[option_size, "ci_upp"]
    for row_num in [1, 2]:
        fig.add_shape(
            type="rect",
            x0=0.2,
            x1=option_size + 0.8,
            y0=y_min,
            y1=y_max,
            fillcolor=px.colors.qualitative.D3[2],
            opacity=0.15,
            line_width=0,
            row=row_num,
            col=i + 1,
            layer="below",
            name="Confidence Interval for Uniform Distribution",
            showlegend=(i == 0) & (row_num == 1),
        )
    sub_df = plot_df[plot_df["closed_q_num_options"] == option_size]
    bar_fig = px.bar(
        sub_df,
        x="rank",
        y="count",
        color="method",
        barmode="group",
        facet_row="method",
        color_discrete_sequence=px.colors.qualitative.D3,
    )
    for trace in bar_fig.data:
        trace.showlegend = i == 0
        row_num = 2 if trace.xaxis == "x" else 1
        fig.add_trace(trace, row=row_num, col=i + 1)

    fig.update_xaxes(
        title_text=f"Rank out of {option_size} options", dtick=1, row=2, col=i + 1
    )
    fig.update_xaxes(title_text="", dtick=1, row=1, col=i + 1)
    fig.update_yaxes(title_text="Count", row=1, col=1, range=[0, 151])
    fig.update_yaxes(title_text="Count", row=2, col=1, range=[0, 151])

    fig.update_layout(
        title_text="Closed Question Response Rank Distributions",
        template="plotly_white",
        width=900,
        height=500,
        legend={"x": 0.64, "y": 1.21},
    )

fig.show()

if out_dir:
    fig.write_image(out_dir + "/closed_q_rank_distributions.png")
    fig.write_html(out_dir + "/closed_q_rank_distributions.html")


# %%

# how often 'none of the above' and cc/sa final code was listed?
method = "cc"
msk = (
    combined_df[f"{method}_final_codability_level_open_q"] == "Sub-class (5-digits)"
) & ~combined_df["survey_assist_open_question"].isna()
combined_df[f"{method}_final_codes_open_q_within_offered_options"] = combined_df.apply(
    lambda row: row[f"{method}_final_codes_open_q"].issubset(row["sa_initial_codes"]),
    axis=1,
)
combined_df[f"{method}_final_codes_open_q_vs_selected_by_user_in_closed"] = (
    combined_df.apply(
        lambda row: (
            "none of the above"
            if len(row["sa_final_codes_closed_q"]) == 0
            else (
                "same code selected"
                if row[f"{method}_final_codes_open_q"].issubset(
                    row["sa_final_codes_closed_q"]
                )
                else "different selected"
            )
        ),
        axis=1,
    )
)
combined_df[msk].groupby(
    [
        f"{method}_final_codes_open_q_vs_selected_by_user_in_closed",
        f"{method}_final_codes_open_q_within_offered_options",
    ]
).size().unstack(fill_value=0)

# %%
msk = (
    (combined_df["cc_final_codability_level_open_q"] == "Sub-class (5-digits)")
    & (combined_df["sa_final_codability_level_open_q"] == "Sub-class (5-digits)")
    & ~combined_df["survey_assist_open_question"].isna()
)
combined_df["both_final_codes_open_q_within_offered_options"] = combined_df.apply(
    lambda row: (
        "both equal and offered"
        if row["cc_final_codes_open_q"] == row["sa_final_codes_open_q"]
        and row["cc_final_codes_open_q"].issubset(row["sa_initial_codes"])
        else (
            "both offered (separately)"
            if row["cc_final_codes_open_q"].issubset(row["sa_initial_codes"])
            and row["sa_final_codes_open_q"].issubset(row["sa_initial_codes"])
            else (
                "only cc offered"
                if row["cc_final_codes_open_q"].issubset(row["sa_initial_codes"])
                else (
                    "only sa offered"
                    if row["sa_final_codes_open_q"].issubset(row["sa_initial_codes"])
                    else "neither offered"
                )
            )
        )
    ),
    axis=1,
)
combined_df["both_final_codes_open_q_vs_selected_by_user_in_closed"] = (
    combined_df.apply(
        lambda row: (
            "none of the above"
            if len(row["sa_final_codes_closed_q"]) == 0
            else (
                "same code selected"
                if row["cc_final_codes_open_q"].issubset(row["sa_final_codes_closed_q"])
                and row["sa_final_codes_open_q"].issubset(
                    row["sa_final_codes_closed_q"]
                )
                else (
                    "sa_code selected"
                    if row["sa_final_codes_open_q"].issubset(
                        row["sa_final_codes_closed_q"]
                    )
                    else (
                        "cc_code selected"
                        if row["cc_final_codes_open_q"].issubset(
                            row["sa_final_codes_closed_q"]
                        )
                        else "different selected"
                    )
                )
            )
        ),
        axis=1,
    )
)
combined_df[msk].groupby(
    [
        "both_final_codes_open_q_vs_selected_by_user_in_closed",
        "both_final_codes_open_q_within_offered_options",
    ]
).size().unstack(fill_value=0)


# %%
def iter_options(row):
    """Iterate over closed question options in a row."""
    if len(row["cc_final_codes_open_q"]) != 1:
        return None
    for j in range(5):
        if (
            next(iter(row["cc_final_codes_open_q"]))
            == row[f"survey_assist_closed_question_option_{j+1}_code"]
        ):
            return row[f"survey_assist_closed_question_option_{j+1}"].lower()
    return None


closed_df["cc_final_code_rephrased"] = closed_df.apply(iter_options, axis=1)

examples_mask = (
    closed_df["cc_final_codes_open_q"] != closed_df["sa_final_codes_closed_q"]
) & closed_df["cc_final_code_rephrased"].notna()

examples_df = closed_df.loc[
    examples_mask,
    [
        "unique_id",
        "job_title",
        "job_description",
        "org_description",
        "survey_assist_open_question",
        "survey_assist_open_question_response",
        "survey_assist_closed_question_response",
        "cc_final_code_rephrased",
        "sa_final_codes_closed_q",
        "cc_final_codes_open_q",
    ],
]

pairs = (
    examples_df.groupby(
        [
            "survey_assist_closed_question_response",
            "cc_final_code_rephrased",
        ]
    )
    .size()
    .sort_values(ascending=False)
)
print(pairs[pairs > 1])

examples_df[examples_df["cc_final_code_rephrased"].isin(pairs.index[0])]

# %%
