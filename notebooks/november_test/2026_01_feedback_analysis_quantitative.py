# %%
"""Notebook to analyse feedback responses.

Note: ### = commented out to pass linting

Using 'type: ignore' to satisfy mypy checks.
"""

# pylint: disable=C0301,C0103,R0801,C0302,C0121

from os import makedirs
from textwrap import wrap
from typing import Any

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from helper_load_data import load_data
from matplotlib.gridspec import GridSpec
from scipy.stats import kruskal, mannwhitneyu
from scipy.stats import t as stats_t  # type: ignore[attr-defined]

from survey_assist_utils.data_cleaning.sic_codes import (
    get_clean_n_digit_codes,
)

# %%
# Load environmental variables & set data input/output locations:
## %matplotlib inline

data_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET") or ""

small_nonzero_number = 1e-256
SIGNIFICANCE_THRESHOLD = 0.05

folder = data_bucket + "analysis-interim-results"

out_folder = (
    data_bucket + "analysis-interim-results/feedback-analysis"
)  # set to None to skip saving
out_folder = None  # type: ignore[assignment]

figures_output_folder = "data/figures/quantitative_feedback_analysis/"
makedirs(figures_output_folder, exist_ok=True)

# %%
# Read data exported from firesore
eval_df = pd.read_parquet(folder + "/evaluation_df_with_sa_clean_codes.parquet")

# %%
# load & merge evaluation datasets
eval_df = load_data(folder)


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
    _section_counts = pd.Series(codes).value_counts()
    freq_sections = _section_counts[_section_counts == _section_counts.max()].index
    if len(freq_sections) == 1:
        return freq_sections[0]
    return row["most_likely_sic_section"]


eval_df.loc[:, "SIC Section"] = eval_df.apply(extract_sic_section, axis=1)

# %%
# Convert feedback responses to scores:

feedback_cols = [
    "feedback_survey_ease",
    "feedback_survey_relevance",
    "feedback_survey_comfort",
]

feedback_score_cols = ["ease_score", "relevance_score", "comfort_score"]
valid_responses_df = eval_df[eval_df["response_valid"]]
feedback_given_df = valid_responses_df[
    valid_responses_df["feedback_survey_ease"].apply(len) > 0
].copy()

ease_map = {
    "very easy": 5,
    "easy": 4,
    "neither easy or difficult": 3,
    "difficult": 2,
    "very difficult": 1,
}

relevance_map = {
    "very relevant": 5,
    "relevant": 4,
    "neither relevant or irrelevant": 3,
    "irrelevant": 2,
    "very irrelevant": 1,
}

comfort_map = {
    "very comfortable": 5,
    "comfortable": 4,
    "neither comfortable or uncomfortable": 3,
    "uncomfortable": 2,
    "very uncomfortable": 1,
}

feedback_given_df.loc[:, "ease_score"] = feedback_given_df[
    "feedback_survey_ease"
].apply(lambda r: ease_map[r])
feedback_given_df.loc[:, "relevance_score"] = feedback_given_df[
    "feedback_survey_relevance"
].apply(lambda r: relevance_map[r])
feedback_given_df.loc[:, "comfort_score"] = feedback_given_df[
    "feedback_survey_comfort"
].apply(lambda r: comfort_map[r])

# %%
print(
    f"{len(feedback_given_df)} respondents submitted feedback ({100*len(feedback_given_df)/len(eval_df):.3f}%)"
)
feedback_given_df[feedback_score_cols].describe()

# %%

# Aggregated summary statistics for quantitative feedback:

midpt_score = 3
for col in feedback_score_cols:
    print(
        f"{col}: +ve = {100*len(feedback_given_df[feedback_given_df[col] > midpt_score])/len(feedback_given_df):.3f}%"
    )
    print(
        f"{col}: neutral = {100*len(feedback_given_df[feedback_given_df[col] == midpt_score])/len(feedback_given_df):.3f}%"
    )
    print(
        f"{col}: -ve = {100*len(feedback_given_df[feedback_given_df[col] < midpt_score])/len(feedback_given_df):.3f}%"
    )

# %%

# Identify responses where dynamic questions were used:


def mark_questions_asked(row):
    """Helper function to mark cases responses where dynamic questions were asked."""
    if row["direct_lookup_classified"] or row["survey_assist_classified"]:
        return False
    if row["survey_assist_open_question"] is not None:
        return True
    print(
        'row not directly classified or SA classified, but no open question recorded. Marking as "None".'
    )
    return None


feedback_given_df.loc[:, "additional_questions_asked"] = feedback_given_df.apply(
    mark_questions_asked, axis=1
)

path_cols = [
    "direct_lookup_classified",
    "survey_assist_classified",
    "additional_questions_asked",
]

feedback_given_df.loc[:, "survey_assist_classified"] = feedback_given_df[
    "survey_assist_classified"
].fillna(False)

# %%

# Plot Feedback Distributions (all):

ease_count_dict = feedback_given_df["feedback_survey_ease"].value_counts().to_dict()
relevance_count_dict = (
    feedback_given_df["feedback_survey_relevance"].value_counts().to_dict()
)
comfort_count_dict = (
    feedback_given_df["feedback_survey_comfort"].value_counts().to_dict()
)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 8), constrained_layout=True)
bar1 = ax.bar(
    [i - 0.375 + 0.25 for i in range(1, 6)],
    ease_count_dict.values(),
    color="#12436D",
    width=0.25,
    label="Survey Ease",
)
bar2 = ax.bar(
    [i - 0.375 + 2 * 0.25 for i in range(1, 6)],
    relevance_count_dict.values(),
    color="#28A197",
    width=0.25,
    label="Survey Relevance",
)
bar3 = ax.bar(
    [i - 0.375 + 3 * 0.25 for i in range(1, 6)],
    comfort_count_dict.values(),
    color="#801650",
    width=0.25,
    label="Survey Comfort",
)

for barplot in [bar1, bar2, bar3]:
    for bar in barplot:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height()}",
            ha="center",
            va="bottom",
            fontsize=14,
        )

ax.set_xticks(
    [i - 0.375 + 2 * 0.25 for i in range(1, 6)],
    labels=["very\npositive", "positive", "neutral", "negative", "very\nnegative"],
    rotation=0,
    fontsize=18,
)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.legend(fontsize=18)

ax.set_ylabel("Number of responses", fontsize=18)
ax.set_yticks([0, 100, 200, 300, 400, 500, 600])
ax.set_yticklabels([0, 100, 200, 300, 400, 500, 600], size=14)  # type: ignore[list-item]
ax.set_title(
    f"Feedback Score Distributions\n(All Respondents - n={len(feedback_given_df)})",
    fontsize=20,
)

plt.tight_layout()
plt.savefig(f"{figures_output_folder}quant_feedback_distributions_concise.png", dpi=275)

# %%

# Plot Feedback Distributions (dynamic only):

feedback_given_dyn_df = feedback_given_df[
    feedback_given_df["additional_questions_asked"]
]

ease_count_dict = feedback_given_dyn_df["feedback_survey_ease"].value_counts().to_dict()
relevance_count_dict = (
    feedback_given_dyn_df["feedback_survey_relevance"].value_counts().to_dict()
)
comfort_count_dict = (
    feedback_given_dyn_df["feedback_survey_comfort"].value_counts().to_dict()
)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 8), constrained_layout=True)
bar1 = ax.bar(
    [i - 0.375 + 0.25 for i in range(1, 6)],
    ease_count_dict.values(),
    color="#12436D",
    width=0.25,
    label="Survey Ease",
)
bar2 = ax.bar(
    [i - 0.375 + 2 * 0.25 for i in range(1, 6)],
    relevance_count_dict.values(),
    color="#28A197",
    width=0.25,
    label="Survey Relevance",
)
bar3 = ax.bar(
    [i - 0.375 + 3 * 0.25 for i in range(1, 6)],
    comfort_count_dict.values(),
    color="#801650",
    width=0.25,
    label="Survey Comfort",
)

for barplot in [bar1, bar2, bar3]:
    for bar in barplot:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height()}",
            ha="center",
            va="bottom",
            fontsize=14,
        )

ax.set_xticks(
    [i - 0.375 + 2 * 0.25 for i in range(1, 6)],
    labels=["very\npositive", "positive", "neutral", "negative", "very\nnegative"],
    rotation=0,
    fontsize=18,
)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.legend(fontsize=18)

ax.set_ylabel("Number of responses", fontsize=18)
ax.set_yticks([0, 100, 200, 300, 400])
ax.set_yticklabels([0, 100, 200, 300, 400], size=14)  # type: ignore[list-item]
ax.set_title(
    f"Feedback Score Distributions\n(Dynamic Question Respondents Only - n={len(feedback_given_dyn_df)})",
    fontsize=20,
)
plt.tight_layout()
plt.savefig(
    f"{figures_output_folder}quant_feedback_distributions_dynamic_only.png", dpi=275
)

# %%

# Visulaise feedback-feedback relationships:

key_variables = feedback_score_cols
dof_overall = len(feedback_given_df) - 2
correlations = feedback_given_df[key_variables].corr(method="kendall")
fisher_transformed_z_scores = np.log(np.sqrt((1 + correlations) / (1 - correlations)))
std_errs = 1 / np.sqrt(dof_overall - 1)
confidence_interval_zspace = 1.96 * std_errs  # Using 90% confidence interval
correlation_CI_lower = (np.exp(2 * (correlations - confidence_interval_zspace)) - 1) / (
    np.exp(2 * (correlations - confidence_interval_zspace)) + 1
)
correlation_CI_upper = (np.exp(2 * (correlations + confidence_interval_zspace)) - 1) / (
    np.exp(2 * (correlations + confidence_interval_zspace)) + 1
)

fig, ax = plt.subplots(figsize=(5, 5))
corr_mat = ax.imshow(correlations, vmax=1, vmin=-1, cmap="PRGn")
colour_change_limit = 0.75

for i in range(len(correlations)):
    for j in range(len(correlations)):
        if i != j:
            annotation = ""
            colour = (
                "w" if abs(correlations.to_numpy()[i, j]) > colour_change_limit else "k"
            )
            ax.text(
                j,
                i,
                r"$\tau=$"
                + f"{correlations.to_numpy()[i, j]:.2f}"
                + "\n"
                + r"CI $=$ "
                + f"({correlation_CI_lower.to_numpy()[i, j]:.2f}, "
                + f"{correlation_CI_upper.to_numpy()[i, j]:.2f})",
                ha="center",
                va="center",
                color=colour,
                fontsize=7,
            )

ax.set_xticks(
    range(len(key_variables)),
    labels=[k.replace("_", "\n") for k in key_variables],
    rotation=20,
)
ax.set_yticks(
    range(len(key_variables)),
    labels=[k.replace("_", "\n") for k in key_variables],
    rotation=0,
)
cbar = fig.colorbar(corr_mat, ax=ax, shrink=0.65, pad=0.02)
cbar.set_label(
    r"Correlation coefficient (Kendall $\tau$)"
    + "\n"
    + "(95% confidence intervals stated)"
)
plt.tight_layout()
plt.savefig(
    f"{figures_output_folder}corr_mat_feedback_kendall_CI.png",
    dpi=275,
    transparent=True,
)

# %%
# Visulaise impact of dynamic questions on feedback:

responses_by_path = [
    feedback_given_df[feedback_given_df["additional_questions_asked"]][
        feedback_score_cols
    ],
    feedback_given_df[~feedback_given_df["additional_questions_asked"]][
        feedback_score_cols
    ],
]

statistic, p_value = kruskal(*responses_by_path)  # type: Any, Any

eta_squared = (statistic - 6 + 1) / (len(feedback_given_df) - 6)

eta_squared_interpretations = [0.01, 0.06, 0.14]


def get_effect_size_interpretations(e2):
    """Convert eta^2 scores to descriptions."""
    if e2 < eta_squared_interpretations[0]:
        return "no noticeable effect"
    if e2 < eta_squared_interpretations[1]:
        return "weak effect"
    if e2 < eta_squared_interpretations[2]:
        return "moderate effect"
    return "strong effect"


for col_id, col in enumerate(feedback_score_cols):
    print(f"{col}:")
    print(f"Kruskal-Wallis H-statistic: {statistic[col_id]:.3f}")
    print(f"p-value: {p_value[col_id]:.3e}")
    print(
        f"eta^2 effect size: {eta_squared[col_id]:.3f} ({get_effect_size_interpretations(eta_squared[col_id])})\n"
    )

# %%
# Section-level analysis

min_count_sections = 30
section_counts = feedback_given_df["SIC Section"].value_counts()
print(section_counts, len(section_counts))
sections_large_enough = section_counts >= min_count_sections

big_sections = sorted(section_counts[sections_large_enough].keys())
small_sections = sorted(section_counts[~sections_large_enough].keys())

lcf_gp_1 = {"B", "D", "E"}
lcf_gp_2 = {"R", "S", "T"}

big_sections = sorted(set(big_sections).difference(lcf_gp_1, lcf_gp_2))
small_sections = sorted(set(small_sections).difference(lcf_gp_1, lcf_gp_2))

# %%
# Create 'dummy' columns for SIC codes & group small / suppressed groups
feedback_given_df_SIC_dummies = pd.get_dummies(
    feedback_given_df, columns=["SIC Section"], prefix="", prefix_sep=""
)

feedback_given_df_SIC_dummies[",".join(sorted(list(lcf_gp_1) + small_sections))] = (
    feedback_given_df_SIC_dummies.apply(
        lambda row: sum(
            row[col]
            for col in sorted(list(lcf_gp_1) + small_sections)
            if col in feedback_given_df_SIC_dummies.columns
        ),
        axis=1,
    )
)

feedback_given_df_SIC_dummies[",".join(sorted(lcf_gp_2))] = (
    feedback_given_df_SIC_dummies.apply(
        lambda row: sum(
            row[col] for col in lcf_gp_2 if col in feedback_given_df_SIC_dummies.columns
        ),
        axis=1,
    )
)

feedback_given_df_SIC_dummies = feedback_given_df_SIC_dummies.drop(
    columns=[
        c
        for c in (lcf_gp_1 | lcf_gp_2 | set(small_sections))
        if c in feedback_given_df_SIC_dummies.columns
    ]
)
ordered_SIC_sections = [
    *big_sections,
    ",".join(sorted(lcf_gp_2)),
    ",".join(sorted(list(lcf_gp_1) + small_sections)),
]

section_counts = feedback_given_df_SIC_dummies[ordered_SIC_sections].sum()
print(section_counts)

# %%
key_variables_SIC = feedback_score_cols.copy()
key_variables_SIC.extend([f"{sec}" for sec in ordered_SIC_sections])

# %%
# Statistical Tests for Feedback Scores for each SIC / SIC group

results_ease = [
    mannwhitneyu(
        feedback_given_df_SIC_dummies[feedback_given_df_SIC_dummies[sec] == 1][
            "ease_score"
        ],
        feedback_given_df_SIC_dummies[feedback_given_df_SIC_dummies[sec] == 0][
            "ease_score"
        ],
        alternative="two-sided",
    )
    for sec in ordered_SIC_sections
]

results_relevance = [
    mannwhitneyu(
        feedback_given_df_SIC_dummies[feedback_given_df_SIC_dummies[sec] == 1][
            "relevance_score"
        ],
        feedback_given_df_SIC_dummies[feedback_given_df_SIC_dummies[sec] == 0][
            "relevance_score"
        ],
        alternative="two-sided",
    )
    for sec in ordered_SIC_sections
]

results_comfort = [
    mannwhitneyu(
        feedback_given_df_SIC_dummies[feedback_given_df_SIC_dummies[sec] == 1][
            "comfort_score"
        ],
        feedback_given_df_SIC_dummies[feedback_given_df_SIC_dummies[sec] == 0][
            "comfort_score"
        ],
        alternative="two-sided",
    )
    for sec in ordered_SIC_sections
]

# Common Language Effect Size:
# CLES = U/n1n2

effect_strength_ease = [
    res.statistic
    / (
        len(feedback_given_df_SIC_dummies[feedback_given_df_SIC_dummies[sec] == 1])
        * len(feedback_given_df_SIC_dummies[feedback_given_df_SIC_dummies[sec] == 0])
    )
    for res, sec in zip(results_ease, ordered_SIC_sections)
]

effect_strength_relevance: Any = [
    res.statistic
    / (
        len(feedback_given_df_SIC_dummies[feedback_given_df_SIC_dummies[sec] == 1])
        * len(feedback_given_df_SIC_dummies[feedback_given_df_SIC_dummies[sec] == 0])
    )
    for res, sec in zip(results_relevance, ordered_SIC_sections)
]

effect_strength_comfort: Any = [
    res.statistic
    / (
        len(feedback_given_df_SIC_dummies[feedback_given_df_SIC_dummies[sec] == 1])
        * len(feedback_given_df_SIC_dummies[feedback_given_df_SIC_dummies[sec] == 0])
    )
    for res, sec in zip(results_comfort, ordered_SIC_sections)
]

effect_strengths_SIC = np.array(
    [effect_strength_ease, effect_strength_relevance, effect_strength_comfort]
)

p_values: Any = np.array(
    [
        [mw.pvalue for mw in fb]
        for fb in [results_ease, results_relevance, results_comfort]
    ]
)

U_statistics = np.array(
    [
        [mw.statistic for mw in fb]
        for fb in [results_ease, results_relevance, results_comfort]
    ]
)

# %%
# Visualising the feedback vs. SIC section analysis results

bonferroni_sign_thresh = SIGNIFICANCE_THRESHOLD / (3 * 9)

fig = plt.figure(figsize=(10, 4))
gs = GridSpec(10, 10, figure=fig, wspace=0.01, hspace=0.005)
colour_threshold = 0.1

ax = fig.add_subplot(gs[3:, :])
ax2 = fig.add_subplot(gs[:3, :])

corr_mat = ax.imshow(effect_strengths_SIC, cmap="PRGn", vmin=0.2, vmax=0.8)
ax.set_xticks(
    range(len(key_variables_SIC[3:])),
    labels=["\n".join(wrap(k, 8)) for k in key_variables_SIC[3:]],
    rotation=0,
)
ax.set_yticks(
    range(len(key_variables_SIC[:3])),
    labels=[k.replace("_", "\n") for k in key_variables_SIC[:3]],
    rotation=0,
)

for i in range(effect_strengths_SIC.shape[0]):
    for j in range(effect_strengths_SIC.shape[1]):
        # annotation = r"$p=$" + f"\n{p_values[i, j]:.3f}"
        annotation = f"{p_values[i, j]:.3f}\n{effect_strengths_SIC[i,j]:.2f}"
        colour = (
            "w" if abs(0.5 - effect_strengths_SIC[i, j]) > colour_threshold else "k"
        )
        if p_values[i, j] is not None:
            ax.text(
                j,
                i,
                annotation if p_values[i, j] < bonferroni_sign_thresh else "",
                ha="center",
                va="center",
                color=colour,
                fontsize=10,
            )

cbar = fig.colorbar(corr_mat, ax=ax, shrink=0.825, aspect=6, pad=0.01)
cbar.set_label("Effect Strength\n(CLES)")
ax.set_xlabel("SIC Section (Most Likely)")

observations_heatmap = ax2.imshow(
    np.array([section_counts[s] for s in key_variables_SIC[3:]]).reshape(1, -1),
    cmap="Greys",
)
section_cbar = fig.colorbar(
    observations_heatmap, ax=ax2, shrink=0.675, aspect=2.2, pad=0.01
)

for sec_idx, sc in enumerate(section_counts):
    colour = "w" if abs(sc) > 0.5 * np.max(section_counts) else "k"
    ax2.text(
        sec_idx,
        0,
        r"$n=$" + f"\n{sc}",
        ha="center",
        va="center",
        color=colour,
        fontsize=10,
    )

ax2.set_xticks(range(len(section_counts)), labels=[""] * len(section_counts))
ax2.set_yticks([])
section_cbar.set_label("Obs.")
section_cbar.set_ticks([50, 100, 150], labels=["50", "100", "150"], fontsize=7)
plt.savefig(
    f"{figures_output_folder}section_level_feedback_correlations_concise.png",
    dpi=275,
    transparent=True,
)

# %%
# Analysis of LLM wait-time impact on Feedback:

### Parsing LLM wait times:

llm_wait_time_df = pd.read_json(
    f"{folder}/logs_scraped_timing_data/time-in-dynamic-questions-08-Dec-1530_extracted.json"
)
# print(
#     llm_wait_time_df["person_id"].value_counts().max()
# )  # check for multiple llm interaction times per user

desired_columns = [
    *feedback_given_df.columns.to_list(),
    *llm_wait_time_df.columns.to_list()[1:],
]

feedback_given_llm_waittime_df = pd.merge(
    feedback_given_df,
    llm_wait_time_df,
    how="inner",
    left_on="user",
    right_on="person_id",
)[desired_columns]

# Handle the 3 cases where people were shown dynamic Qs, then restarted the survey
feedback_given_llm_waittime_df = feedback_given_llm_waittime_df[
    feedback_given_llm_waittime_df["additional_questions_asked"]
]

bonferroni_sign_thresh = SIGNIFICANCE_THRESHOLD / 3

key_variables = ["time_to_show_dynamic_question", *feedback_score_cols]

correlations = (
    feedback_given_llm_waittime_df[key_variables].corr(method="kendall").to_numpy()
)
correlations = correlations[:1, 1:]

uncertainties = (1 - correlations) / np.sqrt(len(feedback_given_df))
dof_overall = len(feedback_given_llm_waittime_df) - 2

fisher_transformed_z_scores = np.log(np.sqrt((1 + correlations) / (1 - correlations)))
std_errs = 1 / np.sqrt(dof_overall - 1)
confidence_interval_zspace = 1.96 * std_errs  # Using 95% confidence interval
correlation_CI_lower = (np.exp(2 * (correlations - confidence_interval_zspace)) - 1) / (
    np.exp(2 * (correlations - confidence_interval_zspace)) + 1
)
correlation_CI_upper = (np.exp(2 * (correlations + confidence_interval_zspace)) - 1) / (
    np.exp(2 * (correlations + confidence_interval_zspace)) + 1
)

t_values = correlations / uncertainties
p_values = 2 * stats_t.sf(np.abs(t_values), dof_overall)

annot_corrs = np.where(
    (p_values > small_nonzero_number) * (p_values < bonferroni_sign_thresh),
    np.round(correlations, 2).astype(str),
    "",
)

fig, ax = plt.subplots(figsize=(7, 3))
corr_mat = ax.imshow(correlations, vmax=1, vmin=-1, cmap="PRGn")

for i in range(correlations.shape[0]):
    for j in range(correlations.shape[1]):
        annotation = ""
        colour = "w" if abs(correlations[i, j]) > colour_change_limit else "k"
        if p_values[i, j] < bonferroni_sign_thresh:
            ax.text(
                j,
                i,
                r"$\tau=$"
                + annot_corrs[i, j]
                + "\n"
                + f"90% CI: [{correlation_CI_lower[i, j]:.2f}, {correlation_CI_upper[i, j]:.2f}]"
                + "\n"
                + r"$p=$"
                + f"{p_values[i, j]:.3f}",
                ha="center",
                va="center",
                color=colour,
                fontsize=10,
            )

ax.set_xticks(
    range(len(key_variables[1:])),
    labels=[k.replace("_", "\n") for k in key_variables[1:]],
    rotation=0,
    fontsize=12,
)
ax.set_yticks(
    range(len(key_variables[:1])),
    labels=[k.replace("_", "\n") for k in key_variables[:1]],
    rotation=0,
    fontsize=12,
)
cbar = fig.colorbar(corr_mat, ax=ax, shrink=0.65, aspect=7, pad=0.01)
cbar.set_label("Kendall correlation\ncoefficient " + r"($\tau$)")
plt.savefig(
    f"{figures_output_folder}corr_mat_feedback_LLM_waittime_kendall.png",
    dpi=275,
    transparent=True,
)

# %%
# Visualise distributions of LLM wait time / dynamic question completion time (not for feedback)

plt.figure()
time_plot = feedback_given_llm_waittime_df["time_to_show_dynamic_question"].plot(
    kind="hist", bins=12
)
time_plot.set_xlabel("Dynamic Question Generation Time (s)")
time_plot.set_ylabel("Number of Respondents")
plt.savefig(f"{figures_output_folder}LLM_waittime_hist.png", dpi=275, transparent=True)

wait_cutoff = 600
plt.figure()
time_plot = feedback_given_llm_waittime_df[
    feedback_given_llm_waittime_df["time_in_seconds"] < wait_cutoff
]["time_in_seconds"].plot(kind="hist", bins=25)
time_plot.set_xlabel("Time taken complete the dynamic questions (s)")
time_plot.set_ylabel("Number of Respondents")
plt.savefig(
    f"{figures_output_folder}survey_completion_time_hist.png", dpi=275, transparent=True
)

# %%
# Analysis of User Path on Feedback

bonferroni_sign_thresh = SIGNIFICANCE_THRESHOLD / 3

key_variables = ["additional_questions_asked", *feedback_score_cols]
feedback_dynamic_df = feedback_given_df[key_variables].copy()
feedback_dynamic_df = feedback_dynamic_df[feedback_dynamic_df.notnull()]
result_ease: Any = mannwhitneyu(
    feedback_dynamic_df[feedback_dynamic_df["additional_questions_asked"]][
        "ease_score"
    ],
    feedback_dynamic_df[~feedback_dynamic_df["additional_questions_asked"]][
        "ease_score"
    ],
    alternative="two-sided",
)
result_relevance: Any = mannwhitneyu(
    feedback_dynamic_df[feedback_dynamic_df["additional_questions_asked"]][
        "relevance_score"
    ],
    feedback_dynamic_df[~feedback_dynamic_df["additional_questions_asked"]][
        "relevance_score"
    ],
    alternative="two-sided",
)
result_comfort: Any = mannwhitneyu(
    feedback_dynamic_df[feedback_dynamic_df["additional_questions_asked"]][
        "comfort_score"
    ],
    feedback_dynamic_df[~feedback_dynamic_df["additional_questions_asked"]][
        "comfort_score"
    ],
    alternative="two-sided",
)
# Common Language Effect Size:
# CLES = U/n1n2

effect_strength_ease = result_ease.statistic / (
    len(feedback_dynamic_df[feedback_dynamic_df["additional_questions_asked"]])
    * len(feedback_dynamic_df[~feedback_dynamic_df["additional_questions_asked"]])
)
effect_strength_relevance = result_relevance.statistic / (
    len(feedback_dynamic_df[feedback_dynamic_df["additional_questions_asked"]])
    * len(feedback_dynamic_df[~feedback_dynamic_df["additional_questions_asked"]])
)
effect_strength_comfort = result_comfort.statistic / (
    len(feedback_dynamic_df[feedback_dynamic_df["additional_questions_asked"]])
    * len(feedback_dynamic_df[~feedback_dynamic_df["additional_questions_asked"]])
)
eff_sizes = [effect_strength_ease, effect_strength_relevance, effect_strength_comfort]
p_values_dyn = [result_ease.pvalue, result_relevance.pvalue, result_comfort.pvalue]
U_ststistics = [
    result_ease.statistic,
    result_relevance.statistic,
    result_comfort.statistic,
]

fig, ax = plt.subplots(figsize=(7, 3))
corr_mat = ax.imshow(
    np.array(
        [
            [effect_strength_ease, effect_strength_relevance, effect_strength_comfort],
        ]
    ),
    cmap="PRGn",
    vmin=0.25,
    vmax=0.75,
)

for j in range(3):
    if p_values_dyn[j] < bonferroni_sign_thresh:
        ax.text(
            j,
            0,
            r"$U=$"
            + f"{U_ststistics[j]:.3e}\n"
            + r"$p=$"
            + f"{p_values_dyn[j]:.3e}\n"
            + r"CLES$=$"
            + f"{eff_sizes[j]:.3f}",
            ha="center",
            va="center",
            color="k",
            fontsize=10,
        )

ax.set_xticks(
    range(3),
    labels=[k.replace("_", "\n") for k in feedback_score_cols],
    rotation=0,
    fontsize=12,
)
ax.set_yticks(
    range(1), labels=["Received\nDynamic\nQuestions"], rotation=0, fontsize=12
)
cbar = fig.colorbar(corr_mat, ax=ax, shrink=0.62, aspect=7, pad=0.01)
cbar.set_label("Effect Size (CLES)")
plt.tight_layout()
plt.savefig(
    f"{figures_output_folder}corr_mat_feedback_dynamic_MW_CLES.png",
    dpi=275,
    transparent=True,
)

# %%
### Age-related analysis:
bonferroni_sign_thresh = SIGNIFICANCE_THRESHOLD / (3 * 5)

feedback_given_df_age_dummies = pd.get_dummies(
    feedback_given_df[feedback_given_df["feedback_age_range"].notnull()],
    columns=["feedback_age_range"],
    prefix="",
    prefix_sep="",
)

ordered_age_ranges = ["16-24", "25-34", "35-49", "50-64", "65-plus"]
key_variables_age = [*feedback_score_cols, *ordered_age_ranges]

# %%
# Age vs Feedback analysis (v1)
# Considering age-groups as distinct samples of a population (ignoring order)

age_counts = feedback_given_df["feedback_age_range"].value_counts()

observation_count_age = np.array([age_counts[c] for c in ordered_age_ranges])
dof_age_matrix = (
    np.array([observation_count_age, observation_count_age, observation_count_age]) - 2
)

correlations = (
    feedback_given_df_age_dummies[key_variables_age].corr(method="kendall").to_numpy()
)
correlations = correlations[:3, 3:]
uncertainties = (1 - correlations) / np.sqrt(dof_age_matrix + 2)

t_values = correlations / uncertainties
p_values = stats_t.sf(np.abs(t_values), dof_age_matrix)

annot_corrs = np.where(
    (p_values > small_nonzero_number) * (p_values < bonferroni_sign_thresh),
    np.round(correlations, 2).astype(str),
    "",
)
annot_uncertainties = np.where(
    (p_values > small_nonzero_number) * (p_values < bonferroni_sign_thresh),
    np.round(2 * uncertainties, 2).astype(str),
    "",
)
annot_pvalues = np.where(  # type: ignore[call-overload]
    (p_values > small_nonzero_number) * (p_values < bonferroni_sign_thresh),
    p_values,
    "",
)

boundary = np.max(np.abs(correlations))

fig = plt.figure(figsize=(8, 6))
gs = GridSpec(10, 10, figure=fig, wspace=0.01, hspace=0.005)

ax = fig.add_subplot(gs[4:, :])
ax2 = fig.add_subplot(gs[1:4, 1:])

corr_mat = ax.imshow(
    correlations,
    vmax=1,
    vmin=-1,
    cmap="PRGn",
)
ax.set_xticks(
    range(len(key_variables_age[3:])),
    labels=[k.replace("_", " ") for k in key_variables_age[3:]],
    rotation=0,
)
ax.set_yticks(
    range(len(key_variables_age[:3])),
    labels=[k.replace("_", " ") for k in key_variables_age[:3]],
    rotation=0,
)

for i in range(correlations.shape[0]):
    for j in range(correlations.shape[1]):
        annotation = ""
        colour = (
            "w" if abs(correlations[i, j]) > colour_change_limit * boundary else "k"
        )
        if annot_pvalues[i, j] != "":
            ax.text(
                j,
                i,
                r"$p=$" + f"{p_values[i, j]:.3f}",
                ha="center",
                va="center",
                color="k",
                fontsize=10,
            )

cbar = fig.colorbar(corr_mat, ax=ax, shrink=1, aspect=15, pad=0.01)
cbar.set_label("Kendall correlation coefficient " + r"$(\tau)$")
ax.set_xlabel("Age Range")

observations_heatmap = ax2.imshow(
    np.array([age_counts[s] for s in key_variables_age[3:]]).reshape(1, -1),
    cmap="Greys",
)
for age_idx, ac in enumerate([age_counts[s] for s in key_variables_age[3:]]):
    colour = "w" if abs(ac) > 0.5 * np.max(age_counts) else "k"
    ax2.text(
        age_idx,
        0,
        r"$n=$" + f"{ac}",
        ha="center",
        va="center",
        color=colour,
        fontsize=10,
    )
age_cbar = fig.colorbar(observations_heatmap, ax=ax2, shrink=0.666, aspect=6, pad=0.01)

ax2.set_xticks(range(len(age_counts)), labels=[""] * len(age_counts))
ax2.set_yticks([])
age_cbar.set_label("Obs.\n\n")
plt.tight_layout()
plt.savefig(
    f"{figures_output_folder}age_feedback_correlations.png", dpi=275, transparent=True
)

# %%
# Age vs Feedback analysis (v2)
# Considering age in an ordered way (Kruskal-Wallis):

bonferroni_sign_thresh = SIGNIFICANCE_THRESHOLD / 3

result_ease = kruskal(
    *[
        feedback_given_df_age_dummies[feedback_given_df_age_dummies[age]]["ease_score"]
        for age in ordered_age_ranges
    ]
)
result_relevance = kruskal(
    *[
        feedback_given_df_age_dummies[feedback_given_df_age_dummies[age]][
            "relevance_score"
        ]
        for age in ordered_age_ranges
    ]
)
result_comfort = kruskal(
    *[
        feedback_given_df_age_dummies[feedback_given_df_age_dummies[age]][
            "comfort_score"
        ]
        for age in ordered_age_ranges
    ]
)

effect_strength_ease = (result_ease.statistic - 10 + 1) / (
    len(feedback_given_df_age_dummies) - 10
)
effect_strength_relevance = (result_relevance.statistic - 10 + 1) / (
    len(feedback_given_df_age_dummies) - 10
)
effect_strength_comfort = (result_comfort.statistic - 10 + 1) / (
    len(feedback_given_df_age_dummies) - 10
)

effect_strengths = [
    effect_strength_ease,
    effect_strength_relevance,
    effect_strength_comfort,
]
p_values_age = [result_ease.pvalue, result_relevance.pvalue, result_comfort.pvalue]
H_ststistics = [
    result_ease.statistic,
    result_relevance.statistic,
    result_comfort.statistic,
]

fig, ax = plt.subplots(figsize=(7, 3))
corr_mat = ax.imshow(
    np.array(
        [
            [effect_strength_ease, effect_strength_relevance, effect_strength_comfort],
        ]
    ),
    cmap="Purples",
    vmin=0,
    vmax=0.2,
)

for j in range(3):
    if p_values_age[j] < bonferroni_sign_thresh:
        ax.text(
            j,
            0,
            r"$H=$"
            + f"{H_ststistics[j]:.2f}\n"
            + r"$p=$"
            + f"{p_values_age[j]:.1e}\n"
            + r"$\eta^2=$"
            + f"{effect_strengths[j]:.3f}",
            ha="center",
            va="center",
            color="k",
            fontsize=10,
        )

ax.set_xticks(
    range(3),
    labels=[k.replace("_", "\n") for k in feedback_score_cols],
    rotation=0,
    fontsize=12,
)
ax.set_yticks(range(1), labels=["Age\nRange"], rotation=0, fontsize=12)
cbar = fig.colorbar(corr_mat, ax=ax, shrink=0.62, aspect=7, pad=0.01)
cbar.set_label(r"Effect Size $(\eta^2)$")
plt.tight_layout()
plt.savefig(
    f"{figures_output_folder}corr_mat_feedback_age_KW_eta2.png",
    dpi=275,
    transparent=True,
)

# %%
# Comparison of expected/observed age distributions with general working population:
# (source: https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/datasets/employmentunemploymentandeconomicinactivitybyagegroupseasonallyadjusteda05sa)

df_expected_ages = pd.read_csv(f"{folder}/LFS_age_distributions.csv")

employed_16_18 = df_expected_ages["employed_16_18"].mean()
employed_18_24 = df_expected_ages["employed_18_24"].mean()
employed_16_24 = employed_16_18 + employed_18_24
employed_25_34 = df_expected_ages["employed_25_34"].mean()
employed_35_49 = df_expected_ages["employed_35_49"].mean()
employed_50_64 = df_expected_ages["employed_50_64"].mean()
employed_65_plus = df_expected_ages["employed_65_plus"].mean()


employed_total = (
    employed_16_24 + employed_25_34 + employed_35_49 + employed_50_64 + employed_65_plus
)

print(
    f"""
16-24:   {100*employed_16_24/employed_total:4.1f}% - expectation {877*employed_16_24/employed_total:5.1f} responses
25-34:   {100*employed_25_34/employed_total:4.1f}% - expectation {877*employed_25_34/employed_total:5.1f} responses
35-49:   {100*employed_35_49/employed_total:4.1f}% - expectation {877*employed_35_49/employed_total:5.1f} responses
50-64:   {100*employed_50_64/employed_total:4.1f}% - expectation {877*employed_50_64/employed_total:5.1f} responses
65+:     {100*employed_65_plus/employed_total:4.1f}% - expectation {877*employed_65_plus/employed_total:5.1f} responses
"""
)

# %%
# General / Specific Open Question Impact Analysis:
# (this cell - visualising distributions)

# For total summary stats
eval_df.loc[:, "generic_open_q"] = eval_df[
    "survey_assist_open_question"
].str.startswith("What is your employer's main business activity?")

# # For feedback analysis
feedback_given_df.loc[:, "generic_open_q"] = feedback_given_df[
    "survey_assist_open_question"
].str.startswith("What is your employer's main business activity?")

eval_df.loc[:, "additional_questions_asked"] = eval_df.apply(
    mark_questions_asked, axis=1
)
sa_coded_dynamic_df = eval_df[eval_df["additional_questions_asked"]]
feedback_given_df.loc[:, "additional_questions_asked"] = feedback_given_df.apply(
    mark_questions_asked, axis=1
)
feedback_given_dynamic_df = feedback_given_df[
    feedback_given_df["additional_questions_asked"]
]

print(
    f"total generic open questions: {sa_coded_dynamic_df['generic_open_q'].value_counts()}, {sa_coded_dynamic_df['generic_open_q'].value_counts()/len(sa_coded_dynamic_df)}%"
)
print(
    f"feedback-given generic open questions: {feedback_given_dynamic_df['generic_open_q'].value_counts()}, {feedback_given_dynamic_df['generic_open_q'].value_counts()/len(feedback_given_dynamic_df)}"
)

# %%
# General / Specific Open Question Impact Analysis:
# (this cell - statistical tests & heatmap visualisation)

bonferroni_sign_thresh = SIGNIFICANCE_THRESHOLD / 3

key_variables = ["generic_open_q", *feedback_score_cols]
feedback_given_dynamic_df = feedback_given_dynamic_df[key_variables].copy()
feedback_given_dynamic_df = feedback_given_dynamic_df[
    feedback_given_dynamic_df.notnull()
]
result_ease = mannwhitneyu(
    feedback_given_dynamic_df[
        feedback_given_dynamic_df["generic_open_q"] == True  # noqa: E712
    ]["ease_score"],
    feedback_given_dynamic_df[
        feedback_given_dynamic_df["generic_open_q"] == False  # noqa: E712
    ]["ease_score"],
    alternative="two-sided",
)
result_relevance = mannwhitneyu(
    feedback_given_dynamic_df[
        feedback_given_dynamic_df["generic_open_q"] == True  # noqa: E712
    ]["relevance_score"],
    feedback_given_dynamic_df[
        feedback_given_dynamic_df["generic_open_q"] == False  # noqa: E712
    ]["relevance_score"],
    alternative="two-sided",
)
result_comfort = mannwhitneyu(
    feedback_given_dynamic_df[
        feedback_given_dynamic_df["generic_open_q"] == True  # noqa: E712
    ]["comfort_score"],
    feedback_given_dynamic_df[
        feedback_given_dynamic_df["generic_open_q"] == False  # noqa: E712
    ]["comfort_score"],
    alternative="two-sided",
)
# Common Language Effect Size:
# CLES = U/n1n2

effect_strength_ease = result_ease.statistic / (
    len(
        feedback_given_dynamic_df[
            feedback_given_dynamic_df["generic_open_q"] == True  # noqa: E712
        ]
    )
    * len(
        feedback_given_dynamic_df[
            feedback_given_dynamic_df["generic_open_q"] == False  # noqa: E712
        ]
    )
)
effect_strength_relevance = result_relevance.statistic / (
    len(
        feedback_given_dynamic_df[
            feedback_given_dynamic_df["generic_open_q"] == True  # noqa: E712
        ]
    )
    * len(
        feedback_given_dynamic_df[
            feedback_given_dynamic_df["generic_open_q"] == False  # noqa: E712
        ]
    )
)
effect_strength_comfort = result_comfort.statistic / (
    len(
        feedback_given_dynamic_df[
            feedback_given_dynamic_df["generic_open_q"] == True  # noqa: E712
        ]
    )
    * len(
        feedback_given_dynamic_df[
            feedback_given_dynamic_df["generic_open_q"] == False  # noqa: E712
        ]
    )
)
eff_sizes = [effect_strength_ease, effect_strength_relevance, effect_strength_comfort]
p_values_dyn = [result_ease.pvalue, result_relevance.pvalue, result_comfort.pvalue]
U_ststistics = [
    result_ease.statistic,
    result_relevance.statistic,
    result_comfort.statistic,
]

fig, ax = plt.subplots(figsize=(7, 3))
corr_mat = ax.imshow(
    np.array(
        [
            [effect_strength_ease, effect_strength_relevance, effect_strength_comfort],
        ]
    ),
    cmap="PRGn",
    vmin=0.25,
    vmax=0.75,
)

for j in range(3):
    if p_values_dyn[j] < bonferroni_sign_thresh:
        ax.text(
            j,
            0,
            r"$U=$"
            + f"{U_ststistics[j]:.3e}\n"
            + r"$p=$"
            + f"{p_values_dyn[j]:.3e}\n"
            + r"CLES$=$"
            + f"{eff_sizes[j]:.3f}",
            ha="center",
            va="center",
            color="k",
            fontsize=10,
        )

ax.set_xticks(
    range(3),
    labels=[k.replace("_", "\n") for k in feedback_score_cols],
    rotation=0,
    fontsize=12,
)
ax.set_yticks(range(1), labels=["Received\nGeneric\nQuestion"], rotation=0, fontsize=12)
cbar = fig.colorbar(corr_mat, ax=ax, shrink=0.62, aspect=7, pad=0.01)
cbar.set_label("Effect Size (CLES)")
plt.tight_layout()
plt.savefig(
    f"{figures_output_folder}corr_mat_feedback_generic_or_specific_Q_MW_CLES.png",
    dpi=275,
    transparent=True,
)
