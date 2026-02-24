# pylint: disable=line-too-long,duplicate-code,invalid-name
# %%
"""This file is a notebook (convert with `jupytext`) for investigation of feedback
from the SurveyAssist testing.
"""
from os import makedirs

# pylint: disable=line-too-long,duplicate-code
from textwrap import wrap

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# %matplotlib inline

# %%
project_id = dotenv.get_key(".env", "PROJECT_ID")
if not project_id:
    raise ValueError("PROJECT_ID not found in .env file. Please set it.")

data_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET") or ""
work_dir = data_bucket + "analysis-interim-results"
# out_dir = work_dir + "/CC_SocSurveys_feedback"
out_dir = None

figures_output_folder = "data/figures/cc_feedback_analysis/"
makedirs(figures_output_folder, exist_ok=True)

# %%
# load combined df with codability levels
cleaned_evaluation_df = pd.read_parquet(
    work_dir + "/clerically-coded/clerical_df_with_cc_clean_codes.parquet"
)

export_df = cleaned_evaluation_df.copy()
export_df.columns = export_df.columns.str.replace(r"\n", " ", regex=True)
export_cols = [
    "unique_id",
    "user",
    "job_title",
    "job_description",
    "org_description",
    "survey_assist_open_question",
    "survey_assist_open_question_response",
    "comments_initial",
    "qa_initial",
    "qa_comments_final",
    "rag_status",
    "rag_status_ (red,_amber,_green,_unnecessary)",
    "rationale_for_rag_status",
    "is_the_follow-up_question_(col_f)_useful? y/n",
    "how_useful_is_the_question_that_was_asked?",
    "is_a_follow-up_question_needed_to_code_the_standard_tlfs_responses?_ y/n",
    "is_a_follow-up_question_needed_to_code_the_survey_assist_response?_ y/n",
    "do_you_think_it_is_possible_to_get_a_single_5-digit_code_with_a_single,_open_question,_based_on_the_initial_tlfs_responses?",
    "is_there_an_alternative_question_you_would_ask?__ y/n",
    "if_yes:_what_question_should_be_asked?",
]
if out_dir:
    export_df[export_cols].to_csv(
        f"{out_dir}/CC_coded_public_test_responses_with_comments.csv",
        index=False,
    )
# %%
# RAG Status Distributions


def align_rag_status(row):
    """Merge and clean the rag status columns."""
    if row["rag_status_\n(red,_amber,_green,_unnecessary)"] is not None:
        return row["rag_status_\n(red,_amber,_green,_unnecessary)"].lower().strip()
    if row["rag_status"] is not None:
        return row["rag_status"].lower().strip()
    return None


cleaned_evaluation_df["aligned_rag_status"] = cleaned_evaluation_df.apply(
    align_rag_status, axis=1
)

cleaned_evaluation_with_cc_openQs = cleaned_evaluation_df[
    cleaned_evaluation_df["survey_assist_open_question"].notna()
].copy()

cleaned_evaluation_with_cc_openQs["aligned_rag_status"].value_counts(dropna=False)

# %%
# Clerical Coder Comments Summary Statistics

comments_cols = [
    "comments_initial",
    "qa_comments_initial",
    "qa_comments_final",
    "how_useful_is_the_question_that_was_asked?",
    "rationale_for_rag_status",
]

print("total responses: ", len(cleaned_evaluation_df))
for c in comments_cols:
    count = cleaned_evaluation_df[c].notna().sum()
    print(c, count, f"({100*count/len(cleaned_evaluation_df):.0f}% of total)")


print("\nreceived dynamic questions: ", len(cleaned_evaluation_with_cc_openQs))
for c in comments_cols:
    count = cleaned_evaluation_with_cc_openQs[c].notna().sum()
    print(
        c, count, f"({100*count/len(cleaned_evaluation_with_cc_openQs):.0f}% of total)"
    )

# %%

# Yes / No Question summary statistics


def clean_yn_col(ans):
    """Clean yes/no columns."""
    if isinstance(ans, str) and len(ans.strip()) > 0:
        ans = ans.lower().strip()
        if ans in ("y", "n"):
            return ans
    return ""


general_yes_no_cols = [
    "is_a_follow-up_question_needed_to_code_the_standard_tlfs_responses?_\ny/n",
    "do_you_think_it_is_possible_to_get_a_single_5-digit_code"
    + "_with_a_single,_open_question,_based_on_the_initial_tlfs_responses?",
]

requires_dyn_yes_no_cols = [
    "is_a_follow-up_question_needed_to_code_the_survey_assist_response?_\ny/n",
    "is_the_follow-up_question_(col_f)_useful?\ny/n",
    "is_there_an_alternative_question_you_would_ask?__\ny/n",
]

general_yn_responses = {}
requires_dyn_yn_responses = {}

for c in general_yes_no_cols:
    cleaned_evaluation_with_cc_openQs[c] = cleaned_evaluation_with_cc_openQs[c].apply(
        clean_yn_col
    )
    yays = cleaned_evaluation_with_cc_openQs[c] == "y"
    nays = cleaned_evaluation_with_cc_openQs[c] == "n"
    missings = cleaned_evaluation_with_cc_openQs[c] == ""
    general_yn_responses[c] = (yays.sum(), nays.sum(), missings.sum())
    print(
        f"""\n{c}:
    missing: {missings.sum()} ({missings.sum()/len(cleaned_evaluation_with_cc_openQs)*100:.0f}% of total)
    Y: {yays.sum()} ({yays.sum()/len(cleaned_evaluation_with_cc_openQs)*100:.0f}% of total)
    N: {nays.sum()} ({nays.sum()/len(cleaned_evaluation_with_cc_openQs)*100:.0f}% of total)"""
    )

for c in requires_dyn_yes_no_cols:
    cleaned_evaluation_with_cc_openQs[c] = cleaned_evaluation_with_cc_openQs[c].apply(
        clean_yn_col
    )
    yays = cleaned_evaluation_with_cc_openQs[c] == "y"
    nays = cleaned_evaluation_with_cc_openQs[c] == "n"
    missings = cleaned_evaluation_with_cc_openQs[c] == ""
    print(cleaned_evaluation_with_cc_openQs[c].value_counts())
    requires_dyn_yn_responses[c] = (yays.sum(), nays.sum(), missings.sum())
    print(
        f"""\n{c}:
    missing: {missings.sum()} ({missings.sum()/len(cleaned_evaluation_with_cc_openQs)*100:.0f}% of responses who received dynamic questions)
    Y: {yays.sum()} ({yays.sum()/len(cleaned_evaluation_with_cc_openQs)*100:.0f}% of responses who received dynamic questions)
    N: {nays.sum()} ({nays.sum()/len(cleaned_evaluation_with_cc_openQs)*100:.0f}% of responses who received dynamic questions)"""
    )


def make_label_tidy(colname: str):
    """Makes a column name readable."""
    return "\n".join(wrap(colname.replace("_", " "), width=15))


# %%
# Combined Yes/No/Missing stacked bar plot

all_yn_responses = {**general_yn_responses, **requires_dyn_yn_responses}

questions = list(all_yn_responses.keys())
yes_counts = [all_yn_responses[q][0] for q in questions]
no_counts = [all_yn_responses[q][1] for q in questions]
missing_counts = [all_yn_responses[q][2] for q in questions]

question_labels = [make_label_tidy(q) for q in questions]

x = np.arange(len(question_labels))

fig, ax = plt.subplots(figsize=(12, 7))

# Stacked bars
bar_yes = ax.bar(x, yes_counts, label="Yes", color="#28A197")
bar_no = ax.bar(x, no_counts, bottom=yes_counts, label="No", color="#F46A25")
bar_missing = ax.bar(
    x,
    missing_counts,
    bottom=[i + j for i, j in zip(yes_counts, no_counts)],
    label="Missing",
    color="#A285D1",
)

ax.set_ylabel("Number of Responses", fontsize=18)
ax.set_title(
    f"Clerical Coder Yes/No Responses (n={len(cleaned_evaluation_with_cc_openQs)})",
    fontsize=22,
)
ax.set_xticks(x)
ax.set_xticklabels(question_labels, rotation=0, fontsize=14)
ax.tick_params(axis="y", labelsize=14)
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), fontsize=14)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Add text labels on bars
for i in range(len(x)):
    # Yes
    if yes_counts[i] > 0:
        ax.text(
            i,
            yes_counts[i] / 2,
            f"{yes_counts[i]} ({yes_counts[i]*100/len(cleaned_evaluation_with_cc_openQs):.0f}%)",
            ha="center",
            va="center",
            color="white",
            fontsize=16,
        )
    # No
    if no_counts[i] > 0:
        ax.text(
            i,
            yes_counts[i] + no_counts[i] / 2,
            f"{no_counts[i]} ({no_counts[i]*100/len(cleaned_evaluation_with_cc_openQs):.0f}%)",
            ha="center",
            va="center",
            color="white",
            fontsize=16,
        )
    # Missing
    if missing_counts[i] > 0:
        ax.text(
            i,
            yes_counts[i] + no_counts[i] + missing_counts[i] / 2,
            f"{missing_counts[i]} ({missing_counts[i]*100/len(cleaned_evaluation_with_cc_openQs):.0f}%)",
            ha="center",
            va="center",
            color="white",
            fontsize=16,
        )


# total_counts = [y+n+m for y,n,m in zip(yes_counts, no_counts, missing_counts)]
# for i, total in enumerate(total_counts):
#     ax.text(i, total, str(total), ha='center', va='bottom', fontsize=14)


plt.tight_layout()
plt.savefig(
    f"{figures_output_folder}cc_yn_distributions_stacked.png", dpi=275, transparent=True
)
plt.show()


# %%


# %%

rag_count_dict = (
    cleaned_evaluation_with_cc_openQs["aligned_rag_status"].value_counts().to_dict()
)


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 5), constrained_layout=True)
bar1 = ax.bar(
    [i - 0.125 for i in range(1, 5)],
    [rag_count_dict[c] for c in ["green", "amber", "red", "unnecessary"]],
    color=["#0f8243", "#fbc900", "#d0021b", "#12436D"],  # "#12436D",
    width=0.75,
)

for bar in bar1:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height()} ({bar.get_height()*100/len(cleaned_evaluation_with_cc_openQs):.0f}%)",
        ha="center",
        va="bottom",
        fontsize=18,
    )

ax.set_xticks(
    [i - 0.125 for i in range(1, 5)],
    labels=["green", "amber", "red", "unnecessary"],
    rotation=0,
    fontsize=18,
)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

ax.set_ylabel("Number of Questions", fontsize=18)
ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350])
ax.set_yticklabels([0, 50, 100, 150, 200, 250, 300, 350], size=14)  # type: ignore[list-item]

plt.tight_layout()
plt.savefig(
    f"{figures_output_folder}cc_rag_distributions.png", dpi=275, transparent=True
)


# %%
comments_initial_df = cleaned_evaluation_df[
    cleaned_evaluation_df["how_useful_is_the_question_that_was_asked?"].notnull()
].copy()
# msk = comments_initial_df["comments_initial"].notna() & (
#     comments_initial_df["comments_initial"].isin(["", " " "", " ", "-", "n0", "None"])
# )

# comments_initial_df = comments_initial_df[~msk].reset_index(drop=True)

# %%
len(
    comments_initial_df[
        ~comments_initial_df["how_useful_is_the_question_that_was_asked?"].isna()
    ]
)


# %%
def mark_questions_asked(row):
    """Helper function to mark cases responses where dynamic questions were asked."""
    return row["survey_assist_open_question"] is not None


# For total summary stats
cleaned_evaluation_with_cc_openQs["generic_open_q"] = cleaned_evaluation_with_cc_openQs[
    "survey_assist_open_question"
].str.startswith("What is your employer's main business activity?")

cleaned_evaluation_with_cc_openQs["additional_questions_asked"] = (
    cleaned_evaluation_with_cc_openQs.apply(mark_questions_asked, axis=1)
)
dynamic_qs_df = cleaned_evaluation_with_cc_openQs[
    cleaned_evaluation_with_cc_openQs["additional_questions_asked"]
]
cc_coded_dynamic_df = dynamic_qs_df[
    dynamic_qs_df["cc_final_codability_level_open_q"] == "Sub-class (5-digits)"
]
cc_uncoded_dynamic_df = dynamic_qs_df[
    dynamic_qs_df["cc_final_codability_level_open_q"] != "Sub-class (5-digits)"
]

print(
    f"total generic open questions: {dynamic_qs_df['generic_open_q'].value_counts()}, {dynamic_qs_df['generic_open_q'].value_counts()/len(dynamic_qs_df)}%"
)
print(
    f"successfully coded generic open questions: {cc_coded_dynamic_df['generic_open_q'].value_counts()}, {cc_coded_dynamic_df['generic_open_q'].value_counts()/len(cc_coded_dynamic_df)}"
)
print(
    f"unsuccessfully coded generic open questions: {cc_uncoded_dynamic_df['generic_open_q'].value_counts()}, {cc_uncoded_dynamic_df['generic_open_q'].value_counts()/len(cc_uncoded_dynamic_df)}"
)

total_counts = dynamic_qs_df["generic_open_q"].value_counts()
coded_counts = cc_coded_dynamic_df["generic_open_q"].value_counts()
uncoded_counts = cc_uncoded_dynamic_df["generic_open_q"].value_counts()

total_n = len(dynamic_qs_df)
specific_n = coded_counts.get(False, 0) + uncoded_counts.get(False, 0)
generic_n = coded_counts.get(True, 0) + uncoded_counts.get(True, 0)
labels = [
    f"Specific Question\n({specific_n:.0f} - {specific_n/total_n:.1%} of total)",
    f"Generic Question\n({generic_n:.0f} - {generic_n/total_n:.1%} of total)",
]
x = np.arange(len(labels))
width = 0.4

fig, ax = plt.subplots(figsize=(10, 5))
# rects1 = ax.bar(
#     x - width,
#     [total_counts.get(False, 0), total_counts.get(True, 0)],
#     width,
#     label=f"All Open Questions (n={len(dynamic_qs_df)})",
#     color="#12436D",
# )
rects2 = ax.bar(
    x - width / 2,
    [coded_counts.get(False, 0), coded_counts.get(True, 0)],
    width,
    label=f"Unambiguously Coded to 5-digits (n={len(cc_coded_dynamic_df)})",
    color="#28A197",
)
rects3 = ax.bar(
    x + width / 2,
    [uncoded_counts.get(False, 0), uncoded_counts.get(True, 0)],
    width,
    label=f"Not Successfully Coded (n={len(cc_uncoded_dynamic_df)})",
    color="#801650",
)

ax.set_ylabel("Number of Respondents", fontsize=18)
ax.set_title("Distributions of Open Question Type", fontsize=18)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16)

ax.set_ylim(0, 410)
ax.legend(fontsize=12)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

for bar in [rects2, rects3]:
    for rect_id, rect in enumerate(bar):
        height = rect.get_height()
        if rect_id == 0:
            total = specific_n
            label = "specific"
        else:
            total = generic_n
            label = "generic"
        print(total, height)
        ax.annotate(
            f"{height} ({100*height/total:.0f}%\nof {label})",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            fontsize=13,
            ha="center",
            va="bottom",
        )

fig.tight_layout()
plt.savefig(
    f"{figures_output_folder}generic_vs_specific_open_q_dist_success.png",
    dpi=275,
    transparent=True,
)

# %%

mw_generic_codable = mannwhitneyu(
    cc_coded_dynamic_df["generic_open_q"],
    cc_uncoded_dynamic_df["generic_open_q"],
    alternative="two-sided",
)

effect_strength_generic_codable = mw_generic_codable.statistic / (
    len(cc_coded_dynamic_df) * len(cc_uncoded_dynamic_df)
)

print(
    mw_generic_codable.statistic,
    mw_generic_codable.pvalue,
    effect_strength_generic_codable,
)

# %%
df_soc_surv = pd.read_csv(
    f"{work_dir}/CC_SocSurveys_feedback/Survey_Assist_Qu_Eval_V2_Beth.csv"
)

# %%
merged_df_export = df_soc_surv.merge(
    dynamic_qs_df[
        ["unique_id", "user", "aligned_rag_status", "rationale_for_rag_status"]
    ],
    on=["unique_id", "user"],
    how="inner",
)

merged_df_export.rename(
    columns={
        "aligned_rag_status": "CC RAG Status",
        "rationale_for_rag_status": "CC RAG Rationale",
        "Overall RAG Status": "Quality Overall RAG Status",
    },
    inplace=True,
)
merged_df_export["CC RAG Status"] = merged_df_export["CC RAG Status"].apply(
    lambda x: x[0].upper()
)


# %%
if out_dir:
    merged_df_export.to_csv(
        f"{out_dir}/Quality_and_CC_RAG_Statuses_SurveyAssist_OpenQs.csv",
        index=False,
    )

# %%
merged_df = dynamic_qs_df.merge(
    df_soc_surv[["unique_id", "user", "Overall RAG Status", "Comments/ Feedback"]],
    on=["unique_id", "user"],
    how="inner",
)
merged_df.rename(
    columns={
        "aligned_rag_status": "CC RAG Status",
        "rationale_for_rag_status": "CC RAG Rationale",
        "Comments/ Feedback": "Quality Comments / Feedback",
        "Overall RAG Status": "Quality Overall RAG Status",
    },
    inplace=True,
)

# %%
merged_df["CC RAG Status"] = merged_df["CC RAG Status"].apply(lambda x: x[0].upper())
merged_df["Quality_CC_agree"] = merged_df.apply(
    lambda row: row["Quality Overall RAG Status"] == row["CC RAG Status"], axis=1
)

# %%
# Check what % of questions CC and MQD agree on RAG status
### merged_df["Quality_CC_agree"].value_counts() * 100 / len(merged_df)

# %%
# Check what % of questions CC and MQD agree on RAG status *for each colour*

print(
    """
    The dynamic (open) questions were assessed using a Red/Amber/Green/[Unnecessary]
    (RAG) system by a) the Clerical Coding team (CC) and b) the Social Surveys team.
    The CC team assessment was based on question 'usefulness' or 'insight', while
    the Soc. Sur. assessment was based on question 'quality'.

    In this section, we compare the two RAG assessments in an attempt to identify
    areas of overlap, which would indicate especially good or bad patterns among
    questions.

    Splitting based on CC RAG Status:
"""
)

# Green
print(
    f"Green (CC n={len(merged_df[merged_df['CC RAG Status']=='G'])}, Quality n={len(merged_df[merged_df['Quality Overall RAG Status']=='G'])})\n",
    merged_df[merged_df["CC RAG Status"] == "G"][
        "Quality Overall RAG Status"
    ].value_counts()
    * 100
    / len(merged_df[merged_df["CC RAG Status"] == "G"]),
    "\n",
)

# Amber
print(
    f"Amber (CC n={len(merged_df[merged_df['CC RAG Status']=='A'])}, Quality n={len(merged_df[merged_df['Quality Overall RAG Status']=='A'])})\n",
    merged_df[merged_df["CC RAG Status"] == "A"][
        "Quality Overall RAG Status"
    ].value_counts()
    * 100
    / len(merged_df[merged_df["CC RAG Status"] == "A"]),
    "\n",
)

# Red
print(
    f"Red (CC n={len(merged_df[merged_df['CC RAG Status']=='R'])}, Quality n={len(merged_df[merged_df['Quality Overall RAG Status']=='R'])})\n",
    merged_df[merged_df["CC RAG Status"] == "R"][
        "Quality Overall RAG Status"
    ].value_counts()
    * 100
    / len(merged_df[merged_df["CC RAG Status"] == "R"]),
    "\n",
)
# %%
merged_df_RAG_dummies = pd.get_dummies(
    merged_df, columns=["CC RAG Status"], prefix="Usefulness", prefix_sep="_"
)
merged_df_RAG_dummies = pd.get_dummies(
    merged_df_RAG_dummies,
    columns=["Quality Overall RAG Status"],
    prefix="Quality",
    prefix_sep="_",
)

# %%
RAG_cols = [
    "Usefulness_R",
    "Usefulness_A",
    "Usefulness_G",
    "Usefulness_U",
    "Quality_R",
    "Quality_A",
    "Quality_G",
    "Quality_U",
]
correlations = merged_df_RAG_dummies[RAG_cols].corr(method="kendall")
correlations = correlations.to_numpy()[4:, :4]

fisher_transformed_z_scores = np.log(np.sqrt((1 + correlations) / (1 - correlations)))
std_errs = 1 / np.sqrt(len(merged_df_RAG_dummies) - 3)
confidence_interval_zspace = 1.96 * std_errs  # Using 95% confidence interval
correlation_CI_lower = (np.exp(2 * (correlations - confidence_interval_zspace)) - 1) / (
    np.exp(2 * (correlations - confidence_interval_zspace)) + 1
)
correlation_CI_upper = (np.exp(2 * (correlations + confidence_interval_zspace)) - 1) / (
    np.exp(2 * (correlations + confidence_interval_zspace)) + 1
)

fig, ax = plt.subplots(figsize=(6, 6))
corr_mat = ax.imshow(correlations, vmax=1, vmin=-1, cmap="PRGn")
colour_change_limit = 0.75

for i in range(correlations.shape[0]):
    for j in range(correlations.shape[1]):
        annotation = ""
        colour = "w" if abs(correlations[i, j]) > colour_change_limit else "k"
        ax.text(
            j,
            i,
            r"$\tau=$"
            + f"{correlations[i, j]:.2f}"
            + "\n"
            + r"CI $=$ "
            + f"({correlation_CI_lower[i, j]:.2f}, "
            + f"{correlation_CI_upper[i, j]:.2f})",
            ha="center",
            va="center",
            color=colour,
            fontsize=7,
        )

ax.set_xticks(
    range(len(RAG_cols[4:])),
    labels=[k.replace("_", "\n") for k in RAG_cols[4:]],
    rotation=20,
)
ax.set_yticks(
    range(len(RAG_cols[:4])),
    labels=[k.replace("_", "\n") for k in RAG_cols[:4]],
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
    f"{figures_output_folder}CC_Quality_RAG_correlation_matrix.png",
    dpi=275,
    transparent=True,
)
# %%
