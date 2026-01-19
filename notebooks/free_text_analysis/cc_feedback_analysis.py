# %%
"""This file is a notebook (convert with `jupytext`) for investigation of feedback
from the SurveyAssist testing.
"""
# pylint: disable=line-too-long,duplicate-code
from textwrap import wrap

import dotenv
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from survey_assist_utils.evaluation.text_analysis import TextAnalyser

# %matplotlib inline

# %%
project_id = dotenv.get_key(".env", "PROJECT_ID")
if not project_id:
    raise ValueError("PROJECT_ID not found in .env file. Please set it.")

data_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET") or ""
work_dir = data_bucket + "analysis-interim-results"

# %%
# load combined df with codability levels
cleaned_evaluation_df = pd.read_parquet(
    work_dir + "/clerically-coded/clerical_df_with_cc_clean_codes.parquet"
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

### cleaned_evaluation_with_cc_openQs.columns

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
    print(c, count, f"{100*count/len(cleaned_evaluation_df):.0f}%")


print("\nreceived dynamic questions: ", len(cleaned_evaluation_with_cc_openQs))
for c in comments_cols:
    count = cleaned_evaluation_with_cc_openQs[c].notna().sum()
    print(c, count, f"{100*count/len(cleaned_evaluation_with_cc_openQs):.0f}%")

# %%

# Yes / No Question summary statistics


def clean_yn_col(ans):
    """Clean yes/no columns."""
    if isinstance(ans, str) and len(ans.strip()) > 0:
        return ans.lower().strip()
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
    cleaned_evaluation_df[c] = cleaned_evaluation_df[c].apply(clean_yn_col)
    yays = cleaned_evaluation_df[c] == "y"
    nays = cleaned_evaluation_df[c] == "n"
    missings = cleaned_evaluation_df[c] == ""
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

fig, (ax1, ax2) = plt.subplots(
    ncols=1, nrows=2, figsize=(10, 10), constrained_layout=True, sharex=True
)


def make_label_tidy(colname: str):
    """Makes a column name readable."""
    return "\n".join(wrap(colname.replace("_", " "), width=14))


# General subplot
bar1 = ax1.bar(
    [i - 0.125 for i in range(1, 4)],
    general_yn_responses[general_yes_no_cols[0]],
    color="#12436D",  # "#12436D",
    width=0.25,
)
bar2 = ax1.bar(
    [i + 0.125 for i in range(1, 4)],
    general_yn_responses[general_yes_no_cols[1]],
    color="#801650",
    width=0.25,
)
for bar_choice in [bar1, bar2]:
    for bar in bar_choice:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height()}",
            ha="center",
            va="bottom",
            fontsize=14,
        )

ax1.legend(
    handles=[
        mpatches.Patch(color="#12436D", label=make_label_tidy(general_yes_no_cols[0])),
        mpatches.Patch(color="#801650", label=make_label_tidy(general_yes_no_cols[1])),
    ],
    ncols=2,
)

ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)

ax1.set_ylabel("Number of Questions", fontsize=18)
ax1.set_yticks([0, 100, 200, 300, 400, 500])
ax1.set_yticklabels([0, 100, 200, 300, 400, 500], size=14)  # type: ignore[list-item]

# Requires Dynamic Questions subplot
bar3 = ax2.bar(
    [i - 0.25 for i in range(1, 4)],  # pylint: disable=unnecessary-comprehension
    requires_dyn_yn_responses[requires_dyn_yes_no_cols[0]],
    color="#28A197",  # "#12436D",
    width=0.25,
)
bar4 = ax2.bar(
    [i for i in range(1, 4)],  # noqa:C416  # pylint: disable=unnecessary-comprehension
    requires_dyn_yn_responses[requires_dyn_yes_no_cols[1]],
    color="#F46A25",
    width=0.25,
)
bar5 = ax2.bar(
    [i + 0.25 for i in range(1, 4)],
    requires_dyn_yn_responses[requires_dyn_yes_no_cols[2]],
    color="#A285D1",
    width=0.25,
)
for bar_choice in [bar3, bar4, bar5]:
    for bar in bar_choice:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height()}",
            ha="center",
            va="bottom",
            fontsize=14,
        )

ax2.legend(
    handles=[
        mpatches.Patch(
            color="#28A197", label=make_label_tidy(requires_dyn_yes_no_cols[0])
        ),
        mpatches.Patch(
            color="#F46A25", label=make_label_tidy(requires_dyn_yes_no_cols[1])
        ),
        mpatches.Patch(
            color="#A285D1", label=make_label_tidy(requires_dyn_yes_no_cols[2])
        ),
    ],
    ncols=3,
)

ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)

ax2.set_ylabel("Number of Questions", fontsize=18)
ax2.set_yticks([0, 100, 200, 300, 400, 500])
ax2.set_yticklabels([0, 100, 200, 300, 400, 500], size=14)  # type: ignore[list-item]

ax2.set_xticks(
    range(1, 4),
    labels=["Yes", "No", "Missing"],
    rotation=0,
    fontsize=18,
)

plt.tight_layout()
plt.savefig("cc_yn_distributions.png", dpi=275)

# %%

rag_count_dict = (
    cleaned_evaluation_with_cc_openQs["aligned_rag_status"].value_counts().to_dict()
)


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 8), constrained_layout=True)
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
        f"{bar.get_height()}",
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
plt.savefig("cc_rag_distributions.png", dpi=275)

# %%
comments_initial_df = cleaned_evaluation_df[
    cleaned_evaluation_df["comments_initial"].notnull()
].copy()
# msk = comments_initial_df["comments_initial"].notna() & (
#     comments_initial_df["comments_initial"].isin(["", " " "", " ", "-", "n0", "None"])
# )

# comments_initial_df = comments_initial_df[~msk].reset_index(drop=True)

# %%
len(comments_initial_df[~comments_initial_df["comments_initial"].isna()])


# %%
initial_comments_ta = TextAnalyser(
    comments_initial_df,
    "comments_initial",
    project_id,
    additional_kwargs={
        "model_name": "text-embedding-004",
        "model_task_type": "SEMANTIC_SIMILARITY",
        "max_batch_size": 250,
        "cleaning_func": lambda x: x.lower().strip(),
        "example_null_responses": [
            "none",
            "no",
            "na",
            "nope",
            "n/a",
            "nil",
        ],
        "null_marker_threshold": 0.6,
    },
)

# %%
initial_comments_ta.investigate_clusters(kmin=1, kmax=30)

initial_comments_ta.apply_kmeans(k=4)
initial_comments_ta.visualise_dim_reduced()
