# %%
"""This file is a notebook (convert with `jupytext`) for investigation of feedback
from the SurveyAssist testing.
"""

import dotenv
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
    work_dir + "/evaluation_df_with_sa_clean_codes.parquet"
)

# %%
feedback_df = cleaned_evaluation_df[
    [
        "unique_id",
        "feedback_comments",
        "sa_codability_gain_open_q",
        "most_likely_sic_section",
    ]
].copy()
msk = feedback_df["feedback_comments"].notna() & (
    feedback_df["feedback_comments"].isin(["", " " "", " ", "-", "n0", "None"])
)

feedback_df = feedback_df[~msk].reset_index(drop=True)


# %%
feedback_ta = TextAnalyser(
    feedback_df,
    "feedback_comments",
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
            "nah",
            "non",
            "no feedback",
            "no comments",
            "no further comments",
            "no additional feedback",
            "not at all",
            "no i don't",
            "nothing",
            "nothing further",
            "non applicable",
            "none thank you",
            "no thanks",
            "nothing to add",
            "not really",
            "not at this moment",
            "no, i am good",
            "no, all good",
        ],
        "null_marker_threshold": 0.6,
    },
)

# %%
feedback_ta.investigate_clusters(kmin=1, kmax=30)

feedback_ta.apply_kmeans(k=7)
feedback_ta.visualise_dim_reduced()

# %%
feedback_ta.drop_null_responses()
feedback_ta.investigate_clusters(kmin=1, kmax=30)

feedback_ta.apply_kmeans(k=7)
feedback_ta.visualise_dim_reduced()

# %%
question_df = cleaned_evaluation_df[
    [
        "unique_id",
        "sa_codability_gain_open_q",
        "survey_assist_open_question",
        "most_likely_sic_section",
    ]
].copy()
question_df = question_df[question_df.survey_assist_open_question.notna()].reset_index(
    drop=True
)

question_ta = TextAnalyser(
    question_df,
    "survey_assist_open_question",
    project_id,
    additional_kwargs={
        "model_name": "text-embedding-004",
        "model_task_type": "SEMANTIC_SIMILARITY",
        "max_batch_size": 250,
        "cleaning_func": lambda x: x.lower().strip(),
        "example_null_responses": [
            "what is your employer's main business activity?",
            "what is your organisation's main activity?",
            "what is the main activity of your organisation?",
            "what is the main business activity of your employer?",
        ],
        "null_marker_threshold": 0.6,
    },
)
# %%
question_ta.investigate_clusters(kmin=1, kmax=30)

question_ta.apply_kmeans(k=8)
question_ta.visualise_dim_reduced()

# %%
question_ta.drop_null_responses()
question_ta.investigate_clusters(kmin=1, kmax=30)

question_ta.apply_kmeans(k=6)
question_ta.visualise_dim_reduced()

# %%
temp_df = question_df.merge(
    question_ta.df,
    on=["unique_id", "sa_codability_gain_open_q", "most_likely_sic_section"],
    how="left",
)
temp_df["cluster"] = temp_df["feedback_comment_labels"].fillna(-1)
temp_df["gain_positive"] = temp_df["sa_codability_gain_open_q"] > 0
# cross tab of gain positive vs clusters
cross_df = (
    temp_df.groupby("cluster", dropna=False)
    .agg(
        {
            "unique_id": "count",
            "gain_positive": "mean",
            "sa_codability_gain_open_q": "mean",
        }
    )
    .reset_index()
    .rename(
        columns={
            "unique_id": "num_responses",
            "gain_positive": "pct_gain_positive",
            "sa_codability_gain_open_q": "avg_gain",
        }
    )
)
cross_df["representative_text"] = [
    "what is the main activity of your organisation?",
    *question_ta.cluster_representatives,
]
# cross tab of sic sections vs clusters
tmp = temp_df.groupby("cluster")["most_likely_sic_section"].value_counts()
cross_df = cross_df.merge(
    tmp[tmp > 2**4]
    .groupby("cluster")
    .apply(lambda x: list(x.index.get_level_values(1)))
    .reset_index(name="common_sic_sections")
)
print(cross_df)

# %%
