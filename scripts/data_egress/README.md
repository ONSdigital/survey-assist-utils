# Firestore Data Extraction Utilities

## Utilities

#### - `retrieve_survey_responses.py`
#### - `reformat_and_make_csvs.py`

`retrieve_survey_responses.py` retrieves survey response documents from a Google Firestore database collection, flattens them, and saves the flattened data .parquet files.
It is designed to handle large datasets efficiently by processing documents in chunks, creating intermediate files.
It can be used to extract both the survey responses and the survey feedback.

`reformat_and_make_csvs.py` reformats and collates the files output by `retrieve_survey_responses.py`, then saves subsets of the data in five files; a 'minimal' CSV for clerical coders (contains only ID and initial question responses), an 'extra' CSV for clerical coders (extends 'minimal' to include open dynamic question and response), and an 'evaluation' CSV for internal use (extends 'extra' to include survey-assist decisions, assigned codes, candidates, closed questions, and optionally feedback).
The remaining two output files have the full set of data fields for responses detected as invalid / duplicate, or not in employment.
It can optionally include feedback data in the evaluation CSV, and can filter the data to a) only output responses entered after a chosen timestamp.
It will detect and flag malformed response data, duplicate responses, and responses where people have selected 'not in employment'.
On completion, it will print a concise summary / breakdown of the valid / invalid / duplicate / not in employment responses.


## Notes

The contents of documents in the Firestore database collection differ depending on the 'path' the respondent took as they completed the survey.
As a result, the processing / flattening / reformatting is performed heuristcally rather than deterministically.
As a result, further validation checks are performed in the `reformat_and_make_csvs.py` script.

## Prerequisites

1.  **Python Virtual Environment**: Activate the `survey-assist-utils` virtual environment.

2.  **Google Cloud Authentication**: You must be authenticated to the Google Cloud project containing the Firestore database.
    ```sh
    gcloud auth application-default login
    ```

## Usage: `retrieve_survey_responses.py`

```text
usage: Utility to retrieve survey responses from a Firestore database.
python retrieve_survey_responses.py [-h] [--timeout TIMEOUT] [--chunk_size CHUNK_SIZE]
                                    project_id database_id collection_name output_name

positional arguments:
  project_id            The Google Cloud project ID.
  database_id           The Firestore database ID.
  collection_name       The collection_name.
  output_name           The base of the name of the output folder.

options:
  -h, --help            show this help message and exit
  --timeout TIMEOUT, -t TIMEOUT
                        The connection timeout in seconds.
  --chunk_size CHUNK_SIZE, -c CHUNK_SIZE
                        The number of documents to process in each chunk.
```

## Usage: `reformat_and_make_csvs.py`

```text
reformat_and_make_csvs.py [-h] [--intermediate_feedback_path INTERMEDIATE_FEEDBACK_PATH] [--only_after ONLY_AFTER]
                          intermediate_responses_path output_name_base

positional arguments:
  intermediate_responses_path
                        path to the files output from the response data egress process.
  output_name_base      The base of the name of the output CSV files.

options:
  -h, --help            show this help message and exit
  --intermediate_feedback_path INTERMEDIATE_FEEDBACK_PATH
                        path to the files output from the feedback data egress process.
  --only_after ONLY_AFTER
                        Restrict results to those collected after specified timestamp. Format Y_m_d__H_M_S (e.g. '2024_01_01__00_00_000000').
```
