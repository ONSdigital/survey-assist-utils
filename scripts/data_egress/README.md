# Firestore Survey Response Retrieval Script

This script retrieves survey response documents from a Google Firestore database collection, processes them, and saves the flattened data into a single CSV file.

It is designed to handle large datasets efficiently by processing documents in chunks, creating intermediate files, and then collating them into a final output file.

## Notes

The contents of documents in the Firestore database collection differ depending on the 'path' the respondent took as they completed the survey.
As a result, the processing / flattening / reformatting is performed heuristcally rather than deterministically.
This will likely mean that we wish to perform further data manipulation on the output CSV file, before it can be used effectively with our metric calculation utilities.

Several columns are very sparsely populated, with some batches having only missing values.
As a result, when the final concatenated output CSV file is loaded in as a dataframe using Pandas, Pandas warns that these columns contain mixed dtypes. This can be handled in post-processing by casting columns as numerical / float types if required.
Excel has no issues reading the final output CSV.

## Prerequisites

1.  **Python Virtual Environment**: Activate the `survey-assist-utils` virtual environment.

2.  **Google Cloud Authentication**: You must be authenticated to the Google Cloud project containing the Firestore database.
    ```sh
    gcloud auth application-default login
    ```

## Usage

```text
usage: Utility to retrieve survey responses from a Firestore database.
python retrieve_survey_responses.py
       [-h] [--timeout TIMEOUT] [--chunk_size CHUNK_SIZE]
       project_id database_id collection_name output_name

positional arguments:
  project_id            The Google Cloud project ID.
  database_id           The Firestore database ID.
  collection_name       The collection_name.
  output_name           The name of the output CSV file.

options:
  -h, --help            show this help message and exit
  --timeout TIMEOUT, -t TIMEOUT
                        The connection timeout in seconds.
                        Defaults to 10.
  --chunk_size CHUNK_SIZE, -c CHUNK_SIZE
                        The number of documents to process in each chunk.
                        Defaults to 500.
```
