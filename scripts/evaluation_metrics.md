# Evaluation Metrics

## Overview
This module allows user to evaluate the quality of the Survey Assist SIC code assignemnt based on responses provided, compared to the decision made by Clerical Coders on the same set of data.

It allows to calculate the accuracy of the assignment as a percentage for one or two prompt pipeline. Those metrics cna be used to compare the effectiveness of two approaches.

## Setup
### Prerequsites
This project uses Poetry for dependency management. Please ensure you have Poetry installed.
- Python 3.12
- Poetry
- Google Cloud SDK

Once Poetry is installed, set up the project by running
``` bash
poetry install
```
### Data
To access the data, use the TLFS data. You can find it in the GCP bucket.
1. Start up the vector store (`sic-classification-vector-store` repo, run `make run-vector-store`).

In the `sic-classification-utils` repo (steps 2-4):

2. Authenticate/re-authenticate gcloud `gcloud auth application-default login`.
3. Create the metadata file (template is available in `sic-classification-utils` repo in `scripts/stage_1_add_semantic_search.py` within the module-level docstring)
4. Process the data using scripts, according to the one- or two- prompt approach. It is recommended to use `run_full_pipeline.sh`, available in `sic_classification_utils/scripts`.
    - With `OnePromptOutputs` as the output folder and 20 as the batch size, for **one prompt** pipeline, use:
    ```
    bash ./run_full_pipeline.sh 1 OnePromptOutputs /path/to/tlfs_data.csv /path/to/tlfs_data_metadata.json 20
    ```
    - With `TwoPromptOutputs` as the output folder and 20 as the batch size, for **two prompt** pipeline, use:
    ```
    bash ./run_full_pipeline.sh 2 TwoPromptOutputs /path/to/tlfs_data.csv /path/to/tlfs_data_metadata.json 20
    ```
The data will be saved in the specified folder (e.g., OnePromptOutputs, or TwoPromptOutputs) as `STG2_oneprompt.parquet` for one prompt pipeline, or `STG5.parquet` for two prompts pipeline.

## Usage
Move the data (`STG2_oneprompt.parquet` and `STG5.parquet`) to this repo (`survey-assist-utils`).
To run the evaluation for **one prompt** use:
```
python scripts/one_prompt_evaluation_revised.py data/STG2_oneprompt.parquet <OO / MM / OM / MO> <full / 2-digit> <-fua / -fa> -n
```
To run the evaluation for **two prompt** use:
```
python scripts/two_prompt_evaluation_revised.py data/STG5.parquet <OO / MM / OM / MO> <full / 2-digit> <-fua / -fa> -n
```
Arguments:
- `scripts/one_prompt_evaluation_revised.py`, `scripts/two_prompt_evaluation_revised.py`: relative path to the script, corresponding to the evaluation type.
- `data/STG2_oneprompt.parquet`, `data/STG5.parquet`: relative path to the parquet dataset.
- `<OO / MM / OM / MO>`: test type, where *M* stnads for Many, *O* stands for One. Represents the number of matches to be found. Format: Clerical Coder - Survey Assist:
    - `OO`: Clerical Coder and Survey Assist model agree exactly.
    - `MM`: Any of the Clerical Coder's choices is in the Survey Assist's choices.
    - `OM`: Clerical Coder's choice is one of the choices made by Survey Assist model.
    - `MO`: Any of the Clerical Coder's choices match one of the choices by Survey Assist model. 
- `<full / 2-digit>`: match type; the length of code to match between CC and SA choices.
    - `full`: match full, 5-digit code, e.g. CC code \`12345\` matches SA code \`12345\`, but CC code \`12345\` does not match SA code \`12333\`.
    - `2-digit`: match first two digits form the SIC code, e.g. CC code \`12345\` matches SA code \`12345\`, and CC code \`12345\` matches SA code \`12333\`, but CC code \`12345\` does not match SA code \`11111\`. 
- `<-fua / -fa>` (optional): boolean filters, default *False*:
    - `-fua`: \`filter unambiguous\`, considers only those responses that Clerical Coders reported as unambiguously codable. 
    - `-fa`: \`filter ambiguous\`, considers only those responses that Clerical Coders reported as NOT unambiguously codable.
    - neither \`-fua`\`, nor \`-fa\` present: consideres all responses, regardless of ambiguity.
- `-n` (optional): \`neglect impossible\`, default *False*, ignore rows where no n-digit Clerical Coder is available when calculating accuracy, i.e., neglects what CC marked as \`4+\`, meaning there is more than four possible SIC codes for the response.

## Metrics
The output should look like this:
```
<`Only considering CC-recorded NOT unambiguously codable records:` / `Only considering CC-recorded unambiguously codable records:` / `Considering ALL records`>
<Optional: `n records had no usable clerically coded answer, and are ignored in calculation`>

test type: <OO/MM/OM/MO>
accuracy <full / 2-digit>: [100 * a/b] %
matches  <full / 2-digit>: [a]
non_matches  <full / 2-digit>: [b-a]
total considered: [b]
```

The output informs which records were selected (ambiguous/unambiguous/all), and if specified, the number of records ignored, as well as the test and match type, followed with the metrics of accuracy, number of matches and number of non matches, number of non-matches, and total number of responses considered.</br>
The accuracy is calculated as a percentage of matches divided by the total responses considered in the metric.

## Abbreviations
- CC - Clerical Coder
- SA - Survey Assist
- SIC - Standard Industrial Classification