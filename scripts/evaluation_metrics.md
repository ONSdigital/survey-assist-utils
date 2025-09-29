# Scripts for Calculating Evaluation Metrics

This module allows user to evaluate the quality of the Survey Assist (**SA**) Standard Industry Code (**SIC**) assignment based on responses provided, compared to the decision made by Clerical Coders (**CC**) on the same set of data.

The individual scripts are designed to handle different data pre-processing steps, but share the same evaluation logic.

The input data are expected to be in a parquet format, created by evaluation pipeline scripts in the `sic-classification-utils` repo, see notes at the end of this document.

## Usage

In general, the scripts can be run from the command line and takes several arguments. The evaluation metrics will be printed to the console.

---
For the `get_evaluation_metrics.py` script, use:
```{shell}
python scripts/get_evaluation_metrics.py <data_path> <num-digit> [-o] [-n]
```
Where:
- `scripts/get_evaluation_metrics.py`: relative path to the script.
- `<data_path>`: relative path to the parquet dataset (could be local or gs bucket).
- `<num-digit>`: number of digits to match between CC and SA choices (accepts "full", "1-digit", "2-digit", ..., "5-digit").
- `-o` (optional): `--old_one_prompt`, default *False*, expect data in a fromat from the old one-prompt pipeline.
- `-c` (optional): `--clerical_file <clerical_file>` Path to the clerically coded file (ground truth).
        If not provided, the main data file is expected to include clerical columns.

If input data used has been prepared by the two prompt pileline (`STG5.parquet`) then it contains final code based on synthetic answer to follow up question. While the data prepared by the one prompt pileline (`STG2_oneprompt.parquet`) includes only initial SIC code and follow slightly different formats, therefore make sure to use the `-o` flag in that case. Use the `-c` flag only if the clerical coding data is stored in a separate file, for example to point to newer iteration of clerical codes. The `unique_id` column is used to match the records between the two files.


---
For the `one_prompt_evaluation_revised.py` and `two_prompt_evaluation_revised.py` scripts, use:
```{shell}
python scripts/one_prompt_evaluation_revised.py data/STG2_oneprompt.parquet <OO / MM / OM / MO> <full / 2-digit> <-fua / -fa> -n

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
- `-n` (optional): \`neglect impossible\`, default *False*, ignore rows that CC marked as \`4+\`, meaning there is more than four possible SIC codes for the response.

## Metrics
The output will include information about which records were considered/filtered and what type of test and match was performed. The accuracy is calculated as a percentage of matches divided by the total responses considered in the metric.

## Abbreviations
- CC - Clerical Coder
- SA - Survey Assist
- SIC - Standard Industrial Classification

---
---
## Additional notes on running the evaluation pipeline

This section provides additional information on preparing the data for metrics calculation. It also includes context on running the evaluation pipelines within `sic-classification-utils` repo.

### Prerequisites
- Python 3.12
- Poetry (This project uses Poetry for dependency management)
- Google Cloud SDK

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
