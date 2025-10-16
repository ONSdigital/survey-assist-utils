# Scripts for Calculating Evaluation Metrics

This module allows user to evaluate the quality of the Survey Assist (**SA**) Standard Industry Code (**SIC**) assignment based on responses provided, compared to the decision made by Clerical Coders (**CC**) on the same set of data.

The individual scripts are designed to handle different data pre-processing steps, but share the same evaluation logic.

The input data are expected to be in a parquet format, created by evaluation pipeline scripts in the `sic-classification-utils` repo, see notes at the end of this document.

Additionally, examples of visualizations of the evaluation metrics are available in the `notebooks/2025-10_semantic_vis.py` notebook.

## Usage

In general, the scripts can be run from the command line and takes several arguments. The evaluation metrics will be printed to the console.

---
For the `get_evaluation_metrics.py` script, use:
```{shell}
python scripts/get_evaluation_metrics.py <data_path> [-n <number-of-digits>] [-c <clerical_file>] [-w]
```
Where:
- `scripts/get_evaluation_metrics.py`: relative path to the script.
- `<data_path>`: relative path to the parquet dataset (could be local or gs bucket).
- `-n <number-of-digits>` (optional): `--number-of-digits <number-of-digits>` number of digits to match between CC and SA choices, default *5* (full match). Accepts values: "0", "1", "2", "3", "4", "5".
- `-c <clerical_file>` (optional): `--clerical_file <clerical_file>` Path to the clerically coded file (ground truth).
        If not provided, the main data file is expected to include clerical columns.
- `-w` (optional): `--write_output`, default *False*. If set, writes the evaluation metrics to a JSON file.


Use the `-c` flag if the clerical coding data is stored in a separate file, for example to point to newer iteration of clerical codes. The `unique_id` column is used to match the records between the two files.


## Abbreviations
- CC - Clerical Coder
- SA - Survey Assist
- SIC - Standard Industrial Classification
- OO: One-to-One match on a subset where the true label as well as the model's label are not ambiguous.
- OM: One-to-Many match on a subset where the true label is not ambiguous. (Is the true label in the model's shortlist?)
- MO: Many-to-One match on a subset where the model is not ambiguous. (Is the model's label in the true label shortlist?)
- MM: Many-to-Many match on the full set. (Is there any overlap between the true label's and model's shortlists?)

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
The data will be saved in the specified folder (e.g., OnePromptOutputs, or TwoPromptOutputs) as `STG5.parquet` for both pipelines.
