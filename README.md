# Survey Assist Utils

Utilities used as part of Survey Assist API or UI

## Overview

Survey Assist utility functions. These are common pieces of functionality that can be used by the UI or API, with a primary focus on providing a framework for batch processing and evaluating LLM-based SIC code classification.

## Features

* **JWT Token Generation:** Authenticate to the Survey Assist API.
* **Batch Processing:** Send large datasets to the API for SIC classification.
* **Data Enrichment:** Add data quality and metadata flags to datasets.
* **Performance Evaluation:** A comprehensive suite of metrics to analyze and compare LLM performance against human coders.

## Local Development & Setup

The Makefile defines a set of commonly used commands and workflows.  Where possible use the files defined in the Makefile.

### Prerequisites

Ensure you have the following installed on your local machine:

* Python 3.12 (Recommended: use `pyenv` to manage versions)
* `poetry` (for dependency management)
* Google Cloud SDK (`gcloud`) with appropriate permissions
* Colima (if running locally with containers)
* Terraform (for infrastructure management)

### Setup Instructions

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/ONSdigital/survey-assist-utils.git](https://github.com/ONSdigital/survey-assist-utils.git)
    cd survey-assist-utils
    ```

2. **Create and activate a virtual environment**

    Using `pyenv` and `pyenv-virtualenv`:

    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    poetry install
    ```

4. **Generate an API Token**

    The API uses Application Default Credentials to generate and authenticate tokens.

    Ensure GOOGLE_APPLICATION_CREDENTIALS are not set in your environment.

    ```bash
    unset GOODLE_APPLICATION_CREDENTIALS
    ```

    Login to gcloud application default:

    ```bash
    gcloud auth application-default login
    ```

    Set to the correct GCP project:

    ```bash
    gcloud auth application-default set-quota-project GCP-PROJECT-NAME
    ```

    Check the project setting:

    ```bash
    cat ~/.config/gcloud/application_default_credentials.json
    ```

    Set the required environment variables:

    ```bash
    export SA_EMAIL="SERVICE-ACCOUNT-FOR-API-ACCESS"
    export API_GATEWAY="API GATEWAY URL NOT INC https://"
    ```

    Then, run the make command to use default expiry (1h):

    ```bash
    make generate-api-token
    ```

    You can run from cli and pass in a chosen expiry time:

    ```bash
    poetry run generate-api-token -e 7200
    ```

## Code Quality & Testing

### Code Quality

Code quality and static analysis are enforced using `isort`, `black`, `ruff`, `mypy`, `pylint`, and `bandit`.

* **To check for errors without auto-fixing:**
    ```bash
    make check-python-nofix
    ```
* **To check and automatically fix errors:**
    ```bash
    make check-python
    ```

### Testing

Pytest is used for testing.

* **To run unit tests:**
    ```bash
    make unit-tests
    ```
* **To run all tests:**
    ```bash
    make all-tests
    ```

### Pre-commit Hooks

Pre-commit hooks are set up to run code quality checks before each commit. They will call `make check-python` under the hood as well.
To install the hooks, run:

```bash
pre-commit install
```

# Methodology for evaluating alignment between clerical coders and Survey Assist outputs

## The Evaluation Workflow

See [`scripts/evaluation_metrics.md`](./scripts/evaluation_metrics.md) for details on running the evaluation scripts.

**Legacy notes on the evaluation process:**

1.  **Stage 1: Batch Processing (`process_tlfs_evaluation_data.py`)**
    * **Input:** A CSV file containing survey responses (e.g., job title, industry description).
    * **Process:** This script iterates through the input data, sending each record to the Survey Assist API to be classified by the LLM.
    * **Output:** A JSON file containing the raw LLM responses, including the list of candidate SIC codes and likelihood scores for each survey record.

2.  **Stage 2: Data Preparation (`prepare_evaluation_data_for_analysis.py`)**
    * **Input:** The original, human-coded dataset.
    * **Process:** This script enriches the original data by adding a series of data quality flags. It analyses the human-coded SICs to determine if a response is complete, ambiguous, or requires special handling.
    * **Output:** An enriched CSV file with additional metadata columns (e.g., `Unambiguous`, `Match_5_digits`).

3. **Stage 3: Data Cleaning (`data_cleaner.py`)**
    * Before analysis, the data file needs to be cleaned weith this module

5.  **Stage 5: JSON merging**
    This is a work in progress and will be added later

6.  **Stage 6: Performance Analysis (`coder_alignment.py`)**
    * **Input:** A merged DataFrame containing both the raw LLM output from Stage 1 and the enriched human-coded data from Stage 2.
    * **Process:** The `LabelAccuracy` class takes this combined data and calculates a suite of metrics to measure the alignment between the LLM's suggestions and the human-provided ground truth.
    * **Output:** Quantitative metrics and visualisations (e.g., heatmaps, charts) that summarise the model's performance.


**Core Evaluation Metrics**

The `coder_alignment` module provides several key metrics to assess performance from different angles:

* **Match Accuracy:** This is the primary KPI, measuring how often a correct code appears anywhere in the model's suggestion list. It provides a top-level view of whether the model is providing useful answers.
* **Jaccard Similarity:** This metric is to measure the overall relevance of the suggestion list. It helps determine if the model's suggestions are closely align with the human coder's choices
* **Candidate Ranking & Contribution:** This analysis assesses the value of each individual suggestion (e.g., the 3rd or 5th candidate). It helps answer business questions about the optimal number of suggestions to display to a user.
* **Error Pattern Analysis (Confusion Matrix):** This provides a visual heatmap to diagnose systematic errors. It shows if the model consistently confuses two specific codes, and is used for prompt engineering and model improvement.
* **Confidence vs. Coverage Analysis:** The framework includes tools to plot model confidence scores against accuracy and coverage, showing the trade-off for confidence at various levels.
