# Survey Assist Utils

Test change - Utilities used as part of Survey Assist API or UI

## Overview

Survey Assist utility functions. These are common pieces of functionality that can be used by the UI or API.

## Features

- Generate a JWT token for authenticating to the API

## Prerequisites

Ensure you have the following installed on your local machine:

- [ ] Python 3.12 (Recommended: use `pyenv` to manage versions)
- [ ] `poetry` (for dependency management)
- [ ] Colima (if running locally with containers)
- [ ] Terraform (for infrastructure management)
- [ ] Google Cloud SDK (`gcloud`) with appropriate permissions

### Local Development Setup

The Makefile defines a set of commonly used commands and workflows.  Where possible use the files defined in the Makefile.

#### Clone the repository

```bash
git clone https://github.com/ONSdigital/survey-assist-utils.git
cd survey-assist-utils
```

#### Install Dependencies

```bash
poetry install
```

#### Run the Token Generator Locally

Set the following environment variables

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/GCP_CREDENTIALS.json"
export SA_EMAIL="GCP-SERVICE-ACCOUNT@SERVICE-ACCOUNT-ID.iam.gserviceaccount.com"
export JWT_SECRET=/path/to/GCP/secret.json
```

To generate an API token execute:

```bash
make generate-api-token
```

### GCP Setup

Placeholder

### Code Quality

Code quality and static analysis will be enforced using isort, black, ruff, mypy and pylint. Security checking will be enhanced by running bandit.

To check the code quality, but only report any errors without auto-fix run:

```bash
make check-python-nofix
```

To check the code quality and automatically fix errors where possible run:

```bash
make check-python
```

### Documentation

Documentation is available in the docs folder and can be viewed using mkdocs

```bash
make run-docs
```

### Testing

Pytest is used for testing alongside pytest-cov for coverage testing.  [/tests/conftest.py](/tests/conftest.py) defines config used by the tests.

Unit testing for utility functions is added to the [/tests/tests_utils.py](./tests/tests_utils.py)

```bash
make unit-tests
```

All tests can be run using

```bash
make all-tests
```

### Environment Variables

Placeholder
