# Getting Started

See the README for general setup instructions.

## API Token Utility

This utility provides functions for generating and managing JWT tokens for authentication with the **Survey Assist API** hosted in **Google Cloud Platform (GCP)**.

### Features

- Generate a JWT using default application credentials.
- Check and refresh tokens before expiry.
- Generate API tokens from the command line.

### Installation

To use this code in another repository using ssh:

```bash
poetry add git+ssh://git@/ONSdigital/survey-assist-utils.git@v0.1.0
```

or https:

```bash
poetry add git+https://github.com/ONSdigital/survey-assist-utils.git@v.0.1.0
```

### Environment

Setup for default application credentials:

Make sure Google Application Credential keyfiles are not set:

```bash
unset GOOGLE_APPLICATION_CREDENTIALS
```

Login to gcloud auth:

```bash
gcloud auth login
```

Apply application default credentials for auth:

```bash
gcloud auth application-default login 
```

Ensure you have the following variables set in your environment where you run this code:

```bash
export API_GATEWAY="https://api.example.com"
export SA_EMAIL="your-service-account@project.iam.gserviceaccount.com"
```

### Generate a Token

From the root of the project execute:

```bash
make generate-api-token
```


