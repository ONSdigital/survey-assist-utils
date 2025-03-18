# Getting Started

See the README for general setup instructions.

## API Token Utility

This utility provides functions for generating and managing JWT tokens for authentication with the **Survey Assist API** hosted in **Google Cloud Platform (GCP)**.

### Features

- Generate a JWT using a Google service account.
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

Ensure you have the following variables set in your environment where you run this code:

```bash
export API_GATEWAY="https://api.example.com"
export SA_EMAIL="your-service-account@project.iam.gserviceaccount.com"
export JWT_SECRET="path/to/service-account.json"
```
