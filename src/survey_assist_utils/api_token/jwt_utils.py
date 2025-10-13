# Generate a JWT token for the example API
"""This module provides utility functions for generating and managing JWT tokens
for API authentication.

Functions:
    current_utc_time() -> datetime:
        Get the current UTC time.

    generate_jwt() -> str:
        Generate a JWT token using a service account key file.

    generate_api_token() -> str:
        Generate an API token using a JWT at the CLI.

"""
import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from google.auth import default
from google.cloud import iam_credentials_v1

TOKEN_EXPIRY: int = 3600  # 1 hour
REFRESH_THRESHOLD: int = 300  # 5 minutes


def current_utc_time() -> datetime:
    """Get the current UTC time.

    Returns:
        datetime: The current time in UTC as a timezone-aware datetime object.
    """
    return datetime.fromtimestamp(time.time(), tz=timezone.utc)


def resolve_jwt_secret_path(jwt_secret_env: str) -> Optional[str]:
    """Resolves the JWT secret environment variable to a file path.
    - If the value is a valid JSON string, writes it to a temp file and returns that path.
    - If it's a path to an existing file, returns it as-is.
    """
    if os.path.isfile(jwt_secret_env):
        return jwt_secret_env  # Local dev case

    try:
        # Try to parse the secret content as JSON
        secret_content = json.loads(jwt_secret_env)
        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as temp:
            json.dump(secret_content, temp)
        return temp.name
    except json.JSONDecodeError as err:
        raise ValueError(
            "JWT_SECRET must be a valid file path or JSON string."
        ) from err


def generate_jwt(
    sa_email: str,
    audience: str,
    expiry_length: int = 3600,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Mint a service account signed JWT using ADC and IAMCredentials.signJwt.

    This function creates a JWT signed by a Google service account, matching API Gateway
    configuration where:
        - x-google-issuer equals sa_email
        - x-google-jwks_uri points to the service account public keys
        - x-google-audiences equals audience

    Args:
        sa_email (str): The service account email address to use as the issuer and subject.
        audience (str): The intended audience for the JWT (e.g., API Gateway URL).
        expiry_length (int, optional): The token expiry time in seconds. Defaults to 3600.
        extra_claims (dict[str, Any] | None, optional): Additional claims to include in
        the JWT payload.

    Returns:
        str: The signed JWT string.

    Raises:
        google.auth.exceptions.DefaultCredentialsError: If ADC credentials cannot be found.
        google.api_core.exceptions.GoogleAPIError: If the IAMCredentials API call fails.

    """
    now = int(time.time())
    payload: dict[str, Any] = {
        "iat": now,
        "exp": now + expiry_length,
        "iss": sa_email,
        "sub": sa_email,
        "aud": audience,
        "email": sa_email,
    }
    if extra_claims:
        payload.update(extra_claims)

    # Get Application Default Credentials to call IAM Credentials API
    adc, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    # Use the official client to call signJwt
    client = iam_credentials_v1.IAMCredentialsClient(credentials=adc, transport="rest")
    name = f"projects/-/serviceAccounts/{sa_email}"

    resp = client.sign_jwt(
        request={"name": name, "payload": json.dumps(payload, separators=(",", ":"))}
    )
    return resp.signed_jwt


def check_and_refresh_token(
    token_start_time: int,
    current_token: str,
    api_gateway: str,
    sa_email: str,
) -> tuple[int, str]:
    """Checks if the current JWT token is still valid and refreshes it if necessary.

    If no token exists or the remaining time for the token is below the refresh threshold,
    a new token is generated using the provided JWT secret, API gateway, and service account email.

    Args:
        token_start_time (int): The UTC timestamp when the current token was created.
        current_token (str): The current JWT token.
        api_gateway (str): The intended audience for the JWT token (e.g., API gateway URL).
        sa_email (str): The service account email used for token generation.

    Returns:
        tuple: A tuple containing the updated token start time (int)
               and the refreshed or current token (str).
    """
    if not token_start_time:
        # If no token exists, create one
        token_start_time = int(current_utc_time().timestamp())
        current_token = generate_jwt(
            sa_email=sa_email, audience=api_gateway, expiry_length=TOKEN_EXPIRY
        )

    elapsed_time = (
        current_utc_time().replace(tzinfo=None)
        - datetime.fromtimestamp(token_start_time)
    ).total_seconds()
    remaining_time = TOKEN_EXPIRY - elapsed_time

    if remaining_time <= REFRESH_THRESHOLD:
        # Refresh the token
        print("Refreshing JWT token...")
        token_start_time = int(current_utc_time().timestamp())

        current_token = generate_jwt(
            sa_email=sa_email, audience=api_gateway, expiry_length=TOKEN_EXPIRY
        )

        print(f"JWT Token ends with {current_token[-5:]} created at {token_start_time}")

    return token_start_time, current_token


def generate_api_token(
    audience: Optional[str] = None,
    expiry_length: Optional[int] = None,
) -> str:
    """Generates an API token using a JWT.

    Args:
        audience (str, optional): The audience (API Gateway URL). If not provided,
            falls back to the API_GATEWAY environment variable.
        expiry_length (int, optional): Token expiry in seconds. If not provided,
            falls back to TOKEN_EXPIRY constant.

    Returns:
        str: The generated JWT token.

    Raises:
        ValueError: If required values (audience or service account email) are missing.
    """
    api_gateway = audience or os.getenv("API_GATEWAY")
    sa_email = os.getenv("SA_EMAIL")
    expiry = expiry_length or TOKEN_EXPIRY

    if not api_gateway:
        raise ValueError("API Gateway not provided and API_GATEWAY env var not set.")
    if not sa_email:
        raise ValueError("API access service account email, SA_EMAIL env var not set.")

    # Calculate expiry time (UTC for consistency)
    expiry_time = datetime.now(timezone.utc) + timedelta(seconds=expiry)
    human_readable = expiry_time.strftime("%Y-%m-%d %H:%M:%S %Z")

    print(f"Token expiry set as {human_readable} (in {expiry // 60} minutes).")

    jwt_token = generate_jwt(
        audience=api_gateway,
        sa_email=sa_email,
        expiry_length=expiry,
    )

    print(jwt_token)
    return jwt_token


def main() -> int:
    """generate-api-token CLI entrypoint.

    Args:
        audience (str, optional): The audience (API Gateway URL). If not provided,
            falls back to the API_GATEWAY environment variable.
        expiry_length (int, optional): Token expiry in seconds. If not provided,
            falls back to TOKEN_EXPIRY constant.

    Returns:
        str: The generated JWT token.
    """
    parser = argparse.ArgumentParser(
        prog="generate-api-token",
        description="Generate a short-lived JWT for the Survey Assist API.",
    )

    parser.add_argument(
        "-a",
        "--api-gateway",
        dest="audience",
        type=str,
        help="""Audience / API Gateway URL (overrides API_GATEWAY env var).
        E.g: example-api-gw.url.aws.dev (Do NOT include https://)""",
    )

    # Optional flags to allow API GATEWAY and token expiry setting from cli
    parser.add_argument(
        "-e",
        "--expiry",
        dest="expiry_length",
        type=int,
        help="Token expiry in seconds (default 3600s / 1h)",
    )
    args = parser.parse_args()

    try:
        generate_api_token(audience=args.audience, expiry_length=args.expiry_length)
        return 0
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
