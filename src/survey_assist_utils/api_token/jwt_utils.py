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
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Optional

from google.auth import jwt as google_jwt
from google.auth.crypt import RSASigner

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
    sa_keyfile: str,
    sa_email: str = "account@project.iam.gserviceaccount.com",
    audience: str = "service-name",
    expiry_length: int = 3600,
) -> str:
    """Generates a JSON Web Token (JWT) for authentication using a Google service account.

    Args:
        sa_keyfile (str): The file path to the service account key file (JSON format).
        sa_email (str, optional): The email address of the service account.
                                  Defaults to "account@project.iam.gserviceaccount.com".
        audience (str, optional): The intended audience for the token, typically the service name.
                                  Defaults to "service-name".
        expiry_length (int, optional): The token's expiration time in seconds.
                                       Defaults to 3600 (1 hour).

    Returns:
        str: The generated JWT as a string.
    """
    now: int = int(time.time())

    payload: dict[str, Any] = {
        "iat": now,
        "exp": now + expiry_length,
        "iss": sa_email,
        "aud": audience,
        "sub": sa_email,
        "email": sa_email,
    }

    signer = RSASigner.from_service_account_file(sa_keyfile)
    jwt: bytes = google_jwt.encode(signer, payload)

    # The actual token is between b'my_jwt_token' so strip the b' and '
    return jwt.decode("utf-8")


def check_and_refresh_token(
    token_start_time: int,
    current_token: str,
    jwt_secret_path: str,
    api_gateway: str,
    sa_email: str,
) -> tuple[int, str]:
    """Checks if the current JWT token is still valid and refreshes it if necessary.

    If no token exists or the remaining time for the token is below the refresh threshold,
    a new token is generated using the provided JWT secret, API gateway, and service account email.

    Args:
        token_start_time (int): The UTC timestamp when the current token was created.
        current_token (str): The current JWT token.
        jwt_secret_path (str): The file path to the JWT secret used for token generation.
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
            jwt_secret_path,
            audience=api_gateway,
            sa_email=sa_email,
            expiry_length=TOKEN_EXPIRY,
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
            jwt_secret_path,
            audience=api_gateway,
            sa_email=sa_email,
            expiry_length=TOKEN_EXPIRY,
        )
        print(f"JWT Token ends with {current_token[-5:]} created at {token_start_time}")

    return token_start_time, current_token


def generate_api_token():
    """Generates an API token using a JWT at the CLI.

    This function retrieves necessary environment variables, such as the API gateway URL,
    service account email, and the path to the JWT secret file. It then generates a JWT
    token with a default expiry of 1 hour.

    Returns:
        str: The generated JWT token.
    """
    api_gateway = os.getenv("API_GATEWAY")
    sa_email = os.getenv("SA_EMAIL")
    jwt_secret_path = os.getenv("JWT_SECRET")

    # Generate JWT (lasts 1 hour - rotate before expiry)
    jwt_token = generate_jwt(
        jwt_secret_path,
        audience=api_gateway,
        sa_email=sa_email,
        expiry_length=TOKEN_EXPIRY,
    )

    print(jwt_token)
