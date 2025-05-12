"""Module that provides example test functions for the Survey Assist API.

Unit tests for endpoints and utility functions in the Survey Assist API.
"""

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from survey_assist_utils.api_token.jwt_utils import (
    REFRESH_THRESHOLD,
    TOKEN_EXPIRY,
    check_and_refresh_token,
    current_utc_time,
    generate_api_token,
    generate_jwt,
)

logger = logging.getLogger(__name__)


@pytest.mark.utils
def test_current_utc_time():
    """Test the current_utc_time function."""
    result = current_utc_time()

    # Assert that the result is a datetime object
    assert isinstance(result, datetime)

    # Assert that the result is timezone-aware and in UTC
    assert result.tzinfo is not None
    assert result.tzinfo.utcoffset(result) == timezone.utc.utcoffset(result)

    # Assert that the result is close to the current time
    now = datetime.now(timezone.utc)
    assert abs((now - result).total_seconds()) < 1  # Allow a small time difference


@patch("utils.api_token.jwt_utils.RSASigner.from_service_account_file")
@patch("utils.api_token.jwt_utils.google_jwt.encode")
@pytest.mark.utils
def test_generate_jwt(mock_encode, mock_rsa_signer):
    """Test the generate_jwt function with mocked dependencies."""
    # Mock inputs
    sa_keyfile = "/path/to/mock/keyfile.json"
    sa_email = "mock-account@project.iam.gserviceaccount.com"
    audience = "mock-service-name"
    expiry_length = 3600

    # Mock RSASigner and encode behavior
    mock_signer_instance = MagicMock()
    mock_rsa_signer.return_value = mock_signer_instance
    mock_encode.return_value = b"mock_jwt_token"

    # Call the function
    jwt_token = generate_jwt(
        sa_keyfile=sa_keyfile,
        sa_email=sa_email,
        audience=audience,
        expiry_length=expiry_length,
    )

    # Assertions
    assert jwt_token == "mock_jwt_token"  # noqa: S105
    mock_rsa_signer.assert_called_once_with(sa_keyfile)
    mock_encode.assert_called_once()
    payload = mock_encode.call_args[0][1]  # Extract the payload from the call arguments
    assert payload["iss"] == sa_email
    assert payload["aud"] == audience
    assert payload["sub"] == sa_email
    assert "iat" in payload
    assert "exp" in payload


@patch("utils.api_token.jwt_utils.current_utc_time")
@patch("utils.api_token.jwt_utils.generate_jwt")
@pytest.mark.utils
def test_check_and_refresh_token(mock_generate_jwt, mock_current_utc_time):
    """Test the check_and_refresh_token function."""
    # Mock current time
    mock_time = datetime(2023, 1, 1, 12, 0, 0)
    mock_current_utc_time.return_value = mock_time

    # Mock JWT generation
    mock_generate_jwt.return_value = "mock_jwt_token"

    # Test case 1: No token exists (token_start_time is None)
    token_start_time = None
    current_token = None
    jwt_secret_path = "/path/to/mock/keyfile.json"  # noqa: S105
    api_gateway = "mock-service-name"
    sa_email = "mock-account@project.iam.gserviceaccount.com"

    token_start_time, current_token = check_and_refresh_token(
        token_start_time, current_token, jwt_secret_path, api_gateway, sa_email
    )

    assert token_start_time == int(mock_time.timestamp())
    assert current_token == "mock_jwt_token"  # noqa: S105
    mock_generate_jwt.assert_called_once_with(
        jwt_secret_path,
        audience=api_gateway,
        sa_email=sa_email,
        expiry_length=TOKEN_EXPIRY,
    )

    # Test case 2: Token exists but needs refreshing
    mock_generate_jwt.reset_mock()
    token_start_time = int(
        (
            mock_time - timedelta(seconds=TOKEN_EXPIRY - REFRESH_THRESHOLD + 1)
        ).timestamp()
    )
    current_token = "old_jwt_token"  # noqa: S105

    token_start_time, current_token = check_and_refresh_token(
        token_start_time, current_token, jwt_secret_path, api_gateway, sa_email
    )

    assert token_start_time == int(mock_time.timestamp())
    assert current_token == "mock_jwt_token"  # noqa: S105
    mock_generate_jwt.assert_called_once_with(
        jwt_secret_path,
        audience=api_gateway,
        sa_email=sa_email,
        expiry_length=TOKEN_EXPIRY,
    )

    # Test case 3: Token exists and does not need refreshing
    mock_generate_jwt.reset_mock()
    token_start_time = int(
        (
            mock_time - timedelta(seconds=TOKEN_EXPIRY - REFRESH_THRESHOLD - 1)
        ).timestamp()
    )
    current_token = "valid_jwt_token"  # noqa: S105

    token_start_time, current_token = check_and_refresh_token(
        token_start_time, current_token, jwt_secret_path, api_gateway, sa_email
    )

    assert token_start_time == int(
        (
            mock_time - timedelta(seconds=TOKEN_EXPIRY - REFRESH_THRESHOLD - 1)
        ).timestamp()
    )
    assert current_token == "valid_jwt_token"  # noqa: S105
    mock_generate_jwt.assert_not_called()


@patch("utils.api_token.jwt_utils.generate_jwt")
@patch("utils.api_token.jwt_utils.os.getenv")
@pytest.mark.utils
def test_generate_api_token(mock_getenv, mock_generate_jwt, capsys):
    """Test the generate_api_token function."""
    # Mock environment variables
    mock_getenv.side_effect = {
        "API_GATEWAY": "mock-api-gateway",
        "SA_EMAIL": "mock-account@project.iam.gserviceaccount.com",
        "JWT_SECRET": "/path/to/mock/keyfile.json",
    }.get

    # Mock JWT generation
    mock_generate_jwt.return_value = "mock_jwt_token"

    # Call the function
    generate_api_token()

    # Capture printed output
    captured = capsys.readouterr()

    # Assertions
    assert "mock_jwt_token" in captured.out
    mock_getenv.assert_any_call("API_GATEWAY")
    mock_getenv.assert_any_call("SA_EMAIL")
    mock_getenv.assert_any_call("JWT_SECRET")
    mock_generate_jwt.assert_called_once_with(
        "/path/to/mock/keyfile.json",
        audience="mock-api-gateway",
        sa_email="mock-account@project.iam.gserviceaccount.com",
        expiry_length=3600,
    )
