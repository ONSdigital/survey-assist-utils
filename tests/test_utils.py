"""Test module for utility functions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch

import survey_assist_utils.api_token.jwt_utils as tokens
from survey_assist_utils import get_logger
from survey_assist_utils.api_token.jwt_utils import current_utc_time

logger = get_logger(__name__)

# pylint: disable=too-few-public-methods, import-outside-toplevel, unused-argument


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


@pytest.mark.utils
@pytest.mark.adc
def test_generate_jwt_success(mock_gcp_adc_and_iam, reset_iam_spy) -> None:
    """Ensure a signed JWT is returned and IAM `sign_jwt` is called with correct payload.

    Verifies:
      * The function returns the fake signed JWT.
      * The request payload contains the expected standard claims.
      * The IAM client is initialised with the ADC credentials.
    """
    # Arrange
    mock_gcp_adc_and_iam("ey.ok.jwt")  # set fake token
    sa_email = "svc-abc@project.iam.gserviceaccount.com"
    audience = "https://apigw-id.a.run.app"

    # Act
    signed = tokens.generate_jwt(
        sa_email=sa_email, audience=audience, expiry_length=3600
    )

    # Assert
    assert signed == "ey.ok.jwt"

    # Introspect IAM spy
    from tests.conftest import _IamClientSpy as Spy  # type: ignore

    assert Spy.last_instance is not None
    req = Spy.last_instance.last_request
    assert req is not None
    assert req["name"] == f"projects/-/serviceAccounts/{sa_email}"

    payload = json.loads(req["payload"])
    assert payload["iss"] == sa_email
    assert payload["sub"] == sa_email
    assert payload["aud"] == audience
    assert payload["email"] == sa_email
    assert isinstance(payload["iat"], int)
    assert isinstance(payload["exp"], int)
    assert payload["exp"] > payload["iat"]


@pytest.mark.utils
@pytest.mark.adc
def test_generate_jwt_includes_extra_claims(
    mock_gcp_adc_and_iam, reset_iam_spy
) -> None:
    """Validate that extra claims are merged into the JWT payload.

    Asserts:
        The `sign_jwt` payload includes the provided custom claims.
    """
    # Arrange
    mock_gcp_adc_and_iam("ey.extra.jwt")
    extra = {"role": "system", "scope": ["read", "write"]}

    # Act
    _ = tokens.generate_jwt(
        sa_email="svc@x.iam.gserviceaccount.com",
        audience="https://gw",
        expiry_length=120,
        extra_claims=extra,
    )

    # Assert
    from tests.conftest import _IamClientSpy as Spy  # type: ignore

    # Narrow optional types for mypy (and fail fast if the spy wasn't exercised)
    assert Spy.last_instance is not None, "IAM client was not initialised"
    req = Spy.last_instance.last_request
    assert req is not None, "sign_jwt was not called"

    payload = json.loads(req["payload"])
    assert payload["role"] == "system"
    assert payload["scope"] == ["read", "write"]


@pytest.mark.utils
def test_generate_jwt_adc_failure(monkeypatch: MonkeyPatch) -> None:
    """Bubble up ADC errors as DefaultCredentialsError."""

    class _ADCError(Exception):
        pass

    def _fail_default(*_: Any, **__: Any) -> tuple[None, None]:
        raise _ADCError("no ADC")

    # Patch at the import location used by the function
    monkeypatch.setattr(tokens, "default", _fail_default, raising=False)

    with pytest.raises(Exception) as exc:
        tokens.generate_jwt("svc@x", "aud")
    assert "no ADC" in str(exc.value)


@pytest.mark.utils
@pytest.mark.adc
def test_generate_jwt_iam_api_error(monkeypatch: MonkeyPatch) -> None:
    """Bubble up underlying IAMCredentials API errors."""

    class _Client:
        def __init__(self, *, credentials: Any, transport="rest") -> None:
            self.credentials = credentials

        def sign_jwt(self, *, request: dict[str, Any]) -> Any:
            """Simulate sign_jwt error."""
            raise RuntimeError("iam failure")

    class _DummyCreds:
        pass

    monkeypatch.setattr(
        "google.auth.default", lambda *a, **k: (_DummyCreds(), None), raising=False
    )
    monkeypatch.setattr(
        "google.cloud.iam_credentials_v1.IAMCredentialsClient", _Client, raising=False
    )

    with pytest.raises(RuntimeError, match="iam failure"):
        tokens.generate_jwt("svc@x", "aud")


@pytest.mark.utils
def test_check_and_refresh_token_creates_when_missing(
    monkeypatch: MonkeyPatch, freeze_time
) -> None:
    """Create a token when none exists and set start time to current UTC timestamp."""
    # Fix time to a known instant
    freeze_time("survey_assist_utils.api_token.jwt_utils", 1_700_000_000.0)

    # Make expiry/threshold deterministic
    monkeypatch.setattr(tokens, "TOKEN_EXPIRY", 3600, raising=True)
    monkeypatch.setattr(tokens, "REFRESH_THRESHOLD", 300, raising=True)

    # Stub generate_jwt to track calls and return a predictable token
    gen_mock = MagicMock(return_value="ey.new.jwt")
    monkeypatch.setattr(tokens, "generate_jwt", gen_mock, raising=True)

    start, tok = tokens.check_and_refresh_token(
        token_start_time=0,
        current_token="",
        api_gateway="https://gw",
        sa_email="svc@x",
    )

    assert start == 1_700_000_000  # noqa: PLR2004
    assert tok == "ey.new.jwt"
    gen_mock.assert_called_once_with(
        sa_email="svc@x", audience="https://gw", expiry_length=3600
    )


@pytest.mark.utils
def test_check_and_refresh_token_no_refresh_needed(
    monkeypatch: MonkeyPatch, freeze_time
) -> None:
    """Do not refresh if remaining lifetime is above the refresh threshold."""
    # Now = t0 + 100 seconds -> plenty of time remaining
    t0 = 1_700_000_000
    freeze_time("survey_assist_utils.api_token.jwt_utils", t0 + 100.0)

    monkeypatch.setattr(tokens, "TOKEN_EXPIRY", 3600, raising=True)
    monkeypatch.setattr(tokens, "REFRESH_THRESHOLD", 300, raising=True)

    gen_mock = MagicMock(return_value="should-not-be-used")
    monkeypatch.setattr(tokens, "generate_jwt", gen_mock, raising=True)

    start, tok = tokens.check_and_refresh_token(
        token_start_time=t0,
        current_token="ey.current",  # noqa: S106
        api_gateway="https://gw",
        sa_email="svc@x",
    )

    assert start == t0
    assert tok == "ey.current"
    gen_mock.assert_not_called()


@pytest.mark.utils
def test_check_and_refresh_token_refreshes_when_threshold_reached(
    monkeypatch: MonkeyPatch, freeze_time
) -> None:
    """Refresh when remaining lifetime is at or below the configured threshold."""
    # TOKEN_EXPIRY = 3600, REFRESH_THRESHOLD = 300
    # If elapsed >= 3300, we must refresh.
    t0 = 1_700_000_000
    freeze_time("survey_assist_utils.api_token.jwt_utils", t0 + 3300.0)

    monkeypatch.setattr(tokens, "TOKEN_EXPIRY", 3600, raising=True)
    monkeypatch.setattr(tokens, "REFRESH_THRESHOLD", 300, raising=True)

    gen_mock = MagicMock(return_value="ey.refreshed")
    monkeypatch.setattr(tokens, "generate_jwt", gen_mock, raising=True)

    start, tok = tokens.check_and_refresh_token(
        token_start_time=t0,
        current_token="ey.stale",  # noqa: S106
        api_gateway="https://gw",
        sa_email="svc@x",
    )

    # Start time should have been updated to "now"
    assert start == t0 + 3300
    assert tok == "ey.refreshed"
    gen_mock.assert_called_once()


@pytest.mark.utils
def test_generate_api_token_reads_env_and_prints(
    monkeypatch: MonkeyPatch, env: dict[str, str]
) -> None:
    """Ensure `generate_api_token` reads environment, calls `generate_jwt`, and prints token.

    This is a CLI function, so stdout is validated as the user visible side effect.
    """
    env["API_GATEWAY"] = "https://gw"
    env["SA_EMAIL"] = "svc@x"

    monkeypatch.setattr(tokens, "TOKEN_EXPIRY", 3600, raising=True)
    monkeypatch.setattr(tokens, "generate_jwt", lambda **_: "ey.cli.jwt", raising=True)

    # Capture stdout
    import io
    import sys

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf, raising=False)

    tokens.generate_api_token()

    # The first line is token expiry, second is actual token
    # just compare the last line.
    out = buf.getvalue().strip().splitlines()[-1]
    assert out == "ey.cli.jwt"
