"""This module contains pytest configuration, hooks and shared fixtures.

It sets up a global logger and defines hooks for pytest to log events
such as the start and finish of a test session.

Functions:
    pytest_configure(config): Applies global test configuration.
    pytest_sessionstart(session): Logs the start of a test session.
    pytest_sessionfinish(session, exitstatus): Logs the end of a test session.
This module contains pytest configuration

Fixtures defined in this file are automatically discovered by pytest and can be
used in any test file within this directory without needing to be imported.

Fixtures:
    raw_data_and_config: Provides a consistent, raw DataFrame and ColumnConfig
                         object for use across multiple test suites.
"""

# pylint: disable=missing-function-docstring, redefined-outer-name, line-too-long, unused-argument
from __future__ import annotations

import datetime as dt
import logging
import os
import sys
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar

import numpy as np
import pandas as pd
import pytest
from pytest import MonkeyPatch

# REFACTOR: Import the config class to be used in the shared fixture.
from survey_assist_utils.configs.column_config import ColumnConfig

# Add src directory to Python path
SRC_PATH = str(Path(__file__).parent.parent / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Configure a global logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Adjust level as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)
)


def pytest_configure(config):  # pylint: disable=unused-argument
    """Hook function for pytest that is called after command line options have been parsed
    and all plugins and initial configuration are set up.

    This function is typically used to perform global test configuration or setup
    tasks before any tests are executed.

    Args:
        config (pytest.Config): The pytest configuration object containing command-line
            options and plugin configurations.
    """
    logger.info("=== Global Test Configuration Applied ===")


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):  # pylint: disable=unused-argument
    """Pytest hook implementation that is executed at the start of a test session.

    This function logs a message indicating that the test session has started.

    Args:
        session: The pytest session object (not used in this implementation).
    """
    logger.info("=== Test Session Started ===")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):  # pylint: disable=unused-argument
    """Hook function called after the test session ends.

    This function is executed after all tests have been run and the test session
    is about to finish. It can be used to perform cleanup or logging tasks.

    Args:
        session (Session): The pytest session object containing information
            about the test session.
        exitstatus (int): The exit status code of the test session. This indicates
            whether the tests passed, failed, or were interrupted.

    Note:
        The `pylint: disable=unused-argument` directive is used to suppress
        warnings for unused arguments in this function.
    """
    logger.info("=== Test Session Finished ===")


# REFACTOR: The sample_data_and_config fixture has been moved here from the
# test_coder_alignment.py file so it can be shared across all test files.
# It is now renamed to 'raw_data_and_config' for clarity.
@pytest.fixture
def raw_data_and_config() -> tuple[pd.DataFrame, ColumnConfig]:
    """A pytest fixture to create a standard set of RAW test data and config."""
    test_data = pd.DataFrame(
        {
            "unique_id": ["A", "B", "C", "D", "E"],
            "clerical_label_1": ["12345", "1234", "-9", "nan", "5432x"],
            "clerical_label_2": ["23456", np.nan, "4+", "", "54321"],
            "model_label_1": ["12345", "01234", "99999", "54321", "54322"],
            "model_label_2": ["99999", "12300", "54322", "88888", "54322"],
            "model_score_1": [0.9, 0.8, 0.99, 0.7, 0.85],
            "model_score_2": [0.1, 0.7, 0.98, 0.6, 0.80],
            "Unambiguous": [True, True, False, True, True],
        }
    )

    config = ColumnConfig(
        model_label_cols=["model_label_1", "model_label_2"],
        model_score_cols=["model_score_1", "model_score_2"],
        clerical_label_cols=["clerical_label_1", "clerical_label_2"],
        id_col="unique_id",
    )
    return test_data, config


@dataclass(slots=True)
class _FakeSignJwtResponse:
    """Lightweight placeholder for the google IAM sign_jwt response."""

    signed_jwt: ClassVar[str] = "ey.fake.jwt.token"


class _IamClientSpy:  # pylint: disable=too-few-public-methods
    """Spy/fake for iam_credentials_v1.IAMCredentialsClient.

    Captures initialisation credentials and last request passed to sign_jwt.
    """

    # Keep reference to last constructed instance for assertions.
    last_instance: _IamClientSpy | None = None

    def __init__(self, *, credentials: Any) -> None:
        self.credentials = credentials
        self.last_request: dict[str, Any] | None = None
        _IamClientSpy.last_instance = self

    # Signature mirrors the real client enough for our tests.
    def sign_jwt(self, *, request: dict[str, Any]) -> _FakeSignJwtResponse:  # type: ignore[override]
        self.last_request = request
        return _FakeSignJwtResponse()


@pytest.fixture()
# pylint: disable=unused-argument
def env(
    monkeypatch: MonkeyPatch,
) -> Iterator[MutableMapping[str, str]]:
    """Provide an isolated, mutable environment mapping.

    Returns:
        MutableMapping[str, str]: Backed by os.environ for the duration of the test,
        cleared on exit.
    """
    original = dict(os.environ)
    try:
        yield os.environ
    finally:
        os.environ.clear()
        os.environ.update(original)


@pytest.fixture()
def mock_gcp_adc_and_iam(monkeypatch: MonkeyPatch) -> Callable[[str], None]:
    """Patch ADC and the IAMCredentials client with deterministic fakes.

    The returned callable allows changing the fake JWT payload for specific tests.

    Args:
        monkeypatch (MonkeyPatch): Pytest monkeypatch utility.

    Returns:
        Callable[[str], None]: A setter to update the fake JWT string.
    """

    # 1) Patch google.auth.default to return (credentials, project)
    class _DummyCreds:  # pylint: disable=too-few-public-methods
        pass

    def _fake_default(*_: Any, **__: Any) -> tuple[_DummyCreds, None]:
        return _DummyCreds(), None

    monkeypatch.setattr("google.auth.default", _fake_default, raising=False)

    # 2) Patch client class to our spy
    monkeypatch.setattr(
        "google.cloud.iam_credentials_v1.IAMCredentialsClient",
        _IamClientSpy,
        raising=False,
    )

    # Closure variable that the spy reads each time.
    _current_signed_token = "ey.fake.jwt.token"  # noqa: S105

    # 3) Allow per-test override of the signed JWT value
    def _set_signed_jwt(token: str) -> None:
        monkeypatch.setattr(
            _FakeSignJwtResponse, "signed_jwt", token, raising=False  # type: ignore[attr-defined]
        )

    return _set_signed_jwt


@pytest.fixture()
def reset_iam_spy() -> None:
    """Ensure the IAM spy last_instance is reset before each test."""
    _IamClientSpy.last_instance = None


@pytest.fixture()
def freeze_time(monkeypatch: MonkeyPatch) -> Callable[[str, float], None]:
    """Patch a module's current_utc_time() to a fixed timestamp.

    Usage:
        freeze_time("app.auth.tokens", 1_700_000_000.0)

    Args:
        monkeypatch (MonkeyPatch): Pytest monkeypatch utility.

    Returns:
        Callable[[str, float], None]: Setter accepting target module path and POSIX timestamp.
    """

    def _setter(module_path: str, ts: float) -> None:
        def _fixed_now() -> dt.datetime:
            # Use fromtimestamp with tz=UTC (replacement for utcfromtimestamp)
            return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)

        monkeypatch.setattr(f"{module_path}.current_utc_time", _fixed_now, raising=True)

    return _setter


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip credentialled ADC tests when running in CI.

    This hook inspects the environment for the ``CI`` variable that GitHub Actions
    sets to ``"true"``. When present, it marks any test carrying the ``adc`` marker
    as skipped. Locally (where ``CI`` is typically unset), the tests run normally.

    Args:
        config: The pytest configuration object (unused, but required by the hook).
        items: The collected test items.

    Best practice:
        • Avoid networked or credential-dependent tests in your default CI path.
        • A marker is introduced for such tests so they can be run locally with:
          ``poetry run pytest -m adc``.
    """
    if os.getenv("CI") == "true":
        skip_adc = pytest.mark.skip(
            reason="Skipped in CI: requires GCP ADC/real credentials."
        )
        for item in items:
            if "adc" in item.keywords:
                item.add_marker(skip_adc)
