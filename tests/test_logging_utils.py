"""Tests for the logging utilities module."""

# pylint: disable=redefined-outer-name,unused-argument

from unittest.mock import MagicMock, patch
from io import StringIO
import json
import logging

import pytest

from survey_assist_utils.logging.logging_utils import (
    get_logger,
)


@pytest.fixture
def mock_gcp_logging():
    """Mock GCP logging module."""
    with patch("google.cloud.logging.Client") as mock_client:
        mock_logger = MagicMock()
        mock_client.return_value.logger.return_value = mock_logger
        yield mock_logger


@pytest.fixture
def mock_google_auth():
    """Mock Google auth."""
    with patch("google.auth.default") as mock_auth:
        mock_credentials = MagicMock()
        mock_auth.return_value = (mock_credentials, "test-project")
        yield mock_auth


@pytest.fixture
def local_logger():
    """Create a local logger instance."""
    return get_logger("test_module")


@pytest.fixture
def gcp_logger(mock_gcp_logging):
    """Create a GCP logger instance."""
    with patch("os.environ.get", return_value="test-service"):
        return get_logger("test_module")


def test_logger_initialisation():
    """Test logger initialization."""
    logger = get_logger("test_module")
    assert logger.name == "test_module"
    assert logger.level == "INFO"


def test_module_name_formatting():
    """Test module name formatting and truncation."""
    # Test short name
    logger = get_logger("short_name")
    assert logger.name == "short_name"

    # Test long name
    long_name = "very_long_module_name_that_should_be_truncated"
    logger = get_logger(long_name)
    assert logger.name == "very_long_modul..."


def test_local_logging(local_logger):
    """Test local logging functionality."""
    with patch("logging.Logger.info") as mock_info:
        local_logger.info("Test message")
        mock_info.assert_called_once()

    with patch("logging.Logger.error") as mock_error:
        local_logger.error("Test error", extra={"context": "test"})
        mock_error.assert_called_once()


def test_gcp_logging(mock_gcp_logging, mock_google_auth):
    """Test GCP logging functionality."""
    with patch("os.environ.get", return_value="test-service"):
        logger = get_logger("test_module")
        logger.info("Test GCP message")
        assert mock_gcp_logging.log_text.called


def test_log_levels():
    """Test different log levels."""
    logger = get_logger("test_module", level="DEBUG")
    assert logger.level == "DEBUG"


def test_message_formatting():
    """Test message formatting with context."""
    logger = get_logger("test_module")
    logger.info("Test message", user_id="123", action="test")


def test_environment_detection():
    """Test environment detection logic."""
    # Test local environment
    with patch("os.environ.get", return_value=None):
        logger = get_logger("test_module")
        assert not hasattr(logger.logger, "log_text")  # Should be a local logger

    # Test GCP environment
    with patch("os.environ.get", return_value="test-service"), patch(
        "google.auth.default"
    ) as mock_auth, patch("google.cloud.logging.Client") as mock_client:
        mock_credentials = MagicMock()
        mock_auth.return_value = (mock_credentials, "test-project")
        mock_logger = MagicMock()
        mock_client.return_value.logger.return_value = mock_logger

        logger = get_logger("test_module")
        assert hasattr(logger.logger, "log_text")  # Should be a GCP logger


def test_log_format_no_duplicate_severity():
    """Test that log messages don't contain duplicate severity levels."""
    # Set up a string buffer to capture logs
    log_buffer = StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setFormatter(logging.Formatter("%(message)s"))
    
    # Get the root logger and add our handler
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    
    # Create a logger and log messages at different levels
    logger = get_logger("test_module")
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    
    # Get the output
    output = log_buffer.getvalue()
    
    # Verify that the severity level only appears once in each log line
    for line in output.strip().split('\n'):
        if not line.strip():
            continue
        # Extract the JSON part (after the last ' - ')
        if ' - ' in line:
            json_part = line.split(' - ', 3)[-1]
        else:
            json_part = line
        log_data = json.loads(json_part)
        # Verify that severity is not in log_data
        assert "severity" not in log_data
        # Verify that the message doesn't contain the severity level
        assert not any(level in log_data["message"].lower() for level in ["info", "warning", "error", "critical", "debug"])
    
    # Clean up
    root_logger.removeHandler(handler)
