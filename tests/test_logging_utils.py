"""Tests for the logging utilities module."""

# pylint: disable=redefined-outer-name,unused-argument

import json
from unittest.mock import MagicMock, patch
import os

import pytest

from utils.logging.logging_utils import (
    VALID_LOG_LEVELS,
    get_logger,
)


@pytest.fixture
def mock_gcp_logging():
    """Mock GCP logging module."""
    with patch('google.cloud.logging.Client') as mock_client:
        mock_logger = MagicMock()
        mock_client.return_value.logger.return_value = mock_logger
        yield mock_logger


@pytest.fixture
def mock_google_auth():
    """Mock Google auth."""
    with patch('google.auth.default') as mock_auth:
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
    with patch("os.environ.get", return_value="test-service"), \
         patch('google.auth.default') as mock_auth, \
         patch('google.cloud.logging.Client') as mock_client:
        mock_credentials = MagicMock()
        mock_auth.return_value = (mock_credentials, "test-project")
        mock_logger = MagicMock()
        mock_client.return_value.logger.return_value = mock_logger
        
        logger = get_logger("test_module")
        assert hasattr(logger.logger, "log_text")  # Should be a GCP logger
