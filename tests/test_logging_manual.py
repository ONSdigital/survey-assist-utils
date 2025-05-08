"""Manual test script for logging utilities."""

import os
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

from utils.logging import get_logger


def test_local_logging():
    """Test local logging functionality."""
    # Create a logger
    logger = get_logger(__name__)

    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Test logging with context
    logger.info(
        "Message with context",
        user_id="123",
        action="test",
        timestamp=datetime.now().isoformat(),
    )

    # Test long module name
    long_name_logger = get_logger("very_long_module_name_that_should_be_truncated")
    long_name_logger.info("Testing long module name")


@patch("google.cloud.logging.Client")
def test_gcp_logging(mock_client):
    """Test GCP logging functionality."""
    # Set up mock
    mock_logger = MagicMock()
    mock_client.return_value.logger.return_value = mock_logger

    # Set environment variable to simulate GCP environment
    os.environ["K_SERVICE"] = "test-service"

    # Create a logger
    logger = get_logger(__name__)

    # Test logging
    logger.info("Test GCP info message")
    logger.warning("Test GCP warning message")
    logger.error("Test GCP error message")

    # Verify mock was called
    assert mock_logger.log_text.called

    # Clean up
    del os.environ["K_SERVICE"]


if __name__ == "__main__":
    print("Testing local logging...")
    test_local_logging()
    time.sleep(1)  # Wait for logs to flush

    print("\nTesting GCP logging...")
    with patch("google.cloud.logging.Client") as gcp_mock:
        test_gcp_logging(gcp_mock)
    time.sleep(1)  # Wait for logs to flush
