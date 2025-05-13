"""Manual test script for logging utilities."""

import logging
import os
import time
from datetime import datetime
from io import StringIO
from unittest.mock import MagicMock, patch

from survey_assist_utils import get_logger


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

    # Print the actual log data being sent
    print("\nGCP Log Data:")
    for call in mock_logger.log_text.call_args_list:
        print(call[0][0])  # Print the first argument (the log message)

    # Verify mock was called
    assert mock_logger.log_text.called

    # Clean up
    del os.environ["K_SERVICE"]


def test_function_name_in_logs():
    """Test that logs show the correct calling function name."""
    # Set up a string buffer to capture logs
    log_buffer = StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setFormatter(
        logging.Formatter("%(message)s")
    )  # Just capture the message part

    # Get the root logger and add our handler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set the level to DEBUG to capture all logs
    root_logger.addHandler(handler)

    # Create a logger and ensure it uses the root logger's handlers
    logger = get_logger("test_module")
    logger.logger.handlers = []  # Clear existing handlers
    logger.logger.addHandler(handler)  # Add our handler

    def inner_function():
        logger.info("Test message from inner function")

    def outer_function():
        logger.info("Test message from outer function")
        inner_function()

    # Run the test
    outer_function()

    # Get the output
    output = log_buffer.getvalue()

    # Print debug info
    print("\nCaptured logs:")
    print(output)

    # Check that we see both messages in the log output with correct function names
    assert '"func": "outer_function"' in output
    assert '"func": "inner_function"' in output
    # Verify we don't see the logger's internal method name
    assert "debug" not in output

    # Clean up
    root_logger.removeHandler(handler)


if __name__ == "__main__":
    print("Testing local logging...")
    test_local_logging()
    time.sleep(1)  # Wait for logs to flush

    print("\nTesting GCP logging...")
    with patch("google.cloud.logging.Client") as mock_gcp_client:
        test_gcp_logging(mock_gcp_client)
    time.sleep(1)  # Wait for logs to flush

    print("\nTesting function names in logs...")
    test_function_name_in_logs()
    time.sleep(1)  # Wait for logs to flush
