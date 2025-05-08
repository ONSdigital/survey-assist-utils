"""Manual test script for logging utilities."""

import os
import time
from datetime import datetime

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


def test_gcp_logging():
    """Test GCP logging functionality."""
    # Set environment variable to simulate GCP environment
    os.environ["K_SERVICE"] = "test-service"

    # Create a logger
    logger = get_logger(__name__)

    # Test different log levels
    logger.debug("This is a debug message in GCP")
    logger.info("This is an info message in GCP")
    logger.warning("This is a warning message in GCP")
    logger.error("This is an error message in GCP")
    logger.critical("This is a critical message in GCP")

    # Test logging with context
    logger.info(
        "GCP message with context",
        user_id="456",
        action="test",
        timestamp=datetime.now().isoformat(),
    )


if __name__ == "__main__":
    print("Testing local logging...")
    test_local_logging()
    time.sleep(1)  # Wait for logs to flush

    print("\nTesting GCP logging...")
    test_gcp_logging()
    time.sleep(1)  # Wait for logs to flush
