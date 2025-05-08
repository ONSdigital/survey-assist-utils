"""
Manual test script for the logging package.
"""

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
    logger.info("User action completed", user_id=123, action="login")
    
    # Test long module name
    long_name_logger = get_logger("very_long_module_name_that_should_be_truncated")
    long_name_logger.info("Testing long module name")

def test_gcp_logging():
    """Test GCP logging functionality."""
    # Simulate GCP environment
    import os
    os.environ["K_SERVICE"] = "test-service"
    
    # Create a logger
    logger = get_logger(__name__)
    
    # Test logging with context
    logger.info("GCP test message", service="test-service", environment="test")

if __name__ == "__main__":
    print("Testing local logging:")
    test_local_logging()
    
    print("\nTesting GCP logging:")
    test_gcp_logging() 