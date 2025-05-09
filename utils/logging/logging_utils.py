"""Logging utilities for Survey Assist applications.

This module provides a unified logging interface that works both locally and in GCP environments.
"""

import inspect
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any, Union, Dict

# Import cloud logging at module level
GCP_LOGGING_AVAILABLE = False
gcp_logging: Any = None

try:
    from google.cloud import logging as _gcp_logging  # type: ignore

    gcp_logging = _gcp_logging
    GCP_LOGGING_AVAILABLE = True
except ImportError:
    pass

# Constants
MAX_MODULE_NAME_LENGTH = 20
DEFAULT_LOG_LEVEL = "INFO"
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
MAX_MESSAGE_LENGTH = 100
MIN_MODULE_LENGTH = 10
MODULE_NAME_TRUNCATE_LENGTH = 15


def _get_cloud_logging():
    """Get the cloud logging module if available."""
    return gcp_logging


class SurveyAssistLogger:
    """A custom logger class that handles both local and GCP logging."""

    def __init__(self, name: str, level: str = DEFAULT_LOG_LEVEL):
        """Initialise the logger.

        Args:
            name: The name of the logger (typically __name__)
            level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.name = self._format_module_name(name)
        self.level = self._validate_log_level(level)
        self.logger = self._setup_logger()

    def _format_module_name(self, name: str) -> str:
        """Format the module name, truncating if necessary.

        Args:
            name: The original module name

        Returns:
            str: The formatted module name
        """
        if len(name) > MAX_MODULE_NAME_LENGTH:
            return f"{name[:15]}..."
        return name

    def _validate_log_level(self, level: str) -> str:
        """Validate the log level.

        Args:
            level: The proposed log level

        Returns:
            str: The validated log level

        Raises:
            ValueError: If the log level is invalid
        """
        level = level.upper()
        if level not in VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log level: {level}. Must be one of {VALID_LOG_LEVELS}"
            )
        return level

    def _setup_logger(self) -> Union[logging.Logger, Any]:
        """Set up the appropriate logger based on the environment.

        Returns:
            Union[logging.Logger, Any]: The configured logger
        """
        if os.environ.get("K_SERVICE") and GCP_LOGGING_AVAILABLE:
            return self._setup_gcp_logger()
        return self._setup_local_logger()

    def _setup_local_logger(self) -> logging.Logger:
        """Set up a local logger with console output.

        Returns:
            logging.Logger: The configured local logger
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _setup_gcp_logger(self) -> Any:
        """Set up a GCP logger.

        Returns:
            Any: The configured GCP logger
        """
        client = gcp_logging.Client()
        logger = client.logger(self.name)
        return logger

    def _format_message(self, message: str, **kwargs) -> str:
        """Format the log message with additional context.

        Args:
            message: The log message
            **kwargs: Additional context to include in the log

        Returns:
            str: The formatted message
        """
        # Get the calling function's name by going up two frames
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back.f_back
            func_name = caller_frame.f_code.co_name if caller_frame else "unknown"
        finally:
            del frame

        # Abbreviate module name if message is long
        module_name = self.name
        if len(message) > MAX_MESSAGE_LENGTH and len(module_name) > MIN_MODULE_LENGTH:
            module_name = f"{module_name[:MODULE_NAME_TRUNCATE_LENGTH]}..."

        context = {
            "message": message,
            "module": module_name,
            "func": func_name,
            **kwargs,
        }
        return json.dumps(context)

    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        formatted_message = self._format_message(message, **kwargs)
        if hasattr(self.logger, "log_text"):  # GCP logger
            self.logger.log_text(formatted_message, severity="DEBUG")
        else:
            self.logger.debug(formatted_message)

    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        formatted_message = self._format_message(message, **kwargs)
        if hasattr(self.logger, "log_text"):  # GCP logger
            self.logger.log_text(formatted_message, severity="INFO")
        else:
            self.logger.info(formatted_message)

    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        formatted_message = self._format_message(message, **kwargs)
        if hasattr(self.logger, "log_text"):  # GCP logger
            self.logger.log_text(formatted_message, severity="WARNING")
        else:
            self.logger.warning(formatted_message)

    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        formatted_message = self._format_message(message, **kwargs)
        if hasattr(self.logger, "log_text"):  # GCP logger
            self.logger.log_text(formatted_message, severity="ERROR")
        else:
            self.logger.error(formatted_message)

    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message."""
        formatted_message = self._format_message(message, **kwargs)
        if hasattr(self.logger, "log_text"):  # GCP logger
            self.logger.log_text(formatted_message, severity="CRITICAL")
        else:
            self.logger.critical(formatted_message)

    def _get_caller_info(self) -> Dict[str, str]:
        """Get information about the calling function.

        Returns:
            Dict[str, str]: Dictionary containing module and function information
        """
        frame = inspect.currentframe()
        try:
            # Go up two frames to get the actual caller
            caller_frame = frame.f_back.f_back
            module_name = caller_frame.f_globals.get("__name__", "unknown")
            function_name = caller_frame.f_code.co_name
            return {
                "module": module_name,
                "func": function_name
            }
        finally:
            del frame


def get_logger(name: str, level: str = DEFAULT_LOG_LEVEL) -> SurveyAssistLogger:
    """Get a configured logger instance.

    Args:
        name: The name of the logger (typically __name__)
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        SurveyAssistLogger: A configured logger instance
    """
    return SurveyAssistLogger(name, level)
