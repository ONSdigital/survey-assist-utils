"""Logging utilities for Survey Assist applications.

This module provides a unified logging interface that works both locally and in GCP environments.
"""

import inspect
import json
import logging
import os
from datetime import datetime
from typing import Any, Union

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


class EnhancedJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that serialises additional Python objects.

    This encoder extends the standard library's JSONEncoder to handle objects
    such as datetime, ensuring they are serialised in a format suitable for JSON output.

    Attributes:
        None

    Methods:
        default(obj): Returns a serialisable version of the object for JSON encoding.
    """

    def default(self, o: object) -> object:
        """Return a serialisable version of the object for JSON encoding.

        Args:
            o (object): The object to serialise.

        Returns:
            o: A serialisable representation of the object.

        Raises:
            TypeError: If the object cannot be serialised.
        """
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


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

        # Remove any existing handlers to prevent duplicates
        logger.handlers = []

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False

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
        func_name = "unknown"
        if frame is not None:
            back_frame = frame.f_back
            if back_frame is not None:
                back_back_frame = back_frame.f_back
                if back_back_frame is not None:
                    func_name = back_back_frame.f_code.co_name

        # Abbreviate module name if message is long
        module_name = self.name
        if len(message) > MAX_MESSAGE_LENGTH and len(module_name) > MIN_MODULE_LENGTH:
            module_name = f"{module_name[:MODULE_NAME_TRUNCATE_LENGTH]}..."

        context = {
            "message": message,
            "module": module_name,
            "func": func_name,
        }
        context.update(kwargs)

        try:
            # If JSON_DEBUG is set, pretty print the context
            return json.dumps(
                context,
                cls=EnhancedJSONEncoder,
                indent=2 if os.getenv("JSON_DEBUG") else None,
            )
        except TypeError as e:
            context["serialization_error"] = str(e)
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

    def _get_caller_info(self) -> dict[str, str]:
        """Get information about the calling function.

        Returns:
            dict[str, str]: Dictionary containing caller information.
        """
        frame = inspect.currentframe()
        if frame is None:
            return {"module": "unknown", "func": "unknown"}

        back_frame = frame.f_back
        if back_frame is None:
            return {"module": "unknown", "func": "unknown"}

        back_back_frame = back_frame.f_back
        if back_back_frame is None:
            return {"module": "unknown", "func": "unknown"}

        module_name = back_back_frame.f_globals.get("__name__", "unknown")
        function_name = back_back_frame.f_code.co_name
        return {"module": module_name, "func": function_name}


def get_logger(name: str, level: str = DEFAULT_LOG_LEVEL) -> SurveyAssistLogger:
    """Get a configured logger instance.

    Args:
        name: The name of the logger (typically __name__)
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        SurveyAssistLogger: A configured logger instance
    """
    return SurveyAssistLogger(name, level)
