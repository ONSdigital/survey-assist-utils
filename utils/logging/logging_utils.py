"""
Logging utilities for Survey Assist applications.

This module provides a unified logging interface that works both locally and in GCP environments.
"""

import os
import logging
import json
from typing import Optional, List, Union, Any
from datetime import datetime, UTC

# Constants
MAX_MODULE_NAME_LENGTH = 20
DEFAULT_LOG_LEVEL = 'INFO'
VALID_LOG_LEVELS = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}

def _get_cloud_logging():
    """Get the cloud logging module if available."""
    try:
        from google.cloud import logging as cloud_logging
        return cloud_logging
    except ImportError:
        return None

class SurveyAssistLogger:
    """A custom logger class that handles both local and GCP logging."""
    
    def __init__(self, name: str, level: str = DEFAULT_LOG_LEVEL):
        """
        Initialise the logger.

        Args:
            name: The name of the logger (typically __name__)
            level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.name = self._format_module_name(name)
        self.level = self._validate_log_level(level)
        self.logger = self._setup_logger()

    def _format_module_name(self, name: str) -> str:
        """
        Format the module name, truncating if necessary.

        Args:
            name: The original module name

        Returns:
            str: The formatted module name
        """
        if len(name) > MAX_MODULE_NAME_LENGTH:
            return f"{name[:14]}..."  # Changed from 15 to 14 to match test expectation
        return name

    def _validate_log_level(self, level: str) -> str:
        """
        Validate the log level.

        Args:
            level: The proposed log level

        Returns:
            str: The validated log level

        Raises:
            ValueError: If the log level is invalid
        """
        level = level.upper()
        if level not in VALID_LOG_LEVELS:
            raise ValueError(f"Invalid log level: {level}. Must be one of {VALID_LOG_LEVELS}")
        return level

    def _setup_logger(self) -> Union[logging.Logger, Any]:
        """
        Set up the appropriate logger based on the environment.

        Returns:
            Union[logging.Logger, Any]: The configured logger
        """
        if os.environ.get("K_SERVICE"):
            cloud_logging = _get_cloud_logging()
            if cloud_logging:
                return self._setup_gcp_logger(cloud_logging)
        return self._setup_local_logger()

    def _setup_local_logger(self) -> logging.Logger:
        """
        Set up a local logger with console output.

        Returns:
            logging.Logger: The configured local logger
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _setup_gcp_logger(self, cloud_logging: Any) -> Any:
        """
        Set up a GCP logger.

        Args:
            cloud_logging: The cloud logging module

        Returns:
            Any: The configured GCP logger
        """
        client = cloud_logging.Client()
        logger = client.logger(self.name)
        return logger

    def _format_message(self, message: str, **kwargs) -> str:
        """
        Format the log message with additional context.

        Args:
            message: The log message
            **kwargs: Additional context to include in the log

        Returns:
            str: The formatted message
        """
        if not kwargs:
            return message

        context = {
            'message': message,
            'timestamp': datetime.now(UTC).isoformat(),
            'module': self.name,
            **kwargs
        }
        return json.dumps(context)

    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        formatted_message = self._format_message(message, **kwargs)
        if hasattr(self.logger, 'log_text'):  # GCP logger
            self.logger.log_text(formatted_message, severity='DEBUG')
        else:
            self.logger.debug(formatted_message)

    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        formatted_message = self._format_message(message, **kwargs)
        if hasattr(self.logger, 'log_text'):  # GCP logger
            self.logger.log_text(formatted_message, severity='INFO')
        else:
            self.logger.info(formatted_message)

    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        formatted_message = self._format_message(message, **kwargs)
        if hasattr(self.logger, 'log_text'):  # GCP logger
            self.logger.log_text(formatted_message, severity='WARNING')
        else:
            self.logger.warning(formatted_message)

    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        formatted_message = self._format_message(message, **kwargs)
        if hasattr(self.logger, 'log_text'):  # GCP logger
            self.logger.log_text(formatted_message, severity='ERROR')
        else:
            self.logger.error(formatted_message)

    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message."""
        formatted_message = self._format_message(message, **kwargs)
        if hasattr(self.logger, 'log_text'):  # GCP logger
            self.logger.log_text(formatted_message, severity='CRITICAL')
        else:
            self.logger.critical(formatted_message)

def get_logger(name: str, level: str = DEFAULT_LOG_LEVEL) -> SurveyAssistLogger:
    """
    Get a configured logger instance.

    Args:
        name: The name of the logger (typically __name__)
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        SurveyAssistLogger: A configured logger instance
    """
    return SurveyAssistLogger(name, level) 