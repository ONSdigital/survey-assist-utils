"""
Logging utilities for Survey Assist applications.

This module provides a unified logging interface that works both locally and in GCP environments.
"""

import os
import logging
import json
from typing import Optional, List, Union
from datetime import datetime

try:
    from google.cloud import logging as cloud_logging
    GCP_LOGGING_AVAILABLE = True
except ImportError:
    GCP_LOGGING_AVAILABLE = False

# Constants
MAX_MODULE_NAME_LENGTH = 20
DEFAULT_LOG_LEVEL = 'INFO'
VALID_LOG_LEVELS = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}

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
            return f"{name[:15]}..."
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

    def _setup_logger(self) -> Union[logging.Logger, cloud_logging.Logger]:
        """
        Set up the appropriate logger based on the environment.

        Returns:
            Union[logging.Logger, cloud_logging.Logger]: The configured logger
        """
        if os.environ.get("K_SERVICE") and GCP_LOGGING_AVAILABLE:
            return self._setup_gcp_logger()
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

    def _setup_gcp_logger(self) -> cloud_logging.Logger:
        """
        Set up a GCP logger.

        Returns:
            cloud_logging.Logger: The configured GCP logger
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
            'timestamp': datetime.utcnow().isoformat(),
            'module': self.name,
            **kwargs
        }
        return json.dumps(context)

    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        formatted_message = self._format_message(message, **kwargs)
        if isinstance(self.logger, cloud_logging.Logger):
            self.logger.log_text(formatted_message, severity='DEBUG')
        else:
            self.logger.debug(formatted_message)

    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        formatted_message = self._format_message(message, **kwargs)
        if isinstance(self.logger, cloud_logging.Logger):
            self.logger.log_text(formatted_message, severity='INFO')
        else:
            self.logger.info(formatted_message)

    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        formatted_message = self._format_message(message, **kwargs)
        if isinstance(self.logger, cloud_logging.Logger):
            self.logger.log_text(formatted_message, severity='WARNING')
        else:
            self.logger.warning(formatted_message)

    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        formatted_message = self._format_message(message, **kwargs)
        if isinstance(self.logger, cloud_logging.Logger):
            self.logger.log_text(formatted_message, severity='ERROR')
        else:
            self.logger.error(formatted_message)

    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message."""
        formatted_message = self._format_message(message, **kwargs)
        if isinstance(self.logger, cloud_logging.Logger):
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