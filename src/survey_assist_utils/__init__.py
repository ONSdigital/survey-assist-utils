"""Survey Assist Utilities package."""

__all__ = ["get_logger"]

# Import at module level to satisfy pylint
from survey_assist_utils.logging.logging_utils import get_logger as _get_logger


def get_logger(name, level="INFO"):
    """Get a logger instance.

    Args:
        name: The name of the logger (typically __name__)
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        SurveyAssistLogger: A configured logger instance
    """
    return _get_logger(name, level)
