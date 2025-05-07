"""
Survey Assist Utils Logging Package

This package provides a unified logging interface for Survey Assist applications,
supporting both local development and GCP environments.
"""

from .logging_utils import get_logger

__all__ = ['get_logger'] 