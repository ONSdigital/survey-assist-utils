"""
Tests for the logging utilities module.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from utils.logging.logging_utils import SurveyAssistLogger, get_logger, VALID_LOG_LEVELS

@pytest.fixture
def mock_gcp_logging():
    """Mock GCP logging client and logger."""
    with patch('utils.logging.logging_utils.cloud_logging') as mock:
        mock_client = MagicMock()
        mock_logger = MagicMock()
        mock.Client.return_value = mock_client
        mock_client.logger.return_value = mock_logger
        yield mock

@pytest.fixture
def local_logger():
    """Create a local logger instance."""
    return get_logger('test_module')

@pytest.fixture
def gcp_logger(mock_gcp_logging):
    """Create a GCP logger instance."""
    with patch.dict(os.environ, {'K_SERVICE': 'test-service'}):
        return get_logger('test_module')

def test_logger_initialisation():
    """Test logger initialisation with different parameters."""
    # Test with default level
    logger = get_logger('test_module')
    assert logger.name == 'test_module'
    assert logger.level == 'INFO'

    # Test with custom level
    logger = get_logger('test_module', level='DEBUG')
    assert logger.level == 'DEBUG'

    # Test with invalid level
    with pytest.raises(ValueError):
        get_logger('test_module', level='INVALID')

def test_module_name_formatting():
    """Test module name formatting and truncation."""
    # Test short name
    logger = get_logger('short_name')
    assert logger.name == 'short_name'

    # Test long name
    long_name = 'very_long_module_name_that_should_be_truncated'
    logger = get_logger(long_name)
    assert logger.name == 'very_long_modu...'

def test_local_logging(local_logger):
    """Test local logging functionality."""
    with patch('logging.Logger.info') as mock_info:
        local_logger.info('Test message')
        mock_info.assert_called_once()

    with patch('logging.Logger.error') as mock_error:
        local_logger.error('Test error', extra={'context': 'test'})
        mock_error.assert_called_once()

def test_gcp_logging(gcp_logger, mock_gcp_logging):
    """Test GCP logging functionality."""
    gcp_logger.info('Test message')
    mock_gcp_logging.Client.return_value.logger.return_value.log_text.assert_called_with(
        json.dumps({
            'message': 'Test message',
            'timestamp': pytest.ANY,
            'module': 'test_module'
        }),
        severity='INFO'
    )

def test_log_levels():
    """Test all log levels."""
    logger = get_logger('test_module')
    
    for level in VALID_LOG_LEVELS:
        log_method = getattr(logger, level.lower())
        with patch('logging.Logger.' + level.lower()) as mock_log:
            log_method(f'Test {level} message')
            mock_log.assert_called_once()

def test_message_formatting():
    """Test message formatting with additional context."""
    logger = get_logger('test_module')
    
    with patch('logging.Logger.info') as mock_info:
        logger.info('Test message', user_id=123, action='test')
        called_message = mock_info.call_args[0][0]
        assert 'Test message' in called_message
        assert 'user_id' in called_message
        assert 'action' in called_message

def test_environment_detection():
    """Test environment detection logic."""
    # Test local environment
    with patch.dict(os.environ, {}, clear=True):
        logger = get_logger('test_module')
        assert not isinstance(logger.logger, MagicMock)  # Should be a real logger

    # Test GCP environment
    with patch.dict(os.environ, {'K_SERVICE': 'test-service'}):
        with patch('utils.logging.logging_utils.cloud_logging'):
            logger = get_logger('test_module')
            assert isinstance(logger.logger, MagicMock)  # Should be a mock GCP logger 