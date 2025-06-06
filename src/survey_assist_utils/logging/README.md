# Survey Assist Logging Package

A unified logging interface for Survey Assist applications that works seamlessly in both local development and Google Cloud Platform (GCP) environments.

## Features

- Automatic environment detection (local vs GCP)
- Consistent logging format across environments
- Structured JSON logging for GCP
- Configurable log levels
- Module name truncation for cleaner logs
- Additional context support for all log messages
- Graceful fallback to local logging when GCP dependencies are unavailable

## Installation

The package is included in the `survey-assist-utils` package. To use it, ensure you have the following dependencies:

```toml
[tool.poetry.dependencies]
google-cloud-logging = "^3.9.0"
```

## Usage

### Basic Usage

```python
from survey_assist_utils.logging import get_logger

# Create a logger instance
logger = get_logger(__name__)

# Log messages
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
```

### Adding Context to Logs

```python
logger.info("User action completed", user_id=123, action="login")
```

### Custom Log Levels

```python
logger = get_logger(__name__, level='DEBUG')
```

## Environment Detection

The logger automatically detects the environment:
- In GCP Cloud Run: Uses `google-cloud-logging` when `K_SERVICE` environment variable is set
- Local: Falls back to standard Python logging with console output
- Graceful fallback: If GCP dependencies are unavailable, automatically uses local logging

## Log Format

### Local Development
```
2024-03-14 10:30:45,123 - INFO - module_name - func_name - Log message
```

### GCP Environment
```json
{
    "message": "Log message",
    "timestamp": "2024-03-14T10:30:45.123Z",
    "module": "module_name",
    "func": "func_name",
    "user_id": 123,
    "action": "login"
}
```

## Best Practices

1. Always use `__name__` as the logger name to maintain proper module hierarchy
2. Include relevant context in log messages using keyword arguments
3. Use appropriate log levels:
   - DEBUG: Detailed information for debugging
   - INFO: General operational messages
   - WARNING: Unexpected but handled situations
   - ERROR: Errors that need attention
   - CRITICAL: System-level errors
4. The logger will automatically handle environment detection and logging format

## Contributing

Please follow the project's coding standards and include tests for any new features. 