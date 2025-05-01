import logging
import sys
import tempfile  # <--- Import tempfile if not already present
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import tomllib  # Or import toml for older Python

from src.data_loader import AppConfig, GoldStandardLoader, setup_logging

# --- Sample Data for Tests --- (Keep as before)
SAMPLE_CONFIG_TOML = """
[paths]
gold_standard_csv = "test_data/gold.csv"
output_dir = "test_output"

[column_names]
unique_id = "UID"
gold_sic = "GOLD_SIC"
gold_flag = "GOLD_FLAG"

[logging]
level = "DEBUG"
file = "logs/test_eval.log"
format = "%(levelname)s:%(name)s:%(message)s"

[other_section]
param = 123
"""

# --- Test Cases ---


class TestAppConfig(unittest.TestCase):
    # ... (Keep tests for AppConfig as before) ...
    pass


class TestSetupLogging(unittest.TestCase):
    """Tests for the setup_logging function."""

    def setUp(self):
        """Create a temporary directory for tests that might need file I/O."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

    def tearDown(self):
        """Clean up the temporary directory."""
        self.test_dir.cleanup()

    @patch("logging.basicConfig")
    @patch("logging.FileHandler")
    @patch("logging.StreamHandler")
    @patch("os.makedirs")  # Mock os.makedirs
    def test_setup_logging_with_file(
        self, mock_makedirs, mock_stream_handler, mock_file_handler, mock_basic_config
    ):
        """Test logging setup with file output specified."""
        # --- FIX: Use temporary directory path ---
        # Create a mock AppConfig instance using the temporary path
        mock_config_data = tomllib.loads(SAMPLE_CONFIG_TOML)
        # Use self.test_path which is a valid temporary directory path
        mock_app_config = AppConfig(
            config_data=mock_config_data, project_root=self.test_path
        )
        # --- End Fix ---

        # Mock handler instances
        mock_fh_instance = MagicMock()
        mock_sh_instance = MagicMock()
        mock_file_handler.return_value = mock_fh_instance
        mock_stream_handler.return_value = mock_sh_instance

        setup_logging(mock_app_config)

        # Check directory creation using os.makedirs mock
        expected_log_path = (self.test_path / "logs/test_eval.log").resolve()

        # Check handlers created (remains the same logic)
        mock_file_handler.assert_called_once_with(expected_log_path, encoding="utf-8")
        mock_stream_handler.assert_called_once_with(sys.stdout)

        # Check basicConfig call (remains the same logic)
        mock_basic_config.assert_called_once()
        call_args, call_kwargs = mock_basic_config.call_args
        self.assertEqual(call_kwargs.get("level"), logging.DEBUG)
        self.assertEqual(
            call_kwargs.get("format"), "%(levelname)s:%(name)s:%(message)s"
        )
        self.assertListEqual(
            call_kwargs.get("handlers"), [mock_fh_instance, mock_sh_instance]
        )
        self.assertTrue(call_kwargs.get("force"))

    @patch("logging.basicConfig")
    @patch("logging.FileHandler")
    @patch("logging.StreamHandler")
    @patch("os.makedirs")
    def test_setup_logging_console_only(
        self, mock_makedirs, mock_stream_handler, mock_file_handler, mock_basic_config
    ):
        """Test logging setup with only console output."""
        # Create config without file path
        config_no_file = SAMPLE_CONFIG_TOML.replace(
            'file = "logs/test_eval.log"', '# file = "logs/test_eval.log"'
        )
        mock_config_data = tomllib.loads(config_no_file)
        # --- FIX: Use temporary directory path ---
        mock_app_config = AppConfig(
            config_data=mock_config_data, project_root=self.test_path
        )
        # --- End Fix ---

        mock_sh_instance = MagicMock()
        mock_stream_handler.return_value = mock_sh_instance

        setup_logging(mock_app_config)

        mock_makedirs.assert_not_called()
        mock_file_handler.assert_not_called()
        mock_stream_handler.assert_called_once_with(sys.stdout)

        mock_basic_config.assert_called_once()
        call_args, call_kwargs = mock_basic_config.call_args
        self.assertEqual(call_kwargs.get("level"), logging.DEBUG)
        self.assertListEqual(call_kwargs.get("handlers"), [mock_sh_instance])

    @patch("logging.basicConfig")
    @patch("logging.StreamHandler")
    @patch("os.makedirs")
    def test_setup_logging_no_config(
        self, mock_makedirs, mock_stream_handler, mock_basic_config
    ):
        """Test logging setup with default settings (no config passed)."""
        mock_sh_instance = MagicMock()
        mock_stream_handler.return_value = mock_sh_instance

        setup_logging(config=None)

        mock_makedirs.assert_not_called()
        mock_stream_handler.assert_called_once_with(sys.stdout)
        mock_basic_config.assert_called_once()
        call_args, call_kwargs = mock_basic_config.call_args
        self.assertEqual(call_kwargs.get("level"), logging.INFO)
        self.assertEqual(
            call_kwargs.get("format"),
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )
        self.assertListEqual(call_kwargs.get("handlers"), [mock_sh_instance])


class TestGoldStandardLoader(unittest.TestCase):
    # --- Add setUp/tearDown if not already present for consistency ---
    def setUp(self):
        """Create a temporary directory for file-based tests."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)
        # Create dummy config and loader instance for tests
        self.mock_config_data = tomllib.loads(SAMPLE_CONFIG_TOML)
        # Point gold standard path to temp dir for tests needing real files
        self.mock_config_data["paths"]["gold_standard_csv"] = str(
            self.test_path / "gold.csv"
        )
        self.mock_app_config = AppConfig(
            config_data=self.mock_config_data, project_root=self.test_path
        )
        self.loader = GoldStandardLoader(config=self.mock_app_config)

    def tearDown(self):
        """Clean up the temporary directory."""
        self.test_dir.cleanup()

    # --- End setUp/tearDown ---

    # ... (Keep tests for GoldStandardLoader as before, they already use tempfile) ...
    pass


if __name__ == "__main__":
    # Ensure logging doesn't interfere with test output capture
    logging.basicConfig(level=logging.CRITICAL)  # Set to high level during tests
    unittest.main()
