"""Unit tests for the batch processing script functions."""

import unittest
import json
from unittest.mock import patch, mock_open, MagicMock
import io # For StringIO

import pandas as pd
import toml
import requests # Import requests for exception types


from scripts.batch import (
    load_config,
    read_sic_data,
    process_row,
    process_test_set
)

# Sample data for tests
MINIMAL_TOML_CONFIG = """
[paths]
gold_standard_csv = "dummy/gold.csv"
output_filepath = "dummy/output.jsonl"

[parameters]
test_num = 1
test_mode = true
"""

MINIMAL_CSV_DATA = """unique_id,soc2020_job_title,soc2020_job_description,sic2007_employee
id1,Test Job,Test Desc,Test Industry
"""

class TestBatchProcessingMinimal(unittest.TestCase):
    """Minimal tests for batch processing script functions."""

    def test_load_config_runs(self):
        """Test that load_config runs and parses minimal TOML."""
        with patch("builtins.open", mock_open(read_data=MINIMAL_TOML_CONFIG)) as _mock_file:
            config = load_config("dummy_config.toml")
            self.assertIsNotNone(config)
            self.assertIn("paths", config)
            self.assertEqual(config["paths"]["gold_standard_csv"], "dummy/gold.csv")

    @patch("pandas.read_csv")
    def test_read_sic_data_runs(self, mock_pd_read_csv):
        """Test that read_sic_data runs and calls pd.read_csv."""
        # Mock pd.read_csv to return an empty DataFrame for simplicity
        mock_df = pd.DataFrame({'unique_id': ['1']})
        mock_pd_read_csv.return_value = mock_df

        df = read_sic_data("dummy_path.csv")

        mock_pd_read_csv.assert_called_once()
        self.assertIsNotNone(df)
        # Basic check on returned DataFrame
        pd.testing.assert_frame_equal(df, mock_df)


    @patch("requests.post")
    def test_process_row_runs(self, mock_post):
        """Test that process_row runs and makes a mock API call."""
        mock_api_response = MagicMock()
        mock_api_response.json.return_value = {"classified": True, "sic_code": "00000"}
        mock_api_response.raise_for_status.return_value = None # Simulate no HTTP error
        mock_post.return_value = mock_api_response

        sample_row_data = pd.Series({
            "unique_id": "test_id_pr",
            "soc2020_job_title": "Sample Job",
            "soc2020_job_description": "Sample Description",
            "sic2007_employee": "Sample Industry"
        })
        secret = "dummy_secret"

        result = process_row(sample_row_data, secret)

        mock_post.assert_called_once()
        self.assertIsNotNone(result)
        self.assertTrue(result["classified"])
        self.assertEqual(result["unique_id"], "test_id_pr")

    @patch("scripts.batch.pd.read_csv")
    @patch("scripts.batch.process_row") # Mock the process_row function
    @patch("builtins.open", new_callable=mock_open) # Mock file opening
    @patch("scripts.batch.time.sleep") # Mock time.sleep
    def test_process_test_set_runs(self, mock_sleep, mock_file_open, mock_proc_row, mock_pd_read_csv):
        """Test that process_test_set runs with minimal data."""
        # Prepare a minimal DataFrame to be returned by the mocked pd.read_csv
        # This DataFrame should have the columns that process_row expects
        header = ["unique_id", "soc2020_job_title", "soc2020_job_description", "sic2007_employee"]
        data_rows = [["id1", "Job1", "Desc1", "Industry1"]]
        minimal_df = pd.DataFrame(data_rows, columns=header)
        mock_pd_read_csv.return_value = minimal_df

        # Define what the mocked process_row should return
        mock_proc_row.return_value = {"unique_id": "id1", "processed_data": "some_result"}

        process_test_set(
            secret_code="dummy_secret",
            csv_filepath="dummy_input.csv",
            output_filepath="dummy_output.jsonl",
            test_mode=True, # Run in test mode
            test_limit=1    # Process only one row
        )

        # Assert that the core functions were called
        mock_pd_read_csv.assert_called_once_with("dummy_input.csv", delimiter=",", dtype=str)
        mock_proc_row.assert_called_once() # Called once due to test_limit=1
        mock_file_open.assert_called_once_with("dummy_output.jsonl", "a", encoding="utf-8")
        mock_file_open().write.assert_called_once() # Write called once
        mock_sleep.assert_called_once() # Sleep called once

if __name__ == "__main__":
    unittest.main()