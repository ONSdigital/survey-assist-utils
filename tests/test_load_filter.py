"""Unit tests for the data loading and configuration functions."""  # C0114 Fix

import unittest
from unittest.mock import mock_open, patch

import numpy as np  # Import numpy for NaN comparison if needed
import pandas as pd
import toml
from pandas.testing import (  # For comparing results
    assert_frame_equal,
    assert_series_equal,
)

from src.config_loader import load_config

# Assuming src/load_filter.py exists and is importable
from src.load_filter import add_data_quality_flags, read_sic_data


class TestLoadConfig(unittest.TestCase):  # Renamed for clarity
    """Tests the configuration loading function."""  # C0115 Fix

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='[settings]\napi_gateway = "https://api.example.com"\n',
    )
    def test_load_config_valid(self, _mock_file):  # W0613 Fix: Renamed mock_file
        """Test loading a valid TOML configuration file."""  # C0116 Fix
        config = load_config("config.toml")
        self.assertEqual(config["settings"]["api_gateway"], "https://api.example.com")

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_config_missing_file(self, _mock_file):  # W0613 Fix: Renamed mock_file
        """Test FileNotFoundError when the config file is missing."""  # C0116 Fix
        with self.assertRaises(FileNotFoundError):
            load_config("config.toml")

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='[settings]\napi_gateway = "https://api.example.com"\ninvalid_toml',
    )
    def test_load_config_malformed_data(
        self, _mock_file
    ):  # W0613 Fix: Renamed mock_file
        """Test TomlDecodeError for malformed TOML data."""  # C0116 Fix
        # Use the actual exception type based on your import (toml or tomllib)
        toml_exception = getattr(
            toml, "TomlDecodeError", AttributeError
        )  # Handle potential differences
        if (
            toml_exception is AttributeError
        ):  # Fallback if using tomllib maybe? Adjust as needed
            toml_exception = ValueError  # Or the specific exception tomllib raises

        with self.assertRaises(toml_exception):
            load_config("config.toml")


class TestReadSicData(unittest.TestCase):  # Renamed for clarity
    """Tests the SIC data reading function."""

    @patch("pandas.read_csv")
    def test_read_sic_data_valid(self, mock_read_csv):
        """Test reading valid SIC data using a mocked DataFrame."""  # C0116 Fix
        mock_df = pd.DataFrame(
            {
                "unique_id": ["1"],
                "sic_section": ["A"],
                "sic2007_employee": ["10"],
                "sic2007_self_employed": ["5"],
                "sic_ind1": ["B"],
                "sic_ind2": ["C"],
                "sic_ind3": ["D"],
                "sic_ind_code_flag": ["E"],
                "soc2020_job_title": ["F"],
                "soc2020_job_description": ["G"],
                "sic_ind_occ1": ["H"],
                "sic_ind_occ2": ["I"],
                "sic_ind_occ3": ["J"],
                "sic_ind_occ_flag": ["K"],
            }
        )
        mock_read_csv.return_value = mock_df
        # Pass dummy config or None if defaults are okay for column names
        df = read_sic_data("sic_data.csv", loaded_config=None)
        mock_read_csv.assert_called_once()
        # Check some arguments passed to read_csv
        call_kwargs = mock_read_csv.call_args
        self.assertEqual(call_kwargs.get("delimiter"), ",")  # Check delimiter used
        self.assertEqual(call_kwargs.get("dtype"), str)
        self.assertEqual(call_kwargs.get("na_filter"), False)
        # Compare the returned DataFrame with the mock
        pd.testing.assert_frame_equal(df, mock_df)

    @patch("pandas.read_csv", side_effect=FileNotFoundError)
    def test_read_sic_data_missing_file(
        self, _mock_read_csv
    ):  # W0613 Fix: Renamed mock_read_csv
        """Test FileNotFoundError when the data file is missing."""  # C0116 Fix
        with self.assertRaises(FileNotFoundError):
            read_sic_data("sic_data.csv", loaded_config=None)

    @patch("pandas.read_csv", side_effect=pd.errors.ParserError("Mock Parser Error"))
    def test_read_sic_data_malformed_csv(
        self, _mock_read_csv
    ):  # W0613 Fix: Renamed mock_read_csv
        """Test ParserError for malformed CSV data."""  # C0116 Fix
        with self.assertRaises(pd.errors.ParserError):
            read_sic_data("sic_data.csv", loaded_config=None)


# --- NEW TEST CLASS ---
class TestAddDataQualityFlags(unittest.TestCase):
    """Tests the add_data_quality_flags function."""

    def setUp(self):
        """Set up sample data for testing quality flags."""
        self.sample_data = pd.DataFrame(
            {
                # sic_ind_occ1 variations
                "sic_ind_occ1": [
                    "12345",  # Match 5
                    "5432x",  # Match 4
                    "987xx",  # Match 3
                    "12xxx",  # Match 2
                    "1xxxx",  # Match 1
                    "abcde",  # Invalid format (non-digit/x)
                    "1234",  # Invalid length
                    "1234xX",  # Invalid length
                    "1234Z",  # Invalid char
                    "123xxY",  # Invalid char
                    None,  # Missing value
                    "98765",  # Match 5 (for single answer test)
                    "11111",  # Match 5 (for unambiguous test)
                    "22222",  # Match 5 (for unambiguous false test)
                ],
                # sic_ind_occ2 variations
                "sic_ind_occ2": [
                    "67890",  # Multiple answer
                    "11111",  # Multiple answer
                    "22222",  # Multiple answer
                    "33333",  # Multiple answer
                    "44444",  # Multiple answer
                    "55555",  # Multiple answer
                    "66666",  # Multiple answer
                    "77777",  # Multiple answer
                    "88888",  # Multiple answer
                    "99999",  # Multiple answer
                    "00000",  # Multiple answer
                    None,  # Single answer (occ2=None)
                    "NA",  # Single answer (occ2="NA")
                    "12345",  # Multiple answer
                ],
                # sic_ind_occ3 variations
                "sic_ind_occ3": [
                    "11223",  # Multiple answer
                    "22334",  # Multiple answer
                    "33445",  # Multiple answer
                    "44556",  # Multiple answer
                    "55667",  # Multiple answer
                    "66778",  # Multiple answer
                    "77889",  # Multiple answer
                    "88990",  # Multiple answer
                    "99001",  # Multiple answer
                    "00112",  # Multiple answer
                    "11223",  # Multiple answer
                    np.nan,  # Single answer (occ3=NaN)
                    "na",  # Single answer (occ3="na") - testing case insensitivity
                    None,  # Single answer (occ3=None) - but occ2 has value
                ],
                # Add other columns if needed by config lookup, otherwise defaults used
                "unique_id": [f"id_{i}" for i in range(14)],
            }
        )
        # Ensure correct initial dtypes (object/string)
        self.sample_data = self.sample_data.astype(str)
        # Manually set None/NaN where intended (astype(str) converts None to 'None')
        self.sample_data.loc[10, "sic_ind_occ1"] = None
        self.sample_data.loc[11, "sic_ind_occ2"] = None
        self.sample_data.loc[12, "sic_ind_occ2"] = "NA"
        self.sample_data.loc[11, "sic_ind_occ3"] = np.nan
        self.sample_data.loc[12, "sic_ind_occ3"] = "na"
        self.sample_data.loc[13, "sic_ind_occ3"] = None

    def test_flag_calculations(self):
        """Test all flag calculations on the sample data."""
        # Use default column names (no config needed for this test)
        print(self.sample_data)
        result_df = add_data_quality_flags(self.sample_data, loaded_config=None)

        # Expected results (Pandas Nullable Boolean)
        expected_single_answer = pd.Series(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                False,  # Only rows 11, 12 should be True
            ],
            dtype="boolean",
        )

        expected_match_5 = pd.Series(
            [
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,  # Row 10 (original None) should be NA
            ],
            dtype="boolean",
        )

        expected_match_4 = pd.Series(
            [
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            dtype="boolean",
        )

        expected_match_3 = pd.Series(
            [
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            dtype="boolean",
        )

        expected_match_2 = pd.Series(
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            dtype="boolean",
        )

        expected_match_1 = pd.Series(
            [
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            dtype="boolean",
        )

        expected_unambiguous = pd.Series(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                False,  # Rows 11, 12 are SingleAnswer & Match5
            ],
            dtype="boolean",
        )

        # Assertions
        assert_series_equal(
            result_df["Single_Answer"], expected_single_answer, check_names=False
        )
        assert_series_equal(
            result_df["Match_5_digits"], expected_match_5, check_names=False
        )
        assert_series_equal(
            result_df["Match_4_digits"], expected_match_4, check_names=False
        )
        assert_series_equal(
            result_df["Match_3_digits"], expected_match_3, check_names=False
        )
        assert_series_equal(
            result_df["Match_2_digits"], expected_match_2, check_names=False
        )
        assert_series_equal(
            result_df["Match_1_digits"], expected_match_1, check_names=False
        )
        assert_series_equal(
            result_df["Unambiguous"], expected_unambiguous, check_names=False
        )

    def test_missing_input_columns(self):
        """Test that the function handles missing input columns gracefully."""
        df_missing_cols = self.sample_data.drop(
            columns=["sic_ind_occ2", "sic_ind_occ3"]
        )
        # Function should log an error and return the original df
        result_df = add_data_quality_flags(df_missing_cols, loaded_config=None)
        # Check that no new flag columns were added
        self.assertNotIn("Single_Answer", result_df.columns)
        self.assertNotIn("Match_5_digits", result_df.columns)
        self.assertNotIn("Unambiguous", result_df.columns)
        # Check that the returned df is the same as the input (or a copy)
        assert_frame_equal(result_df, df_missing_cols)


#    def test_empty_dataframe(self):
#        """Test that the function handles an empty input DataFrame."""
#        empty_df = pd.DataFrame(columns=self.sample_data.columns)
#        result_df = add_data_quality_flags(empty_df, config=None)
#        self.assertTrue(result_df.empty)
#        # Check that flag columns weren't added
#        self.assertNotIn("Single_Answer", result_df.columns)


class TestLoadFilter(unittest.TestCase):
    """Tests the configuration loading and data reading functions."""  # C0115 Fix

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='[settings]\napi_gateway = "https://api.example.com"\n',
    )
    def test_load_config_valid(self, _mock_file):  # W0613 Fix: Renamed mock_file
        """Test loading a valid TOML configuration file."""  # C0116 Fix
        config = load_config("config.toml")
        self.assertEqual(config["settings"]["api_gateway"], "https://api.example.com")

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_config_missing_file(self, _mock_file):  # W0613 Fix: Renamed mock_file
        """Test FileNotFoundError when the config file is missing."""  # C0116 Fix
        with self.assertRaises(FileNotFoundError):
            load_config("config.toml")

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='[settings]\napi_gateway = "https://api.example.com"\ninvalid_toml',
    )
    def test_load_config_malformed_data(
        self, _mock_file
    ):  # W0613 Fix: Renamed mock_file
        """Test TomlDecodeError for malformed TOML data."""  # C0116 Fix
        with self.assertRaises(toml.TomlDecodeError):
            load_config("config.toml")

    @patch("pandas.read_csv")
    def test_read_sic_data_valid(
        self, mock_read_csv
    ):  # W0613 Fix: _mock_read_csv -> mock_read_csv (Now used)
        """Test reading valid SIC data using a mocked DataFrame."""  # C0116 Fix
        mock_df = pd.DataFrame(
            {
                "unique_id": ["1"],
                "sic_section": ["A"],
                "sic2007_employee": ["10"],
                "sic2007_self_employed": ["5"],
                "sic_ind1": ["B"],
                "sic_ind2": ["C"],
                "sic_ind3": ["D"],
                "sic_ind_code_flag": ["E"],
                "soc2020_job_title": ["F"],
                "soc2020_job_description": ["G"],
                "sic_ind_occ1": ["H"],
                "sic_ind_occ2": ["I"],
                "sic_ind_occ3": ["J"],
                "sic_ind_occ_flag": ["K"],
            }
        )
        mock_read_csv.return_value = mock_df
        df = read_sic_data("sic_data.csv")
        # Check the mock was called (optional but good)
        mock_read_csv.assert_called_once()
        # Compare the returned DataFrame with the mock
        pd.testing.assert_frame_equal(df, mock_df)

    @patch("pandas.read_csv", side_effect=FileNotFoundError)
    def test_read_sic_data_missing_file(
        self, _mock_read_csv
    ):  # W0613 Fix: Renamed mock_read_csv
        """Test FileNotFoundError when the data file is missing."""  # C0116 Fix
        with self.assertRaises(FileNotFoundError):
            read_sic_data("sic_data.csv")

    @patch("pandas.read_csv", side_effect=pd.errors.ParserError)
    def test_read_sic_data_malformed_csv(
        self, _mock_read_csv
    ):  # W0613 Fix: Renamed mock_read_csv
        """Test ParserError for malformed CSV data."""  # C0116 Fix
        with self.assertRaises(pd.errors.ParserError):
            read_sic_data("sic_data.csv")


if __name__ == "__main__":
    unittest.main()
