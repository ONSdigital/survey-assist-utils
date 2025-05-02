import unittest
from unittest.mock import patch, mock_open
import toml
import pandas as pd

from src.load_filter import load_config, read_sic_data

class TestLoadFilter(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data='[settings]\napi_gateway = "https://api.example.com"\n')
    def test_load_config_valid(self, mock_file):
        config = load_config("config.toml")
        self.assertEqual(config["settings"]["api_gateway"], "https://api.example.com")

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_config_missing_file(self, mock_file):
        with self.assertRaises(FileNotFoundError):
            load_config("config.toml")

    @patch("builtins.open", new_callable=mock_open, read_data='[settings]\napi_gateway = "https://api.example.com"\ninvalid_toml')
    def test_load_config_malformed_data(self, mock_file):
        with self.assertRaises(toml.TomlDecodeError):
            load_config("config.toml")

    @patch("pandas.read_csv")
    def test_read_sic_data_valid(self, mock_read_csv):
        mock_df = pd.DataFrame({
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
            "sic_ind_occ_flag": ["K"]
        })
        mock_read_csv.return_value = mock_df
        df = read_sic_data("sic_data.csv")
        pd.testing.assert_frame_equal(df, mock_df)

    @patch("pandas.read_csv", side_effect=FileNotFoundError)
    def test_read_sic_data_missing_file(self, mock_read_csv):
        with self.assertRaises(FileNotFoundError):
            read_sic_data("sic_data.csv")

    @patch("pandas.read_csv", side_effect=pd.errors.ParserError)
    def test_read_sic_data_malformed_csv(self, mock_read_csv):
        with self.assertRaises(pd.errors.ParserError):
            read_sic_data("sic_data.csv")

if __name__ == "__main__":
    unittest.main()
