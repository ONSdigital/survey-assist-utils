import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open # For mocking file I/O and requests
import os
import tempfile # For creating temporary files/dirs for testing

# Import functions from your script
from sic_evaluation import (
    load_gold_standard,
    load_llm_responses,
    merge_data,
    extract_llm_predictions,
    calculate_digit_matches,
    plot_digit_match_histogram,
    COL_UNIQUE_ID, COL_GOLD_SIC, COL_GOLD_FLAG, COL_RESPONSE_CANDIDATES,
    COL_LLM_TOP_SIC, COL_LLM_TOP_LIKELIHOOD
)

# --- Sample Data for Tests ---
SAMPLE_GOLD_CSV = f"{COL_UNIQUE_ID},{COL_GOLD_SIC},{COL_GOLD_FLAG}\n1,12345,KB\n2,54321,CC\n3,98765,MC\n4,11111,KB"
SAMPLE_RESPONSE_JSONL = f"""
{{"{COL_UNIQUE_ID}": "1", "{COL_RESPONSE_CANDIDATES}": [{{"sic_code": "12340", "likelihood": 0.9}}, {{"sic_code": "55555", "likelihood": 0.1}}]}}
{{"{COL_UNIQUE_ID}": "2", "{COL_RESPONSE_CANDIDATES}": [{{"sic_code": "54321", "likelihood": 0.8}}, {{"sic_code": "99999", "likelihood": 0.1}}]}}
{{"{COL_UNIQUE_ID}": "3", "error": "API timeout"}}
{{"{COL_UNIQUE_ID}": "5", "{COL_RESPONSE_CANDIDATES}": [{{"sic_code": "00000", "likelihood": 0.7}}]}}
""" # ID 5 won't merge, ID 3 has error

class TestSicEvaluation(unittest.TestCase):

    def setUp(self):
        # Create temporary directory for test outputs if needed
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.test_dir.name

    def tearDown(self):
        # Clean up temporary directory
        self.test_dir.cleanup()

    def test_load_gold_standard_success(self):
        # Use mock_open to simulate reading from SAMPLE_GOLD_CSV
        with patch('builtins.open', mock_open(read_data=SAMPLE_GOLD_CSV)) as mock_file:
            df = load_gold_standard("dummy_path.csv")
            mock_file.assert_called_once_with("dummy_path.csv", 'r', encoding='utf-8')
            self.assertEqual(len(df), 4)
            self.assertListEqual(list(df.columns), [COL_UNIQUE_ID, COL_GOLD_SIC, COL_GOLD_FLAG])
            self.assertEqual(df.loc[df[COL_UNIQUE_ID] == '1', COL_GOLD_SIC].iloc[0], '12345')

    def test_load_gold_standard_missing_cols(self):
        bad_csv = f"{COL_UNIQUE_ID},{COL_GOLD_SIC}\n1,12345"
        with patch('builtins.open', mock_open(read_data=bad_csv)):
             with self.assertRaisesRegex(ValueError, "missing required columns: {'sic_ind_code_flag'}"):
                  load_gold_standard("dummy_path.csv")

    def test_load_llm_responses_success_and_error(self):
        with patch('builtins.open', mock_open(read_data=SAMPLE_RESPONSE_JSONL)) as mock_file:
            df = load_llm_responses("dummy_responses.jsonl")
            mock_file.assert_called_once_with("dummy_responses.jsonl", 'r', encoding='utf-8')
            self.assertEqual(len(df), 3) # Row with error still loaded, row 5 not included
            self.assertIn(COL_RESPONSE_CANDIDATES, df.columns)
            # Check that the error row has 'error' key
            self.assertTrue('error' in df.loc[df[COL_UNIQUE_ID] == '3'].iloc[0])
            self.assertTrue(pd.isna(df.loc[df[COL_UNIQUE_ID] == '3', COL_RESPONSE_CANDIDATES].iloc[0]))


    def test_merge_data(self):
        gold_df = pd.DataFrame({COL_UNIQUE_ID: ['1', '2', '3', '4'], COL_GOLD_SIC: ['G1', 'G2', 'G3', 'G4']})
        resp_df = pd.DataFrame({COL_UNIQUE_ID: ['1', '2', '5'], 'resp_data': ['R1', 'R2', 'R5']})
        merged = merge_data(gold_df, resp_df)
        self.assertEqual(len(merged), 2) # Only IDs 1 and 2 match
        self.assertListEqual(list(merged[COL_UNIQUE_ID]), ['1', '2'])
        self.assertIn('resp_data', merged.columns)

    def test_extract_llm_predictions(self):
        resp_df = pd.DataFrame({
            COL_UNIQUE_ID: ['1', '2', '3'],
            COL_RESPONSE_CANDIDATES: [
                [{'sic_code': 'A1', 'likelihood': 0.9}, {'sic_code': 'A2', 'likelihood': 0.1}],
                [{'sic_code': 'B1', 'likelihood': 0.8}], # Only one candidate
                None # Missing candidates
            ]
        })
        extracted_df = extract_llm_predictions(resp_df, max_candidates=2)
        self.assertEqual(extracted_df.loc[0, 'llm_sic_code_1'], 'A1')
        self.assertEqual(extracted_df.loc[0, 'llm_likelihood_1'], 0.9)
        self.assertEqual(extracted_df.loc[0, 'llm_sic_code_2'], 'A2')
        self.assertEqual(extracted_df.loc[0, 'llm_likelihood_2'], 0.1)
        self.assertEqual(extracted_df.loc[1, 'llm_sic_code_1'], 'B1')
        self.assertEqual(extracted_df.loc[1, 'llm_likelihood_1'], 0.8)
        self.assertIsNone(extracted_df.loc[1, 'llm_sic_code_2']) # Check None for missing 2nd candidate
        self.assertTrue(pd.isna(extracted_df.loc[1, 'llm_likelihood_2']))
        self.assertIsNone(extracted_df.loc[2, 'llm_sic_code_1']) # Check None for missing candidates list
        self.assertTrue(pd.isna(extracted_df.loc[2, 'llm_likelihood_1']))


    def test_calculate_digit_matches(self):
        test_df = pd.DataFrame({
            'gold': ['12345', '54321', '11111', '98765', '12000', None, '12345', '1'],
            'pred': ['12340', '54321', '11100', '12345', '', '12345', None, '12']
        })
        match_df = calculate_digit_matches(test_df, gold_col='gold', predicted_col='pred', match_level_col='level')
        expected_levels = [4, 5, 3, 0, 0, 0, 0, 1] # Expected match levels
        self.assertListEqual(match_df['level'].tolist(), expected_levels)

    # Mocking matplotlib/seaborn is more complex, often test data aggregation part
    @patch('matplotlib.pyplot.savefig') # Mock savefig to avoid actual file writing
    @patch('matplotlib.pyplot.show') # Mock show
    @patch('matplotlib.pyplot.close') # Mock close
    def test_plot_digit_match_histogram_runs(self, mock_close, mock_show, mock_savefig):
        test_df = pd.DataFrame({'match_level': [5, 4, 4, 3, 3, 3, 1, 0, 0, 5, 4]})
        # Test that the function runs without errors
        try:
            plot_digit_match_histogram(test_df, output_dir=self.output_dir)
            mock_savefig.assert_called_once() # Check that savefig was called
        except Exception as e:
            self.fail(f"plot_digit_match_histogram raised exception: {e}")

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_digit_match_histogram_stratified_runs(self, mock_close, mock_show, mock_savefig):
         test_df = pd.DataFrame({
              'match_level': [5, 4, 4, 3, 1, 0, 5, 4],
              COL_GOLD_FLAG: ['KB', 'KB', 'CC', 'CC', 'MC', 'MC', 'KB', 'CC']
         })
         try:
              plot_digit_match_histogram(test_df, stratify_by=COL_GOLD_FLAG, output_dir=self.output_dir)
              mock_savefig.assert_called_once()
         except Exception as e:
              self.fail(f"plot_digit_match_histogram (stratified) raised exception: {e}")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) # Add argv/exit for running in notebooks/scripts