# tests/test_batch.py

from unittest.mock import patch, mock_open, MagicMock

import json
import logging
import unittest
from scripts.batch import add, subtract
import pandas as pd
from io import StringIO


from scripts.batch import read_sic_data
from scripts.batch import subset_data
from scripts.batch import process_test_set, process_row


class TestMyFunctions(unittest.TestCase):

    def test_add(self):
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(-1, 1), 0)

    def test_subtract(self):
        self.assertEqual(subtract(2, 1), 1)
        self.assertEqual(subtract(2, 2), 0)


class TestReadSicData(unittest.TestCase):
    
    def setUp(self):
        # Sample data to mimic the CSV file content
        self.sample_data = """1\tA\t10\t5\tind1\tind2\tind3\tflag1\ttitle1\tdescription1\tocc1\tocc2\tocc3\tflag2
2\tB\t20\t10\tind4\tind5\tind6\tflag3\ttitle2\tdescription2\tocc4\tocc5\tocc6\tflag4"""
        
        # Expected column names
        self.column_names = [
            "unique_id", "sic_section", "sic2007_employee", "sic2007_self_employed", 
            "sic_ind1", "sic_ind2", "sic_ind3", "sic_ind_code_flag", 
            "soc2020_job_title", "soc2020_job_description", "sic_ind_occ1", 
            "sic_ind_occ2", "sic_ind_occ3", "sic_ind_occ_flag"
        ]
    
    def test_read_sic_data(self):
        # Use StringIO to simulate reading from a file
        test_data = StringIO(self.sample_data)
        
        # Call the function with the simulated file
        df = read_sic_data(test_data)
        
        # Check if the DataFrame has the correct columns
        self.assertEqual(list(df.columns), self.column_names)
        
        # Check if the data is read correctly
        self.assertEqual(df.iloc[0].tolist(), ["1", "A", "10", "5", "ind1", "ind2", "ind3", "flag1", "title1", "description1", "occ1", "occ2", "occ3", "flag2"])
        self.assertEqual(df.iloc[1].tolist(), ["2", "B", "20", "10", "ind4", "ind5", "ind6", "flag3", "title2", "description2", "occ4", "occ5", "occ6", "flag4"])



class TestSubsetData(unittest.TestCase):
    
    def setUp(self):
        # Sample data to mimic the CSV file content
        self.sample_data = """1\tA\t10\t5\t12345\tind2\tind3\tflag1\ttitle1\tdescription1\tocc1\tocc2\tocc3\tflag2
2\tB\t20\t10\t67890\tind5\tind6\tflag3\ttitle2\tdescription2\tocc4\tocc5\tocc6\tflag4
3\tC\t-9\t15\t54321\tind8\tind9\tflag5\ttitle3\tdescription3\tocc7\tocc8\tocc9\tflag6
4\tD\t30\t-8\t98765\tind11\tind12\tflag7\ttitle4\tdescription4\tocc10\tocc11\tocc12\tflag8
5\tE\t40\t20\t12345\tind14\tind15\tflag9\ttitle5\tdescription5\tocc13\tocc14\tocc15\tflag10"""
        
        # Expected column names
        self.column_names = [
            "unique_id", "sic_section", "sic2007_employee", "sic2007_self_employed", 
            "sic_ind1", "sic_ind2", "sic_ind3", "sic_ind_code_flag", 
            "soc2020_job_title", "soc2020_job_description", "sic_ind_occ1", 
            "sic_ind_occ2", "sic_ind_occ3", "sic_ind_occ_flag"
        ]
    
    def test_subset_data(self):
        # Use StringIO to simulate reading from a file
        test_data = StringIO(self.sample_data)
        
        # Call the function with the simulated file
        df = subset_data(test_data)
        
        # Check if the DataFrame has the correct columns
        self.assertEqual(list(df.columns), self.column_names)
        
        # Check if the data is filtered correctly
        self.assertEqual(df.shape[0], 1)  # Only one row should remain after filtering
        self.assertEqual(df.iloc[0]['sic_ind1'], '12345')  # The remaining row should have 'sic_ind1' as '12345'


class TestAPIProcessor(unittest.TestCase):

    @patch('api_processor.requests.post')
    def test_process_row_success(self, mock_post):
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'classification': 'example_classification'}
        mock_post.return_value = mock_response

        # Sample row data
        row = pd.Series({
            'unique_id': '123',
            'soc2020_job_title': 'Software Engineer',
            'soc2020_job_description': 'Develops software applications',
            'sic2007_employee': 'Technology'
        })

        secret_code = 'test_secret_code'
        response_json = process_row(row, secret_code)

        # Check the response
        self.assertEqual(response_json['unique_id'], '123')
        self.assertEqual(response_json['job_title'], 'Software Engineer')
        self.assertEqual(response_json['job_description'], 'Develops software applications')
        self.assertEqual(response_json['industry_descr'], 'Technology')
        self.assertEqual(response_json['classification'], 'example_classification')

    @patch('api_processor.requests.post')
    def test_process_row_failure(self, mock_post):
        # Mock the API response with an error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        # Sample row data
        row = pd.Series({
            'unique_id': '123',
            'soc2020_job_title': 'Software Engineer',
            'soc2020_job_description': 'Develops software applications',
            'sic2007_employee': 'Technology'
        })

        secret_code = 'test_secret_code'
        response_json = process_row(row, secret_code)

        # Check the response
        self.assertEqual(response_json['unique_id'], '123')
        self.assertEqual(response_json['job_title'], 'Software Engineer')
        self.assertEqual(response_json['job_description'], 'Develops software applications')
        self.assertEqual(response_json['industry_descr'], 'Technology')
        self.assertIn('error', response_json)

    @patch('api_processor.process_row')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pandas.read_csv')
    def test_process_test_set(self, mock_read_csv, mock_open, mock_process_row):
        # Mock the CSV reading
        mock_read_csv.return_value = pd.DataFrame({
            'unique_id': ['123', '456'],
            'soc2020_job_title': ['Software Engineer', 'Data Scientist'],
            'soc2020_job_description': ['Develops software applications', 'Analyzes data'],
            'sic2007_employee': ['Technology', 'Science']
        })

        # Mock the process_row function
        mock_process_row.side_effect = [
            {'unique_id': '123', 'classification': 'example_classification'},
            {'unique_id': '456', 'classification': 'example_classification'}
        ]

        secret_code = 'test_secret_code'
        csv_filepath = 'test.csv'
        output_filepath = 'output.json'

        process_test_set(secret_code, csv_filepath, output_filepath, test_mode=True, test_limit=2)

        # Check that the file was written to
        mock_open().write.assert_called()
        self.assertEqual(mock_open().write.call_count, 2)




class TestAPIProcessor(unittest.TestCase):

    @patch('api_processor.requests.post')
    def test_process_row_success(self, mock_post):
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'classification': 'example_classification'}
        mock_post.return_value = mock_response

        # Sample row data
        row = pd.Series({
            'unique_id': '123',
            'soc2020_job_title': 'Software Engineer',
            'soc2020_job_description': 'Develops software applications',
            'sic2007_employee': 'Technology'
        })

        secret_code = 'test_secret_code'
        response_json = process_row(row, secret_code)

        # Check the response
        self.assertEqual(response_json['unique_id'], '123')
        self.assertEqual(response_json['job_title'], 'Software Engineer')
        self.assertEqual(response_json['job_description'], 'Develops software applications')
        self.assertEqual(response_json['industry_descr'], 'Technology')
        self.assertEqual(response_json['classification'], 'example_classification')

    @patch('api_processor.requests.post')
    def test_process_row_failure(self, mock_post):
        # Mock the API response with an error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        # Sample row data
        row = pd.Series({
            'unique_id': '123',
            'soc2020_job_title': 'Software Engineer',
            'soc2020_job_description': 'Develops software applications',
            'sic2007_employee': 'Technology'
        })

        secret_code = 'test_secret_code'
        response_json = process_row(row, secret_code)

        # Check the response
        self.assertEqual(response_json['unique_id'], '123')
        self.assertEqual(response_json['job_title'], 'Software Engineer')
        self.assertEqual(response_json['job_description'], 'Develops software applications')
        self.assertEqual(response_json['industry_descr'], 'Technology')
        self.assertIn('error', response_json)

    @patch('api_processor.process_row')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pandas.read_csv')
    def test_process_test_set(self, mock_read_csv, mock_open, mock_process_row):
        # Mock the CSV reading
        mock_read_csv.return_value = pd.DataFrame({
            'unique_id': ['123', '456'],
            'soc2020_job_title': ['Software Engineer', 'Data Scientist'],
            'soc2020_job_description': ['Develops software applications', 'Analyzes data'],
            'sic2007_employee': ['Technology', 'Science']
        })

        # Mock the process_row function
        mock_process_row.side_effect = [
            {'unique_id': '123', 'classification': 'example_classification'},
            {'unique_id': '456', 'classification': 'example_classification'}
        ]

        secret_code = 'test_secret_code'
        csv_filepath = 'test.csv'
        output_filepath = 'output.json'

        process_test_set(secret_code, csv_filepath, output_filepath, test_mode=True, test_limit=2)

        # Check that the file was written to
        mock_open().write.assert_called()
        self.assertEqual(mock_open().write.call_count, 2)

if __name__ == '__main__':
    unittest.main()



