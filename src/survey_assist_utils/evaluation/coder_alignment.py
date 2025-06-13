"""Module contains functionality to evaluate alignment between Clerical Coders (CC)
and Survey Assist (SA) results.

The class is AlignmentEvaluator

The methods are:
calculate_first_choice_rate
calculate_match_rate_at_n
"""

import pandas as pd


class AlignmentEvaluator:
    """A class to handle comparison between CC labels and outputs from SA.

    This object loads, preprocesses, and calculates metrics on SIC code data,
    encapsulating all the related logic in one place.
    """

    def __init__(self, filepath: str):
        """Initialises the analyser by loading and preprocessing the data.

        Args:
            filepath (str): The path to the CSV file containing merged results from both CC
            and SA.
        """
        # Some asserts / try for fileread fails
        self._data = pd.read_csv(filepath, dtype=str)
        print(f"Successfully loaded data with shape: {self._data.shape}")

        self._add_helper_columns()

    def _add_helper_columns(self):
        """A private helper method to perform the initial data setup.

        SIC Division (2 digits), Sub division (3 digits) Class (4 digits).
        """
        print("Adding helper columns for analysis...")
        code_lengths = [2, 3, 4]

        # Generate substrings for 'chosen_sic_code'
        for n in code_lengths:
            self._data[f"code_{n}"] = self._data["chosen_sic_code"].str[:n]

        # Generate substrings for 'sic_ind_occ1'
        for n in code_lengths:
            self._data[f"code_cc_{n}"] = self._data["sic_ind_occ1"].str[:n]

    def calculate_first_choice_rate(self, col1, col2) -> float:
        """Calculates the percentage of exact matches between 'sic_ind_occ1'.
        and 'chosen_sic_code'. - Do we want to change which cols we compare.
        self,
        col1,
        col2.
        """
        data = self._data

        # The total number of records is simply the length of the DataFrame.
        # This is clearer and less error-prone than the original calculation.
        total = len(data)

        # Calculate matching values
        matching = (
            data[col1] == data[col2]
        ).sum()  # matching = (data[col1].values == data[col2].values).sum() ?

        # Calculate percentage
        matching_percent = round(100 * matching / total, 2) if total > 0 else 0.0

        return matching_percent

    def calculate_match_rate_at_n(self, n: int) -> float:
        """Calculates the match rate for the first N digits of the SIC code."""
        data = self._data
        total = len(data)

        match_col_1 = f"code_{n}"
        match_col_2 = f"code_cc_{n}"

        if match_col_1 not in data.columns or match_col_2 not in data.columns:
            raise ValueError(f"Helper columns for n={n} do not exist.")

        matching = (data[match_col_1] == data[match_col_2]).sum()
        return round(100 * matching / total, 1) if total > 0 else 0.0


# Where are the results and original input data kept?
# Test that was run:
# Input data (eg 2000 random selection)
# Output data - some json files
#
# Job list
# @staticmethod
#    def save_output(
#        metadata: dict, eval_result: dict, save_path: str = "../data/"
#    ) -> str:
#        """Save evaluation results to files.

#    Args:
#        metadata: Dictionary of metadata parameters
#        eval_result: Dictionary containing evaluation metrics
#        save_path: (str) The folder where results should be saved. Default is "../data/".

#        Returns
#        -------
#            str: The folder path where results were stored
#        """
#
# Hard coded? chosen_sic_code LLM column name
