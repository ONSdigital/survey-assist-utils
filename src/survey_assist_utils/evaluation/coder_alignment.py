"""Module contains functionality to evaluate alignment between Clerical Coders (CC)
and Survey Assist (SA) results.

The class is AlignmentEvaluator

The methods are:
calculate_first_choice_rate
calculate_match_rate_at_n
"""

from typing import Optional

import pandas as pd


# pylint: disable=too-few-public-methods
class AlignmentEvaluator:
    """A class to handle comparison between two sets of classification codes.

    This object loads data and calculates alignment metrics, such as exact
    and partial (n-digit) match rates between any two specified columns.
    """

    def __init__(self, filepath: str):
        """Initialises the evaluator by loading the data.

        Args:
            filepath (str): The path to the CSV file containing the data.
        """
        try:
            self._data = pd.read_csv(filepath, dtype=str)
            print(f"Successfully loaded data with shape: {self._data.shape}")
        except FileNotFoundError:
            print(f"ERROR: The file was not found at {filepath}")
            raise

    def calculate_match_rate(
        self, col1: str, col2: str, n: Optional[int] = None
    ) -> float:
        """Calculates the match rate between two columns, either fully or at n-digits.

        Args:
            col1 (str): The name of the first column to compare.
            col2 (str): The name of the second column to compare.
            n (Optional[int], optional): The number of leading digits to compare.
                If None, a full string comparison is performed. Defaults to None.

        Returns:
            float: The percentage of rows that match, rounded to two decimal places.
        """
        data = self._data
        if col1 not in data.columns or col2 not in data.columns:
            raise ValueError(
                f"One or both columns ('{col1}', '{col2}') not found in data."
            )

        total = len(data)
        if total == 0:
            return 0.0

        # Determine the series to compare based on 'n'
        if n is None:
            # Full match - use the original columns
            series1 = data[col1]
            series2 = data[col2]
        else:
            # Partial match - generate substrings on the fly
            series1 = data[col1].str[:n]
            series2 = data[col2].str[:n]

        matching = (series1 == series2).sum()

        return round(100 * matching / total, 2)
