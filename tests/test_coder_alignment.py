"""tests the coder alignment module."""

import pandas as pd

from survey_assist_utils.evaluation.coder_alignment import AlignmentEvaluator


def test_sic_analysis_calculations():
    """Unit test for the AlignmentEvaluator class.
    It follows the "Arrange, Act, Assert" pattern.
    """
    # Create a small, predictable DataFrame for the test.
    test_data = pd.DataFrame(
        {
            "sic_ind_occ1": ["1234", "5678", "1299", "8800"],
            "chosen_sic_code": ["1234", "9999", "1234", "7700"],
        }
    )

    # Create a path to a temporary CSV file inside the temp directory.
    temp_csv_path = "test_data.csv"

    # Save our test DataFrame to that temporary CSV file.
    test_data.to_csv(temp_csv_path, index=False)

    # Expected results from our 4 rows of test data:
    # - Full match: 1 out of 4 rows ("1234" == "1234") -> 25.0%
    # - 2-digit match: 2 out of 4 rows ('12'=='12', '12'=='12') -> 50.0%
    expected_full_match_rate = 25.0
    expected_2_digit_match_rate = 50.0

    analyzer = AlignmentEvaluator(filepath=temp_csv_path)

    # Call the methods to get the actual results.
    actual_full_match_rate = analyzer.calculate_match_rate(
        col1="sic_ind_occ1", col2="chosen_sic_code"
    )
    print("actual_full_match_rate", actual_full_match_rate)
    actual_2_digit_match_rate = analyzer.calculate_match_rate(
        col1="sic_ind_occ1", col2="chosen_sic_code", n=2
    )
    print("actual_2_digit_match_rate", actual_2_digit_match_rate)

    assert actual_full_match_rate == expected_full_match_rate
    assert actual_2_digit_match_rate == expected_2_digit_match_rate
