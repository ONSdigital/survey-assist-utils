"""Script that calculates accuracy metrics for survey assist evaluation model.

Takes evaluation_data and test_type as positional arguments.

Allows parsing --neglect_impossible as optional arguments.

Use:
    -h, --help to show help message.
"""

from argparse import ArgumentParser as AP

import pandas as pd

from survey_assist_utils.data_cleaning.sic_codes import (
    get_clean_n_digit_codes,
    parse_clerical_code,
)
from survey_assist_utils.evaluation.metrics import (
    calc_simple_metrics,
)


def prep_dataframe(input_df: pd.DataFrame, digits: int = 5) -> pd.DataFrame:
    """Prepares the input DataFrame for evaluation by ensuring necessary columns exist.

    Args:
        input_df: Input DataFrame to be prepared.
        digits: Number of digits to which SIC codes should be cleaned/expanded.

    Returns:
        Prepared DataFrame with necessary columns.

    Raises:
        ValueError: If required columns are missing in the input DataFrame.
    """
    required_columns = {
        "Unambiguous",
        "All_Clerical_codes",
        "initial_code",
        "alt_sic_candidates",
        "final_sic",
    }
    if miss := required_columns - set(input_df.columns):
        raise ValueError(f"Input DataFrame is missing required columns: {miss}")

    # Parse clerical coder column to actual list of strings
    input_df["clerical_codes"] = (
        input_df["All_Clerical_codes"]
        .apply(parse_clerical_code)
        .apply(get_clean_n_digit_codes, n=digits)
    )

    # Extract the codes from the model's alt_sic_candidates if ambiguous
    input_df["initial_code_combined"] = input_df["initial_code"]
    fill_alternatives = input_df["initial_code"].isna() | (
        input_df["initial_code"] == ""
    )
    input_df.loc[fill_alternatives, "initial_code_combined"] = input_df.loc[
        fill_alternatives, "alt_sic_candidates"
    ].apply(lambda x: [y["code"] for y in x])
    input_df["sa_initial_codes"] = input_df["initial_code_combined"].apply(
        get_clean_n_digit_codes, n=digits
    )

    # Parse the final sic code from the model output
    input_df["sa_final_codes"] = (
        input_df["final_sic"]
        .apply(parse_clerical_code)
        .apply(get_clean_n_digit_codes, n=digits)
    )

    return input_df


def remove_four_plus(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows where clerical coder recorded '4+' or similar.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame without '4+' clerical codes.
    """
    clerical_missing = df["clerical_codes"].apply(lambda x: len(x) == 0)

    if clerical_missing.any():
        print(
            f"{clerical_missing.sum()} records had no usable clerically coded "
            "answer (such as 4+), and are ignored in calculation"
        )
    return df[~clerical_missing].reset_index(drop=True)


if __name__ == "__main__":
    parser = AP()

    parser.add_argument(
        "evaluation_data", type=str, help="relative path to the parquet dataset"
    )

    parser.add_argument("match_type", type=str, help="match type: full / n-digit")

    parser.add_argument(
        "--neglect_impossible",
        "-n",
        action="store_true",
        default=False,
        help="ignore rows where no n-digit clerical code is available when calculating accuracy",
    )

    args = parser.parse_args()

    if not args.match_type.startswith(
        ("full", "1-digit", "2-digit", "3-digit", "4-digit", "5-digit")
    ):
        raise ValueError("illegal value passed for match_type")

    # Load final-stage output DataFrame
    try:
        my_dataframe = pd.read_parquet(args.evaluation_data)
    except FileNotFoundError as e:
        print(f"Could not read file: {args.evaluation_data}")
        raise e

    # Prepare the DataFrame for evaluation
    my_dataframe = prep_dataframe(
        my_dataframe, digits=5 if args.match_type == "full" else int(args.match_type[0])
    )

    # Remove rows with no usable clerical code (if specified)
    if args.neglect_impossible:
        my_dataframe = remove_four_plus(my_dataframe)

    evaluation_metrics = calc_simple_metrics(my_dataframe)

    evaluation_metrics.print_metrics()
