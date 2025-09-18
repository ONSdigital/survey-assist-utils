"""Script that calculates accuracy metrics for survey assist evaluation model.

Takes evaluation_data and match_digits as positional arguments.

Optional arguments:
    -n, --neglect_four_plus to exclude entries with 4 or more clerical codes.
    -o, --old_one_prompt to expect data format from one_prompt pipeline.

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
from survey_assist_utils.logging import get_logger

logger = get_logger(__name__)


def prep_dataframe(
    input_df: pd.DataFrame,
    col_names: dict,
    digits: int = 5,
) -> pd.DataFrame:
    """Prepares the input DataFrame for evaluation by ensuring necessary columns exist.

    Args:
        input_df: Input DataFrame to be prepared.
        digits: Number of digits to which SIC codes should be cleaned/expanded.
        col_names: Dictionary with column names:
            clerical_codes_col: Column name for clerical codes (string or list).
            initial_code_col: Column name for initial model predicted code (string).
            initial_alt_codes_col: Column name for alternative codes (list of dicts).
            final_sic: Column name for final model predicted code (string), if available.
            code_name: Key name to extract codes from alternative predictions.

    Returns:
        Prepared DataFrame with necessary columns.

    Raises:
        ValueError: If required columns are missing in the input DataFrame.
    """
    clerical_codes_col = col_names.get("clerical_codes_col", "All_Clerical_codes")
    initial_code_col = col_names.get("initial_code_col", "sa_initial_codes")
    initial_alt_codes_col = col_names.get(
        "initial_alt_codes_col", "sa_initial_alt_codes"
    )
    final_sic = col_names.get("final_sic")
    code_name = col_names.get("code_name", "code")

    if final_sic and final_sic not in input_df.columns:
        logger.warning(
            "No column for final code assignment provided. Evaluation"
            "of codability gain and final accuracy won't be possible."
        )
        final_sic = None

    required_columns = [
        clerical_codes_col,
        initial_code_col,
        initial_alt_codes_col,
    ] + ([final_sic] if final_sic else [])
    if miss := set(required_columns) - set(input_df.columns):
        raise ValueError(f"Input DataFrame is missing required columns: {miss}")
    input_df = input_df[required_columns].copy()

    # Parse clerical coder column to actual list of strings
    input_df["clerical_codes"] = (
        input_df[clerical_codes_col]
        .apply(parse_clerical_code)
        .apply(get_clean_n_digit_codes, n=digits)
    )

    # Extract the codes from the model's alt_sic_candidates if ambiguous
    input_df["initial_code_combined"] = input_df[initial_code_col]
    fill_alternatives = input_df[initial_code_col].isna() | (
        input_df[initial_code_col] == ""
    )
    input_df.loc[fill_alternatives, "initial_code_combined"] = input_df.loc[
        fill_alternatives, initial_alt_codes_col
    ].apply(lambda x: [y[code_name] for y in x])
    input_df["sa_initial_codes"] = input_df["initial_code_combined"].apply(
        get_clean_n_digit_codes, n=digits
    )

    if final_sic:
        # Parse the final sic code from the model output
        input_df.loc[~fill_alternatives, final_sic] = input_df.loc[
            ~fill_alternatives, initial_code_col
        ]
        input_df["sa_final_codes"] = (
            input_df[final_sic]
            .apply(parse_clerical_code)
            .apply(get_clean_n_digit_codes, n=digits)
        )

    return input_df


if __name__ == "__main__":
    parser = AP()

    parser.add_argument(
        "evaluation_data", type=str, help="relative path to the parquet dataset"
    )

    parser.add_argument("match_digits", type=str, help="match type: full / n-digit")

    parser.add_argument(
        "--neglect_four_plus",
        "-n",
        action="store_true",
        default=False,
        help="ignore rows where additional clerical codes are saved outside of the input DataFrame",
    )

    parser.add_argument(
        "--old_one_prompt",
        "-o",
        action="store_true",
        default=False,
        help="expect data format from one_prompt pipeline (column names)",
    )

    args = parser.parse_args()

    if not args.match_digits.startswith(
        ("full", "1-digit", "2-digit", "3-digit", "4-digit", "5-digit")
    ):
        raise ValueError("illegal value passed for match_digits")

    # Load final-stage output DataFrame
    try:
        my_dataframe = pd.read_parquet(args.evaluation_data)
    except FileNotFoundError as e:
        logger.error(f"Could not read file: {args.evaluation_data}")
        raise e

    # Remove rows with no usable clerical code (if specified)
    if args.neglect_four_plus:
        my_dataframe = my_dataframe[~my_dataframe["Four_Or_More"]].reset_index(
            drop=True
        )

    column_names = (
        {
            "clerical_codes_col": "All_Clerical_codes",
            "initial_code_col": "initial_code",
            "initial_alt_codes_col": "alt_sic_candidates",
            "final_sic": "final_sic",
            "code_name": "code",
        }
        if not args.old_one_prompt
        else {
            "clerical_codes_col": "All_Clerical_codes",
            "initial_code_col": "final_sic_code",
            "initial_alt_codes_col": "sic_candidates",
            "code_name": "sic_code",
        }
    )

    # Prepare the DataFrame for evaluation
    my_dataframe = prep_dataframe(
        my_dataframe,
        digits=5 if args.match_digits == "full" else int(args.match_digits[0]),
        col_names=column_names,
    )

    evaluation_metrics = calc_simple_metrics(my_dataframe)

    evaluation_metrics.print_metrics()
