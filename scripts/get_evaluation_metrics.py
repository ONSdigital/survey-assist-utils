"""Script that calculates accuracy metrics for survey assist evaluation model.

Takes evaluation_data and match_digits as positional arguments.

Optional arguments:
    -n, --neglect_four_plus to exclude entries with 4 or more clerical codes.
    -o, --old_one_prompt to expect data format from one_prompt pipeline.

Use:
    -h, --help to show help message.
"""

import logging
from argparse import ArgumentParser as AP

import pandas as pd

from survey_assist_utils.data_cleaning.sic_codes import (
    INVALID_VALUES,
    extract_alt_sic_candidates,
    get_clean_n_digit_codes,
    parse_numerical_code,
)
from survey_assist_utils.evaluation.metrics import (
    calc_simple_metrics,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
            threshold: Likelihood threshold for pruning alternative candidates.

    Returns:
        Prepared DataFrame with necessary columns.

    Raises:
        ValueError: If required columns are missing in the input DataFrame.
    """
    clerical_codes_col = col_names.get("clerical_codes_col", "All_Clerical_codes")
    initial_code_col = col_names.get("initial_code_col", "sa_initial_codes")
    initial_alt_codes_col = col_names.get("initial_alt_codes_col")
    final_sic = col_names.get("final_sic")
    code_name = col_names.get("code_name", "code")
    threshold = float(col_names.get("threshold", 0))  # default no pruning

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
        .apply(parse_numerical_code)
        .apply(get_clean_n_digit_codes, n=digits)
    )

    # Extract the codes from the model's alt_sic_candidates if ambiguous
    input_df["initial_code_combined"] = input_df[initial_code_col]
    fill_alternatives = input_df[initial_code_col].isna() | (
        input_df[initial_code_col].isin(INVALID_VALUES)
    )

    if initial_alt_codes_col is not None:
        logger.info(
            "Filling initial codes from alternatives for %d rows.",
            fill_alternatives.sum(),
        )
        input_df.loc[fill_alternatives, "initial_code_combined"] = input_df.loc[
            fill_alternatives, initial_alt_codes_col
        ].apply(extract_alt_sic_candidates, code_name=code_name, threshold=threshold)

    input_df["sa_initial_codes"] = input_df["initial_code_combined"].apply(
        get_clean_n_digit_codes, n=digits
    )

    if final_sic is not None:
        # Parse the final sic code from the model output
        input_df.loc[~fill_alternatives, final_sic] = input_df.loc[
            ~fill_alternatives, initial_code_col
        ]
        input_df["sa_final_codes"] = input_df[final_sic].apply(
            get_clean_n_digit_codes, n=digits
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
        logger.info(
            "Loaded %d rows from %s for evaluation.",
            len(my_dataframe),
            args.evaluation_data,
        )
    except FileNotFoundError as e:
        logger.error("Could not read file: %s", args.evaluation_data)
        raise e

    # Remove rows with no usable clerical code (if specified)
    if args.neglect_four_plus:
        my_dataframe = my_dataframe[~my_dataframe["Four_Or_More"]].reset_index(
            drop=True
        )
        logger.info(
            "Removed entries with 4 or more clerical codes. %d rows remain for evaluation.",
            len(my_dataframe),
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
            "threshold": "0.7",
        }
    )

    # Prepare the DataFrame for evaluation
    logger.info(
        "Calculating evaluation metrics with %s match. Using following columns: %s",
        args.match_digits,
        column_names,
    )
    my_dataframe = prep_dataframe(
        my_dataframe,
        digits=5 if args.match_digits == "full" else int(args.match_digits[0]),
        col_names=column_names,
    )

    evaluation_metrics = calc_simple_metrics(my_dataframe)

    logger.info(evaluation_metrics.report_metrics())
