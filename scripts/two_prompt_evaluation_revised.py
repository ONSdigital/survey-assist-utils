"""Script that calculates accuracy metrics for two prompts evaluation model.

Takes evaluation_data, test_type, and match_type as positional arguments.

Allows parsing --filter_unambiguous, --filter_ambiguous, and
--neglect_impossible as optional atguments.

Use:
    -h, --help to show help message.
"""

from argparse import ArgumentParser as AP

import pandas as pd

from survey_assist_utils.data_cleaning.sic_codes import (
    get_clean_n_digit_codes,
    parse_clerical_code,
)
from survey_assist_utils.evaluation.code_comparison import (
    compare_codes,
)
from survey_assist_utils.evaluation.metrics import (
    calc_ambiguity_metrics,
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
    input_df["clerical_ambiguous"] = input_df["clerical_codes"].apply(
        lambda x: len(x) != 1
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
    input_df["sa_initial_ambiguous"] = input_df["sa_initial_codes"].apply(
        lambda x: len(x) != 1
    )

    # Parse the final sic code from the model output
    input_df["sa_final_codes"] = (
        input_df["final_sic"]
        .apply(parse_clerical_code)
        .apply(get_clean_n_digit_codes, n=digits)
    )
    input_df["sa_final_ambiguous"] = input_df["sa_final_codes"].apply(
        lambda x: len(x) != 1
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

    parser.add_argument(
        "test_type",
        type=str,
        help="test type: OO / MM / OM / MO (M=Many, O=One, format: CC-SA)",
    )
    parser.add_argument("match_type", type=str, help="match type: full / n-digit")

    parser.add_argument(
        "--filter_unambiguous",
        "-fua",
        action="store_true",
        default=False,
        help="add flag to only consider CC-reported unambiguously codable responses",
    )
    parser.add_argument(
        "--filter_ambiguous",
        "-fa",
        action="store_true",
        default=False,
        help="add flag to only consider CC-reported NOT unambiguously codable responses",
    )
    parser.add_argument(
        "--neglect_impossible",
        "-n",
        action="store_true",
        default=False,
        help="ignore rows where no n-digit clerical code is available when calculating accuracy",
    )

    args = parser.parse_args()

    if args.test_type not in [
        "OO",
        "MM",
        "OM",
        "MO",
    ]:
        raise ValueError("illegal value passed for test_type")
    if args.match_type not in ["full"] + [str(x + 1) + "-digit" for x in range(5)]:
        raise ValueError("illegal value passed for match_type")

    # Load final-stage output DataFrame
    try:
        my_dataframe = pd.read_parquet(args.evaluation_data)
    except FileNotFoundError as e:
        print(f"no such file: {args.evaluation_data}")
        raise e

    # Prepare the DataFrame for evaluation
    my_dataframe = prep_dataframe(
        my_dataframe, digits=5 if args.match_type == "full" else int(args.match_type[0])
    )

    # Apply filtering (if specified)
    if args.filter_unambiguous:
        my_dataframe = my_dataframe[my_dataframe["Unambiguous"]]
    elif args.filter_ambiguous:
        my_dataframe = my_dataframe[~my_dataframe["Unambiguous"]]

    # Remove rows with no usable clerical code (if specified)
    if args.neglect_impossible:
        my_dataframe = remove_four_plus(my_dataframe)

    # Calculate ambiguity metrics
    ambig_metrics = calc_ambiguity_metrics(
        my_dataframe, model_col="sa_initial_ambiguous", truth_col="clerical_ambiguous"
    )

    sa_initial_cadable = sum(~my_dataframe["sa_initial_ambiguous"])
    sa_final_cadable = sum(~my_dataframe["sa_final_ambiguous"])

    # Calculate classification accuracy metrics
    total_count = len(my_dataframe)

    # Define the row-wise applyable test function
    def _compare_row(row: pd.Series, model_col="sa_initial_codes") -> bool:
        return compare_codes(
            row["clerical_codes"], row[model_col], method=args.test_type
        )

    my_dataframe["initial_match"] = my_dataframe.apply(
        _compare_row, model_col="sa_initial_codes", axis=1
    )
    initial_match = my_dataframe["initial_match"].sum()

    my_dataframe["final_match"] = my_dataframe.apply(
        _compare_row, model_col="sa_final_codes", axis=1
    )
    final_match = my_dataframe["final_match"].sum()

    # Print filtering information and results
    if args.filter_unambiguous:
        print(
            f"\nOnly considering CC-recorded unambiguously codable records ({total_count}):"
        )
    elif args.filter_ambiguous:
        print(
            f"\nOnly considering CC-recorded NOT unambiguously codable records ({total_count}):"
        )
    else:
        print(f"\nConsidering ALL records ({total_count}):")

    print("\nAmbiguity decision statistics:")
    print(
        f"Precision: {100 * ambig_metrics['precision']:.2f}%, Recall: "
        f"{100 * ambig_metrics['recall']:.2f}%, F1: {100 * ambig_metrics['f1']:.2f}%"
    )
    print(
        f"(TP: {ambig_metrics['TP']}, FP: {ambig_metrics['FP']},"
        f"FN: {ambig_metrics['FN']}, TN: {ambig_metrics['TN']})"
    )

    print(
        f"\nGain in codability: {100 * (sa_final_cadable - sa_initial_cadable) / total_count:.2f}pp"
    )
    print(f"Initial codability: {100 * sa_initial_cadable / total_count:.2f}%")
    print(f"Final codability: {100 * sa_final_cadable / total_count:.2f}%")

    print(f"\nClassification quality metrics ({args.test_type})")
    print(
        f"Initial accuracy ({args.match_type}): {100 * initial_match / total_count:.2f}%"
    )
    print(f"(matches: {initial_match}, non_matches: {total_count - initial_match})")
    if args.test_type == "OO":
        print(
            "Initial accuracy (on non-ambiguous subset by CC&SA): "
            f"{100 * initial_match / ambig_metrics['TN']:.2f}%"
        )
    print(f"Final accuracy ({args.match_type}): {100 * final_match / total_count:.2f}%")
    print(f"(matches: {final_match}, non_matches: {total_count - final_match})")
