"""Read clerical data from standard clerical format."""

import logging

import pandas as pd

from survey_assist_utils.data_cleaning.sic_codes import (
    INVALID_VALUES,
    extract_alt_sic_candidates,
    get_clean_n_digit_codes,
    parse_numerical_code,
)

logger = logging.getLogger(__name__)

ID_COL = "unique_id"


def prep_clerical_codes(
    df: pd.DataFrame,
    df_four_plus: pd.DataFrame | None = None,
    clerical_col: str = "sic_ind_occ",
    out_col: str = "clerical_codes",
    digits: int = 5,
) -> pd.DataFrame:
    """Extract and clean clerical codes from the DataFrame.

    Args:
        df: Input DataFrame containing clerical codes.
        df_four_plus: Optional DataFrame containing clerical codes for '4+' cases.
            If None no extra codes are expected. Defaults to None.
        clerical_col: Column name where clerical codes are stored.
            Defaults to "sic_ind_occ".
        out_col: Column name for the output cleaned clerical codes.
            Defaults to "clerical_codes"
        digits: Number of digits to which SIC codes should be cleaned/expanded.
            Defaults to 5.

    Returns:
        DataFrame with cleaned clerical codes.
    """
    clerical_3cols = [clerical_col + str(i) for i in range(1, 4)]

    df = df[[ID_COL, *clerical_3cols]].copy()
    df[clerical_col] = df[clerical_3cols].agg(
        lambda x: ";".join(x.dropna().astype(str)), axis=1
    )
    if df_four_plus is not None:
        # Merge the two DataFrames on the unique identifier
        df = df.merge(
            df_four_plus[[ID_COL, clerical_col]].copy(),
            on=ID_COL,
            how="outer",
            suffixes=("", "_4plus"),
        )
        msk = df[f"{clerical_col}_4plus"].notna()
        logging.info(
            "Merging clerical codes from '4+' DataFrame for %d entries.", msk.sum()
        )
        df.loc[msk, clerical_col] = df.loc[msk, f"{clerical_col}_4plus"]

    df[out_col] = (
        df[clerical_col]
        .apply(parse_numerical_code)
        .apply(get_clean_n_digit_codes, n=digits)
    )

    return df[[ID_COL, out_col]]


def prep_model_dataframe(
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
        ID_COL,
        initial_code_col,
        initial_alt_codes_col,
    ] + ([final_sic] if final_sic else [])
    if miss := set(required_columns) - set(input_df.columns):
        raise ValueError(f"Input DataFrame is missing required columns: {miss}")
    input_df = input_df[required_columns].copy()

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
