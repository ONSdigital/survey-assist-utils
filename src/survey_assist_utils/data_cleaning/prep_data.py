"""Read clerical data from standard clerical format."""

import logging

import pandas as pd

from survey_assist_utils.data_cleaning.sic_codes import (
    extract_alt_candidates_n_digit_codes,
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


# allow more arguments than 5
# pylint: disable=R0913, R0917
def prep_model_codes(  # noqa:PLR0913
    input_df: pd.DataFrame,
    codes_col: str | None = "initial_code",
    alt_codes_col: str | None = "alt_sic_candidates",
    out_col: str = "model_codes",
    alt_codes_name: str = "code",
    threshold: float = 0,
    digits: int = 5,
) -> pd.DataFrame:
    """Prepares the input DataFrame for evaluation by ensuring necessary columns exist.

    Args:
        input_df: Input DataFrame to be prepared.
        codes_col: Column name for initial model predicted code (string).
        alt_codes_col: Column name for alternative codes (list of dicts).
        out_col: Column name for the output cleaned model codes.
        alt_codes_name: Key name to extract codes from alternative predictions.
        threshold: Likelihood threshold for pruning alternative candidates.
        digits: Number of digits to which SIC codes should be cleaned/expanded.

    Returns:
        Prepared DataFrame with necessary columns.

    Raises:
        ValueError: If required columns are missing in the input DataFrame.
    """
    if ID_COL not in input_df.columns:
        raise ValueError(f"Input DataFrame must contain a column '{ID_COL}'")
    if codes_col not in input_df.columns:
        codes_col = None
    if alt_codes_col not in input_df.columns:
        alt_codes_col = None

    if codes_col is None and alt_codes_col is None:
        raise ValueError(
            "At least one of 'codes_col' or 'alt_codes_col' must be provided."
        )

    out_df = input_df[[ID_COL]].copy()
    out_df[out_col] = [{} for _ in range(len(input_df))]
    if codes_col is not None:
        out_df[out_col] = (
            input_df[codes_col]
            .apply(parse_numerical_code)
            .apply(get_clean_n_digit_codes, n=digits)
        )

    # Extract the codes from the model's alt_sic_candidates if ambiguous
    if alt_codes_col is not None:
        miss_msk = out_df[out_col].apply(lambda x: not x)
        logger.info(
            "Filling initial codes from alternatives for %d rows.",
            miss_msk.sum(),
        )
        alternatives = input_df.loc[miss_msk, alt_codes_col].apply(
            extract_alt_candidates_n_digit_codes,
            code_name=alt_codes_name,
            n=digits,
            threshold=threshold,
        )
        out_df.loc[miss_msk, out_col] = alternatives

    return out_df
