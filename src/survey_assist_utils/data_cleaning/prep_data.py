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
    """Prepare and clean clerical SIC codes from one or two DataFrames.

    This function aggregates clerical codes from multiple columns into a single
    column, optionally merges additional codes from a secondary DataFrame (for
    "4+" cases), and cleans all codes to valid n-digit SIC codes. It also identifies
    invalid codes that cannot be cleaned.

    Args:
        df: Primary DataFrame containing clerical codes.
            Must include the unique identifier column (ID_COL) and up to three
            columns for clerical codes (e.g., sic_ind_occ1, sic_ind_occ2, sic_ind_occ3).
        df_four_plus: Optional DataFrame containing additional
            clerical codes for "4+" cases. If provided, codes from this DataFrame
            will be merged into the primary DataFrame. Defaults to None.
        clerical_col: Base name for clerical code columns in df.
            Defaults to "sic_ind_occ".
        out_col: Name of the output column that will contain cleaned clerical codes.
            Defaults to "clerical_codes".
        digits (int): Number of digits to which SIC codes should be cleaned or expanded.
            Defaults to 5.

    Returns:
        pd.DataFrame: A DataFrame containing:
            - ID_COL: Unique identifier.
            - out_col: Set of cleaned SIC codes.
            - invalid_codes: Set of original codes that could not be cleaned.

    Raises:
        ValueError: If the input DataFrame is missing the required unique identifier column.
    """
    # Set a dynamic invalid col name:
    invalid_col = f"{out_col}_invalid"
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

    # Added a check for illegal codes,
    df["out_col_temp"] = df[clerical_col].apply(parse_numerical_code)

    # But first, check if the entire df is empty:
    if df.empty:
        # create the return df
        df[out_col] = pd.Series([], dtype=object)
        df[invalid_col] = pd.Series([], dtype=object)
    else:
        # Only run apply if we have a df
        df[[out_col, invalid_col]] = df["out_col_temp"].apply(
            lambda x: pd.Series(get_clean_n_digit_codes(x, n=digits))
        )

    return df[[ID_COL, out_col, invalid_col]]


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
    """Prepare the input DataFrame containing model-predicted SIC codes.

    Cleans codes to valid n-digit SIC codes and identifies invalid codes.
    Optionally extracts alternative candidate codes if the primary code is missing.

    Args:
        input_df: Input DataFrame to be prepared.
        codes_col: Column name for initial model predicted code.
        alt_codes_col: Column name for alternative codes (list of dicts).
        out_col: Column name for the output cleaned model codes.
        alt_codes_name: Key name to extract codes from alternative predictions.
        threshold: Likelihood threshold for pruning alternative candidates.
        digits: Number of digits to which SIC codes should be cleaned.

    Returns:
        A DataFrame containing:
            - ID_COL: Unique identifier.
            - out_col: Set of cleaned model codes.
            - invalid_codes: Set of original codes that could not be cleaned.

    Raises:
        ValueError: If required columns are missing in the input DataFrame.
    """
    # Set a dynamic invalid col name:
    invalid_col = f"{out_col}_invalid"
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

    # Initialise MAIN output column
    out_df = input_df[[ID_COL]].copy()
    out_df[out_col] = [{} for _ in range(len(input_df))]

    # Initialise INVALID column (This prevents the KeyError when codes_col is None)
    out_df[invalid_col] = [set() for _ in range(len(input_df))]

    if codes_col:
        # Make a temp col for the partially cleaned codes:
        out_df["out_col_temp"] = input_df[codes_col].apply(parse_numerical_code)

        out_df[[out_col, invalid_col]] = out_df["out_col_temp"].apply(
            lambda x: pd.Series(get_clean_n_digit_codes(x, n=digits))
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
        # Note: We are NOT extracting invalid codes from alternatives here

    return out_df[[ID_COL, out_col, invalid_col]]
