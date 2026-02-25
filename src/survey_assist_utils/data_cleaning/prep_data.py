"""Read clerical data from standard clerical format.
Cleans and prepares clerical and model SIC codes for further processing.
"""

import logging

import pandas as pd

# Assuming these are imported from your utils
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
    if ID_COL not in df.columns:
        raise ValueError(f"Input DataFrame must contain a column '{ID_COL}'")
    invalid_col = f"{out_col}_invalid"
    if df.empty:
        df[out_col] = pd.Series([], dtype=object)
        df[invalid_col] = pd.Series([], dtype=object)
        return df[[ID_COL, out_col, invalid_col]]

    clerical_3cols = list(
        {clerical_col + str(i) for i in range(1, 4)}.intersection(df.columns)
    )
    if not clerical_3cols:
        raise ValueError(
            f"Input DataFrame must contain at least one of the clerical code columns: "
            f"{', '.join([clerical_col + str(i) for i in range(1, 4)])}"
        )

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

    df[[out_col, invalid_col]] = (
        df[clerical_col]
        .apply(parse_numerical_code)
        .apply(lambda x: pd.Series(get_clean_n_digit_codes(x, n=digits)))
    )

    return df[[ID_COL, out_col, invalid_col]]


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
    This function hasd been overloaded to accept either individual parameters
    or a single configuration dataclass.

    Cleans codes to valid n-digit SIC codes and identifies invalid codes.
    Optionally extracts alternative candidate codes if the primary code is missing.

    Args: (legacy style)
        input_df: Input DataFrame to be prepared.
        codes_col: Column name for initial model predicted code.
        alt_codes_col: Column name for alternative codes (list of dicts).
        out_col: Column name for the output cleaned model codes.
        alt_codes_name: Key name to extract codes from alternative predictions.
        threshold: Likelihood threshold for pruning alternative candidates.
        digits: Number of digits to which SIC codes should be cleaned.

    Args: (config style)
        input_df: Input DataFrame to be prepared.
        codes_col: ModelPrepConfig dataclass containing all configuration.

    Returns:
        A DataFrame containing:
            - ID_COL: Unique identifier.
            - out_col: Set of cleaned model codes.
            - invalid_codes: Set of original codes that could not be cleaned.

    Raises:
        ValueError: If required columns are missing in the input DataFrame.

    """
    if ID_COL not in input_df.columns:
        raise ValueError(f"Input DataFrame must contain a column '{ID_COL}'")

    invalid_col = f"{out_col}_invalid"
    if input_df.empty:
        out_df = input_df[[ID_COL]].copy()
        out_df[out_col] = pd.Series([], dtype=object)
        out_df[invalid_col] = pd.Series([], dtype=object)
        return out_df

    if codes_col not in input_df.columns:
        codes_col = None
    if alt_codes_col not in input_df.columns:
        alt_codes_col = None
    if codes_col is None and alt_codes_col is None:
        raise ValueError(
            "At least one of 'codes_col' or 'alt_codes_col' must be provided."
        )
    out_df = input_df[[ID_COL]].copy()
    out_df[out_col] = [set() for _ in range(len(input_df))]
    out_df[invalid_col] = [set() for _ in range(len(input_df))]

    if codes_col is not None:
        out_df[[out_col, invalid_col]] = (
            input_df[codes_col]
            .apply(parse_numerical_code)
            .apply(lambda x: pd.Series(get_clean_n_digit_codes(x, n=digits)))
        )

    if alt_codes_col is not None:
        miss_msk = out_df[out_col].apply(lambda x: not x)
        logger.info(
            "Filling initial codes from alternatives for %d rows.",
            miss_msk.sum(),
        )

        alternatives = input_df.loc[miss_msk, alt_codes_col].apply(
            lambda x: pd.Series(
                extract_alt_candidates_n_digit_codes(
                    x,
                    code_name=alt_codes_name,
                    n=digits,
                    threshold=threshold,
                ),
                index=[out_col, invalid_col],
            )
        )

        out_df.loc[miss_msk, out_col] = alternatives[out_col]
        out_df.loc[miss_msk, invalid_col] = out_df.loc[miss_msk, invalid_col].combine(
            alternatives[invalid_col],
            lambda existing, new: (existing or set()) | (new or set()),
        )

    return out_df[[ID_COL, out_col, invalid_col]]
