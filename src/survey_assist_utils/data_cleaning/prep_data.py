"""Read clerical data from standard clerical format.
Cleans and prepares clerical and model SIC codes for further processing.
"""

import logging
from dataclasses import dataclass
from typing import overload

import pandas as pd

# Assuming these are imported from your utils
from survey_assist_utils.data_cleaning.sic_codes import (
    extract_alt_candidates_n_digit_codes,
    get_clean_n_digit_codes,
    parse_numerical_code,
)

logger = logging.getLogger(__name__)
ID_COL = "unique_id"


@dataclass
class ModelPrepConfig:
    """Configuration for preparing model codes during data cleaning."""

    codes_col: str | None = "initial_code"
    alt_codes_col: str | None = "alt_sic_candidates"
    out_col: str = "model_codes"
    alt_codes_name: str = "code"
    threshold: float = 0
    digits: int = 5

    @property
    def invalid_col(self) -> str:
        """Name of the column containing invalid or uncleanable codes."""
        return f"{self.out_col}_invalid"


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


# pylint: disable=too-many-arguments
# pylint: disable=R0917
def _resolve_config(  # noqa:PLR0913
    input_df: pd.DataFrame,
    codes_col: str | None | ModelPrepConfig,
    alt_codes_col: str | None,
    out_col: str,
    alt_codes_name: str,
    threshold: float,
    digits: int,
) -> ModelPrepConfig:
    """Normalizes arguments into a config object and validates columns."""
    if isinstance(codes_col, ModelPrepConfig):
        cfg = codes_col
    else:
        cfg = ModelPrepConfig(
            codes_col=codes_col,
            alt_codes_col=alt_codes_col,
            out_col=out_col,
            alt_codes_name=alt_codes_name,
            threshold=threshold,
            digits=digits,
        )

    if ID_COL not in input_df.columns:
        raise ValueError(f"Input DataFrame must contain a column '{ID_COL}'")

    # Soft validation: set to None if column missing in DF
    if cfg.codes_col and cfg.codes_col not in input_df.columns:
        cfg.codes_col = None
    if cfg.alt_codes_col and cfg.alt_codes_col not in input_df.columns:
        cfg.alt_codes_col = None

    if cfg.codes_col is None and cfg.alt_codes_col is None:
        raise ValueError(
            "At least one of 'codes_col' or 'alt_codes_col' must be provided."
        )

    return cfg


def _process_primary_codes(df: pd.DataFrame, cfg: ModelPrepConfig) -> pd.DataFrame:
    """Initialize output DataFrame and process primary codes if available."""
    # Initialize output structure
    out_df = df[[ID_COL]].copy()
    out_df[cfg.out_col] = [set() for _ in range(len(df))]
    out_df[cfg.invalid_col] = [set() for _ in range(len(df))]

    if cfg.codes_col:
        # Process primary codes
        temp_parsed = df[cfg.codes_col].apply(parse_numerical_code)

        cleaned_results = temp_parsed.apply(
            lambda x: pd.Series(get_clean_n_digit_codes(x, n=cfg.digits))
        )

        # Assign results to the correct columns
        out_df[[cfg.out_col, cfg.invalid_col]] = cleaned_results

    return out_df


def _fill_missing_from_alternatives(
    out_df: pd.DataFrame, input_df: pd.DataFrame, cfg: ModelPrepConfig
) -> pd.DataFrame:
    """Fills missing codes in out_df using alternatives from input_df."""
    # 1. Quick Exit if no alt column
    if not cfg.alt_codes_col:
        return out_df

    # 2. Identify rows that need filling (empty set in out_col)
    miss_msk = out_df[cfg.out_col].apply(lambda x: not x)
    df_to_fill = input_df.loc[miss_msk]

    # 3. Quick Exit if nothing to fill
    if df_to_fill.empty:
        return out_df

    logger.info("Filling initial codes from alternatives for %d rows.", len(df_to_fill))

    # 4. Extract Alternatives
    alt_extracted = df_to_fill[cfg.alt_codes_col].apply(
        lambda x: pd.Series(
            extract_alt_candidates_n_digit_codes(
                x,
                code_name=cfg.alt_codes_name,
                n=cfg.digits,
                threshold=cfg.threshold,
            ),
            index=[cfg.out_col, cfg.invalid_col],
        )
    )

    # 5. Assign Valid Codes (Column 0)
    out_df.loc[miss_msk, cfg.out_col] = alt_extracted[cfg.out_col]

    out_df.loc[miss_msk, cfg.invalid_col] = out_df.loc[
        miss_msk, cfg.invalid_col
    ].combine(
        alt_extracted[cfg.invalid_col],
        lambda existing, new: (existing or set()) | (new or set()),
    )

    return out_df


# ---------------------------------------------------------
# Main Function
# ---------------------------------------------------------


@overload
def prep_model_codes(
    input_df: pd.DataFrame, codes_col: ModelPrepConfig
) -> pd.DataFrame: ...


@overload
def prep_model_codes(
    input_df: pd.DataFrame,
    codes_col: str | None = "initial_code",
    alt_codes_col: str | None = "alt_sic_candidates",
    out_col: str = "model_codes",
    alt_codes_name: str = "code",
    threshold: float = 0,
    digits: int = 5,
) -> pd.DataFrame: ...


# pylint: disable=too-many-arguments
def prep_model_codes(  # noqa:PLR0913
    input_df: pd.DataFrame,
    codes_col: str | None | ModelPrepConfig = "initial_code",
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
    # 1. Resolve Configuration
    cfg = _resolve_config(
        input_df, codes_col, alt_codes_col, out_col, alt_codes_name, threshold, digits
    )

    # 2. Process Primary Codes (Creates the base Output DF)
    out_df = _process_primary_codes(input_df, cfg)

    # 3. Fill Gaps with Alternatives (Updates Output DF in place/returns it)
    out_df = _fill_missing_from_alternatives(out_df, input_df, cfg)

    # 4. Return Final Result
    return out_df[[ID_COL, cfg.out_col, cfg.invalid_col]]
