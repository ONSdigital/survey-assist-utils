"""This module defines the `FlagGenerator` class, which enriches a raw evaluation
DataFrame with data quality and codability flags for downstream analysis.

The flags include:
- `num_answers`: Counts the number of valid SIC labels provided.
- `Match_5_digits`, `Match_3_digits`, `Match_2_digits`: Indicators of format conformity.
- `Unambiguous`: A boolean flag denoting whether a record has a single, well-formed SIC label.

These enhancements support filtering, validation, and interpretation of human-coded
classification data in the Survey Assist pipeline.
"""

import re

import pandas as pd

# --- Constants for Data Quality ---
_EXPECTED_SIC_LENGTH = 5
_X_COUNT_FOR_MATCH_3 = 2
_X_COUNT_FOR_MATCH_2 = 3


# pylint: disable=too-few-public-methods
# pylint: disable=duplicate-code
class FlagGenerator:
    """Adds data quality and codability flag columns to a DataFrame.

    This class takes a raw DataFrame containing human-coded SIC labels and
    enriches it with several analytical columns, including 'Unambiguous',
    'num_answers', and various format-matching flags.
    """

    def _calculate_num_answers(self, df: pd.DataFrame, cols: list[str]) -> pd.Series:
        """Calculates the number of provided answers in the given columns."""
        num_answers = pd.Series(0, index=df.index, dtype="int")
        for col in cols:
            if col in df.columns:
                is_valid = (
                    ~df[col].isna()
                    & (df[col].astype(str).str.strip() != "")
                    & (df[col].astype(str).str.upper() != "NA")
                )
                num_answers += is_valid.astype(int)
        return num_answers

    def _create_sic_match_flags(self, sic_series: pd.Series) -> dict[str, pd.Series]:
        """Calculates various SIC code format match flags for a given Series."""
        flags = {"Match_5_digits": sic_series.str.match(r"^[0-9]{5}$", na=False)}
        is_len_expected = sic_series.str.len() == _EXPECTED_SIC_LENGTH
        x_count = sic_series.str.count("x", re.I)
        only_digits_or_x = sic_series.str.match(r"^[0-9xX]*$", na=False)
        base_check = is_len_expected & only_digits_or_x
        flags["Match_3_digits"] = base_check & (x_count == _X_COUNT_FOR_MATCH_3)
        flags["Match_2_digits"] = base_check & (x_count == _X_COUNT_FOR_MATCH_2)
        return flags

    def add_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main method to add all data quality flag columns to the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with added flag columns.
        """
        df_out = df.copy()

        clerical_cols = [
            col
            for col in df.columns
            if col.startswith("sic_ind_occ") and col != "sic_ind_occ_flag"
        ]

        col_occ1 = clerical_cols[0] if clerical_cols else "sic_ind_occ1"

        # --- 1. Number of Answers ---
        df_out["num_answers"] = self._calculate_num_answers(df_out, clerical_cols)
        df_out.loc[df_out[col_occ1] == "-9", "num_answers"] = 0
        df_out.loc[df_out[col_occ1] == "4+", "num_answers"] = 4

        # --- 2. Digit/Character Match Flags ---
        s_occ1 = df_out[col_occ1].fillna("").astype(str)
        match_flags = self._create_sic_match_flags(s_occ1)
        for flag_name, flag_series in match_flags.items():
            df_out[flag_name] = flag_series

        # --- 3. Unambiguous Flag ---
        if "Match_5_digits" in df_out.columns:
            df_out["Unambiguous"] = (df_out["num_answers"] == 1) & (
                df_out["Match_5_digits"]
            )
        else:
            df_out["Unambiguous"] = False

        # Convert flag columns to a proper boolean type
        for flag_col in [
            "Match_5_digits",
            "Match_3_digits",
            "Match_2_digits",
            "Unambiguous",
        ]:
            if flag_col in df_out.columns:
                df_out[flag_col] = df_out[flag_col].astype("boolean")

        print("Successfully added data quality flag columns (e.g., 'Unambiguous').")
        return df_out
