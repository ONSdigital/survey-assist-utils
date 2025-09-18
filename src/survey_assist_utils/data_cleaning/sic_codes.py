"""Helper functions for cleaning sic code data before evaluation."""

import re

import pandas as pd

INVALID_VALUES = (("-9", "4+", "", None, pd.NA, "NAN", "NaN", "nan"),)


def parse_clerical_code(candidates_str: str):
    """Converts the clerical coder responses from a
    stringified list to a proper list of strings.
    """
    candidates_str = str(candidates_str).strip()
    if candidates_str in INVALID_VALUES:
        return []

    try:
        # remove -9 and 4+ from the string
        candidates_str = candidates_str.replace("-9", "").replace("4+", "")
        # Extract all RagCandidate entries using regex
        pattern = r"([0-9]+x*X*)"
        matches = re.findall(pattern, candidates_str)

        return matches
    except Exception as e:
        raise ValueError(
            f"Error parsing clerical codes: {candidates_str} \n {e}"
        ) from e


def expand_to_n_digit_str(input_str: str, n: int) -> set[str]:
    """Return set of codes in the hierarchy expanded to n digits.
    !!NOT IMPLEMENTED YET FOR REAL HIERARCHY!!
    For now it returns all numerically posssible subcodes.
    E.g. '86' -> {'86000', '86100', ..., '86999'} for n=5.
    """
    fill_digits = n - len(input_str)

    return {input_str + str(x).zfill(fill_digits) for x in range(10**fill_digits)}


def get_clean_n_digit_one_code(input_str: str, n: int) -> set[str]:
    """Converts a n-digit string to either a valid SIC code format
    or an empty string. E.g. '86011' -> '86011'; '86xxx' -> ''.
    """
    # cut x's from the back if they are there
    input_str = input_str.rstrip("xX")
    # check the rest is numeric
    if not input_str.isdigit():
        return set()

    if len(input_str) < n:
        return expand_to_n_digit_str(input_str, n)

    return {input_str[:n]}


def get_clean_n_digit_codes(input_list: str | set[str] | list[str], n: int) -> set[str]:
    """Converts a list of possible codes to a list containing only
    valid n-digit SIC codes.
    E.g. ['86011', '86012', '85xxx'] -> ['86011', '86012'].
    """
    if isinstance(input_list, str):
        input_list = [input_list]
    if not isinstance(input_list, (set, list)):
        raise ValueError("input_list must be a list of strings.")

    cleaned_list = [get_clean_n_digit_one_code(i, n) for i in input_list]
    # Flatten the sets and deduplicate
    pruned_list = set().union(*cleaned_list)
    return pruned_list
