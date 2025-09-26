"""Helper functions for cleaning sic code data before evaluation."""

import logging
import re
from collections.abc import Iterable

from survey_assist_utils.data_cleaning.sic_code_section_list import (
    SECTION_LOOKUP,
    VALID_SIC_CODES,
)

logger = logging.getLogger(__name__)

INVALID_VALUES = (
    "-9",
    "4+",
    "",
    ".",
    " ",
    None,
    "NAN",
    "NaN",
    "nan",
    "None",
    "Null",
    "<NA>",
)

EXPECTED_CODE_LENGTH = 5

SIC_REGEX_PATTERN = r"([0-9]+x*X*)"


def parse_numerical_code(
    candidates_str: str,
    code_regex_pattern: str = SIC_REGEX_PATTERN,
    padding: int = EXPECTED_CODE_LENGTH,
) -> set[str]:
    """Converts the clerical coder responses from a
    stringified list to a proper list of strings.
    E.g. "[8601x, 1410, nan]" -> {'8601x', '01410'}.

    Args:
        candidates_str: String containing the clerical coder responses.
        code_regex_pattern: Regex pattern to extract codes from the string.
        padding: Number of digits to which the codes should be zero-padded.

    Returns:
        List of cleaned and zero-padded code strings.

    """
    candidates_str = str(candidates_str).strip()
    if candidates_str in INVALID_VALUES:
        return set()
    try:
        # remove -9 and 4+ from the string
        candidates_str = candidates_str.replace("-9", "").replace("4+", "")
        # Extract all RagCandidate entries using regex
        matches = re.findall(code_regex_pattern, candidates_str)
        return {matches.zfill(padding) for matches in matches}
    except re.error as e:
        logger.warning("Error parsing numerical codes: %s \n %s", candidates_str, e)
        return set()


def expand_to_n_digit_str(input_str: str, n: int) -> set[str]:
    """Return set of codes in the hierarchy expanded to n digits.
    !!NOT IMPLEMENTED YET FOR REAL HIERARCHY!!
    For now it returns all numerically posssible subcodes.
    E.g. '86' -> {'86000', '86100', ..., '86999'} for n=5.

    Args:
        input_str: String containing a possible code.
        n: Number of digits to which the code should be expanded.

    Returns:
        Set of expanded n-digit SIC code strings.
    """
    if n <= len(input_str):
        return {input_str}

    fill_digits = n - len(input_str)

    return {input_str + str(x).zfill(fill_digits) for x in range(10**fill_digits)}


def get_clean_n_digit_one_code(input_str: str, n: int) -> set[str]:
    """Converts a n-digit string to either a valid SIC code format
    or an empty string. E.g. '860112' -> {'86011'}; '86xxx' -> {'86000', ..., '86999'}.

    Args:
        input_str: String containing a possible code.
        n: Number of digits to which the code should be cleaned/expanded.

    Returns:
        Set of cleaned n-digit SIC code strings.
    """
    # cut x's from the back if they are there
    input_str = str(input_str).rstrip("xX")
    # check the rest is numeric
    if not input_str.isdigit():
        return set()

    n_digits = n if n > 0 else 2  # for section assignment

    prep_set = (
        {input_str[:n_digits]}
        if len(input_str) >= n_digits
        else expand_to_n_digit_str(input_str, n_digits)
    )

    if n == 0:
        prep_set = {
            SECTION_LOOKUP.get(code, "") for code in prep_set if code in SECTION_LOOKUP
        }

    return validate_sic_codes(prep_set)


def get_clean_n_digit_codes(input_list: str | set[str] | list[str], n: int) -> set[str]:
    """Converts a list of possible codes to a list containing only
    valid n-digit SIC codes.
    E.g. ['86011', '86012', '85xxx'] -> ['86011', '86012', '85000', ..., '85999'].

    Args:
        input_list: List or set of strings containing possible codes.
        n: Number of digits to which the codes should be cleaned/expanded.

    Returns:
        Set of cleaned n-digit SIC code strings.
    """
    if isinstance(input_list, str):
        input_list = [input_list]
    if not isinstance(input_list, (set, list)):
        logger.warning(
            "Expected a list or set of strings for input_list, got %s", type(input_list)
        )
        return set()

    cleaned_list = [get_clean_n_digit_one_code(i, n) for i in input_list]
    # Flatten the sets and deduplicate
    cleaned_set = set().union(*cleaned_list)

    return cleaned_set


def validate_sic_codes(input_set: str | set[str] | list[str]) -> set[str]:
    """Validate if the input code is a valid SIC code.

    Args:
        input_set: A string representing the SIC code to validate.

    Returns:
        A set of valid SIC codes.
    """
    if isinstance(input_set, str):
        input_set = {input_set}
    if not isinstance(input_set, (set, list)):
        logger.warning(
            "Expected a list or set of strings for input_list, got %s", type(input_set)
        )
        return set()
    return {str(x) for x in input_set}.intersection(VALID_SIC_CODES)


def extract_alt_candidates_n_digit_codes(
    alt_candidates: list[dict],
    code_name: str,
    n: int = EXPECTED_CODE_LENGTH,
    score_name: str = "likelihood",
    threshold: float = 0,
) -> set[str]:
    """Extracts alternative sic codes from the model predictions
    and prunes them based on the threshold (i.e. if there is one entry
    with score above the threshold, keep only that one.).

    If threshold is 0 or negative, no pruning is done.

    Args:
        alt_candidates: List of alternative candidate dictionaries.
        code_name: Key name to extract codes from alternative predictions.
        n: Number of digits to which the codes should be considered equivalent for pruning.
        score_name: Key name to extract score from alternative predictions.
        threshold: Score threshold for pruning alternative candidates.

    Returns:
        List of extracted and pruned alternative sic code strings.
    """
    if isinstance(alt_candidates, str):
        return get_clean_n_digit_codes(parse_numerical_code(alt_candidates), n)

    if not isinstance(alt_candidates, Iterable):
        logger.warning(
            "Expected a list of dicts for alt_candidates, got %s", type(alt_candidates)
        )
        return {}

    cleaned: dict[str, float] = {}
    for item in alt_candidates:
        codes = get_clean_n_digit_one_code(item.get(code_name, None), n)
        score = item.get(score_name, 0)
        for code in codes:
            if code in cleaned:
                cleaned[code] = max(cleaned[code], score)
            else:
                cleaned[code] = score

    pruned = {code for code, score in cleaned.items() if score >= threshold}
    if len(pruned) == 1:
        return pruned

    return set(cleaned)
