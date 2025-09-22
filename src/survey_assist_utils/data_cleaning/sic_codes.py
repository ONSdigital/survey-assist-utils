"""Helper functions for cleaning sic code data before evaluation."""

import logging
import re
from collections.abc import Iterable

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
    """
    fill_digits = n - len(input_str)

    return {input_str + str(x).zfill(fill_digits) for x in range(10**fill_digits)}


def get_clean_n_digit_one_code(input_str: str, n: int) -> set[str]:
    """Converts a n-digit string to either a valid SIC code format
    or an empty string. E.g. '860112' -> {'86011'}; '86xxx' -> {'86000', ..., '86999'}.
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
    E.g. ['86011', '86012', '85xxx'] -> ['86011', '86012', '85000', ..., '85999'].
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
    pruned_list = set().union(*cleaned_list)
    return pruned_list


def extract_alt_sic_candidates(
    alt_candidates: list[dict],
    code_name: str,
    score_name="likelihood",
    threshold: float = 0,
) -> list[str]:
    """Extracts alternative sic codes from the model predictions
    and prunes them based on the threshold (i.e. if there is one entry
    with score above the threshold, keep only that one.).

    If threshold is 0 or negative, no pruning is done.

    Args:
        alt_candidates: List of alternative candidate dictionaries.
        code_name: Key name to extract codes from alternative predictions.
        score_name: Key name to extract score from alternative predictions.
        threshold: Score threshold for pruning alternative candidates.
    """
    if not isinstance(alt_candidates, Iterable):
        logger.warning(
            "Expected a list of dicts for alt_candidates, got %s", type(alt_candidates)
        )
        return []

    if threshold >= 0:
        pruned = [x for x in alt_candidates if x.get(score_name, 0) >= threshold]
        if len(pruned) == 1:
            alt_candidates = pruned

    return [x.get(code_name, "-9") for x in alt_candidates]
