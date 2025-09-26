"""Functions to compare clerical codes with model codes."""

from collections.abc import Iterable

from survey_assist_utils.data_cleaning.sic_codes import INVALID_VALUES


def cast_code_to_set(
    input_data: str | Iterable[str] | None,
) -> set[str]:
    """Cast input codes to a set of strings.

    Args:
        input_data: The input code(s) to cast. Can be a single string,
            an iterable of strings, or None.

    Returns:
        A set of strings representing the input codes, with invalid values removed.
    """
    if input_data is None:
        return set()
    if isinstance(input_data, str) or not isinstance(input_data, Iterable):
        input_data = {input_data}
    return {str(x) for x in input_data}.difference(INVALID_VALUES)


def compare_codes(
    clerical_col: str | Iterable[str] | None,
    model_col: str | Iterable[str] | None,
    method: str = "MM",
) -> bool:
    """Compare clerical and model codes using desired comparison method.

    Args:
        clerical_col: The clerical code(s) to compare.
        model_col: The model code(s) to compare.
        method: The comparison method to use. One of 'OO', 'OM',
            'MO', 'MM'. Defaults to 'OO'.

    Returns:
        bool: True if the codes match according to the method, False otherwise.

    Raises:
        ValueError: If an invalid comparison method is provided.
    """
    clerical_col = cast_code_to_set(clerical_col)
    model_col = cast_code_to_set(model_col)

    return _compare_codes(clerical_col, model_col, method)


def _compare_codes(
    clerical_col: set[str],
    model_col: set[str],
    method: str,
) -> bool:
    """Compare clerical and model codes using desired comparison method.

    This is a private function that assumes inputs are already cast to sets.

    Args:
        clerical_col: The clerical code(s) to compare.
        model_col: The model code(s) to compare.
        method: The comparison method to use. One of 'OO', 'OM',
            'MO', 'MM'. Defaults to 'OO'.

    Returns:
        bool: True if the codes match according to the method, False otherwise.

    Raises:
        ValueError: If an invalid comparison method is provided.
    """
    overlap = bool(clerical_col & model_col)
    single_clerical = len(clerical_col) == 1
    single_model = len(model_col) == 1

    if method == "OO":
        return single_clerical and single_model and overlap
    if method == "OM":
        return single_clerical and overlap
    if method == "MO":
        return single_model and overlap
    if method == "MM":
        return overlap
    raise ValueError(f"Invalid comparison method: {method}")
