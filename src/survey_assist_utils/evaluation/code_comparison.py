"""Functions to compare clerical codes with model codes."""

from collections.abc import Iterable

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
    if method == "OO":
        return compare_oo(clerical_col, model_col)
    if method == "OM":
        return compare_om(clerical_col, model_col)
    if method == "MO":
        # invert order of arguments to avoid code duplication
        return compare_om(clerical_col=model_col, model_col=clerical_col)
    if method == "MM":
        return compare_mm(clerical_col, model_col)
    raise ValueError(f"Invalid comparison method: {method}")


def cast_code_to_set(
    input_data: str | Iterable[str] | None,
) -> set[str]:
    """Cast input codes to a set of strings."""
    if input_data is None:
        return set()
    if isinstance(input_data, str) or not isinstance(input_data, Iterable):
        input_data = {input_data}
    return {str(x) for x in input_data}.difference(INVALID_VALUES)


def cast_code_to_str(
    input_data: str | Iterable[str] | None,
) -> str | None:
    """Cast input codes to a string if unique code is presented.

    Args:
        input_data: Input data which can be a string, an iterable of strings.

    Returns:
        A single string if input_data contains exactly one valid code string,
        None otherwise.
    """
    # convert to set to enable len calculation and to remove duplicates
    input_set = cast_code_to_set(input_data)

    return next(iter(input_set)) if len(input_set) == 1 else None


def compare_oo(
    clerical_col: str | Iterable[str] | None, model_col: str | Iterable[str] | None
) -> bool:  # pylint: disable=C0103
    """Returns true where clerical coders and model agree exactly.
    Assumes cleaned input columns.
    Applicable to both 2-digit and 5-digit columns.
    If one is an empty string, returns False.
    """
    clerical_col = cast_code_to_str(clerical_col)
    model_col = cast_code_to_str(model_col)

    if (clerical_col in INVALID_VALUES) or (model_col in INVALID_VALUES):
        return False

    return clerical_col == model_col


def compare_om(
    clerical_col: str | Iterable[str] | None, model_col: str | Iterable[str] | None
) -> bool:
    """Returns true where clerical coder choice is in the model's shortlist.
    Assumes cleaned input columns.
    Applicable to both 2-digit and 5-digit columns.
    If clerical code is an empty string, returns False.
    If the model's shortlist is empty, returns False.
    """
    clerical_col = cast_code_to_str(clerical_col)
    model_col = cast_code_to_set(model_col)
    if clerical_col in INVALID_VALUES:
        return False

    return clerical_col in model_col


def compare_mm(
    clerical_col: str | Iterable[str] | None, model_col: str | Iterable[str] | None
) -> bool:
    """Returns true where any clerical coder choice is in the model's shortlist.
    Assumes cleaned input columns.
    Applicable to both 2-digit and 5-digit columns.
    If either list is empty, returns False.
    """
    clerical_col = cast_code_to_set(clerical_col)
    model_col = cast_code_to_set(model_col)

    return bool(clerical_col & model_col)
