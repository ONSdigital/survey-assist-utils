"""Functions to compare clerical codes with model codes."""

from survey_assist_utils.data_cleaning.sic_codes import (
    INVALID_VALUES,
)


def compare_codes(
    clerical_col: str | set[str] | list[str],
    model_col: str | set[str] | list[str],
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
        return compare_mo(clerical_col, model_col)
    if method == "MM":
        return compare_mm(clerical_col, model_col)
    raise ValueError(f"Invalid comparison method: {method}")


def compare_oo(
    clerical_col: str | set[str] | list[str], model_col: str | set[str] | list[str]
) -> bool:  # pylint: disable=C0103
    """Returns true where clerical coders and model agree exactly.
    Assumes cleaned input columns.
    Applicable to both 2-digit and 5-digit columns.
    If one is an empty string, returns False.
    """
    if isinstance(clerical_col, (set, list)):
        if len(clerical_col) != 1:
            return False
        clerical_col = next(iter(clerical_col))
    if isinstance(model_col, (set, list)):
        if len(model_col) != 1:
            return False
        model_col = next(iter(model_col))
    if not isinstance(clerical_col, str) or not isinstance(model_col, str):
        raise ValueError(
            "For 'OO' method, both clerical_col and model_col must be strings."
        )

    if (clerical_col in INVALID_VALUES) or (model_col in INVALID_VALUES):
        return False

    return clerical_col == model_col


def compare_om(
    clerical_col: str | set[str] | list[str], model_col: str | set[str] | list[str]
) -> bool:
    """Returns true where clerical coder choice is in the model's shortlist.
    Assumes cleaned input columns.
    Applicable to both 2-digit and 5-digit columns.
    If clerical code is an empty string, returns False.
    If the model's shortlist is empty, returns False.
    """
    if isinstance(clerical_col, (set, list)):
        if len(clerical_col) != 1:
            return False
        clerical_col = next(iter(clerical_col))
    if isinstance(model_col, str):
        model_col = [model_col]
    if not isinstance(clerical_col, str) or not isinstance(model_col, (set, list)):
        raise ValueError(
            "For 'OM' method, both clerical_col must be a string and model_col "
            "must be a set or list."
        )

    return clerical_col in model_col


def compare_mo(
    clerical_col: str | set[str] | list[str], model_col: str | set[str] | list[str]
) -> bool:
    """Returns true where any clerical coder option matches model choice.
    Assumes cleaned input columns.
    Applicable to both 2-digit and 5-digit columns.
    If clerical code list is empty, returns False.
    If the model's top choice is empty string, returns False.
    """
    if isinstance(clerical_col, str):
        clerical_col = set(clerical_col)
    if isinstance(model_col, (set, list)):
        if len(model_col) != 1:
            return False
        model_col = next(iter(model_col))
    if not isinstance(clerical_col, (set, list)) or not isinstance(model_col, str):
        raise ValueError(
            "For 'MO' method, both clerical_col must be a set or list and model_col "
            "must be a string."
        )

    return model_col in clerical_col


def compare_mm(
    clerical_col: str | set[str] | list[str], model_col: str | set[str] | list[str]
) -> bool:
    """Returns true where any clerical coder choice is in the model's shortlist.
    Assumes cleaned input columns.
    Applicable to both 2-digit and 5-digit columns.
    If either list is empty, returns False.
    """
    if isinstance(clerical_col, str):
        clerical_col = set(clerical_col)
    if isinstance(model_col, str):
        model_col = set(model_col)
    if not isinstance(clerical_col, (set, list)) or not isinstance(
        model_col, (set, list)
    ):
        raise ValueError(
            "For 'MM' method, both clerical_col and model_col must be sets or lists."
        )

    return bool(set(clerical_col) & set(model_col))
