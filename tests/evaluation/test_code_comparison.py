"""Tests for code comparison functions."""

import pytest

from survey_assist_utils.evaluation.code_comparison import (
    INVALID_VALUES,
    cast_code_to_set,
    cast_code_to_str,
    compare_codes,
    compare_mm,
    compare_om,
    compare_oo,
)


def test_cast_code_to_set():
    """Test casting various code inputs to set of strings."""
    assert cast_code_to_set(None) == set(), "None input"
    assert cast_code_to_set(-9) == set(), "Invalid integer input"
    assert cast_code_to_set(INVALID_VALUES) == set(), "Invalid values input"
    assert cast_code_to_set("") == set(), "Empty string input"
    assert cast_code_to_set("86011") == {"86011"}, "Single valid string input"
    assert cast_code_to_set(["86011", "86012"]) == {
        "86011",
        "86012",
    }, "List of valid strings"
    assert cast_code_to_set(["86011", "-9", None, ""]) == {
        "86011"
    }, "List with invalid values"
    assert cast_code_to_set({"86011", "86012"}) == {
        "86011",
        "86012",
    }, "Set of valid strings"
    assert cast_code_to_set(range(86000, 86003)) == {
        "86000",
        "86001",
        "86002",
    }, "Iterable of integers"


def test_cast_code_to_str():
    """Test casting various code inputs to a single string."""
    assert cast_code_to_str(None) is None, "None input"
    assert cast_code_to_str("") is None, "Empty string input"
    assert cast_code_to_str(-9) is None, "Invalid integer -9 input"
    assert cast_code_to_str(43) == "43", "Invalid string '43' input"
    assert cast_code_to_str("86011") == "86011", "Single valid string input"
    assert cast_code_to_str(["86011"]) == "86011", "List with single valid string"
    assert cast_code_to_str({"86011"}) == "86011", "Set with single valid string"
    assert cast_code_to_str(["86011", "86012"]) is None, "List with multiple strings"
    assert cast_code_to_str({"86011", "86012"}) is None, "Set with multiple strings"
    assert (
        cast_code_to_str(range(86000, 86003)) is None
    ), "Iterable with multiple integers"


def test_compare_oo_exact_match():
    """Test the OO comparison method for exact matches."""
    assert compare_oo("86011", "86011") is True, "OO exact match"
    assert compare_oo(["86011"], ["86011"]) is True, "OO exact match (list)"
    assert compare_oo({"86011"}, {"86011"}) is True, "OO exact match (set)"
    assert compare_oo("86011", "86012") is False, "OO different codes"
    assert compare_oo(["86011"], ["86012"]) is False, "OO different codes (list)"
    assert (
        compare_oo(["86011", "86012"], ["86011"]) is False
    ), "OO different codes (list)"
    assert compare_oo("", "86011") is False


def test_compare_om_in_shortlist():
    """Test the OM comparison method for matches in shortlist."""
    assert compare_om("86011", ["86011", "86012"]) is True, "OM exact match in list"
    assert compare_om("86012", ["86012"]) is True, "OM exact match in list (single)"
    assert compare_om("86013", ["86011", "86012"]) is False, "OM no match in list"
    assert (
        compare_om(["86011"], ["86011", "86012"]) is True
    ), "OM exact match in list (single)"
    assert (
        compare_om(["86013"], ["86011", "86012"]) is False
    ), "OM no match in list (single)"
    assert (
        compare_om(["86011", "86012"], ["86011", "86012"]) is False
    ), "OM exact match in list (multiple)"
    assert compare_om([], ["86011"]) is False, "OM no match in list (empty)"


def test_compare_mm_any_in_both():
    """Test the MM comparison method for any matches in both sets."""
    assert (
        compare_mm(["86011", "86012"], ["86012", "86013"]) is True
    ), "MM overlapping lists"
    assert compare_mm(["86011", "86012"], ["86013", "86014"]) is False, "MM no overlap"
    assert compare_mm({"86011"}, "86011") is True, "MM exact match"
    assert (
        compare_mm("86011", ["86011", "86012"]) is True
    ), "MM exact match (single left)"
    assert (
        compare_mm(["86011", "86012"], "86012") is True
    ), "MM exact match (single right)"
    assert compare_mm([], ["86011"]) is False, "MM no match (empty left)"
    assert compare_mm(["86011"], None) is False, "MM no match (empty right)"
    assert (
        compare_mm("86011", range(86000, 87000)) is True
    ), "MM exact match (both single)"


def test_compare_codes_methods():
    """Test the main compare_codes function with different methods."""
    assert compare_codes("86011", ["86011"], method="OO") is True, "OO exact match"
    assert (
        compare_codes("86011", ["86011", "86012"], method="OO") is False
    ), "OO different codes"
    assert (
        compare_codes("86011", ["86011", "86012"], method="OM") is True
    ), "OM exact match in set right"
    assert (
        compare_codes(["86011", "86012"], "86011", method="MO") is True
    ), "MO exact match in set left"
    assert (
        compare_codes("86011", ["86011", "86012"], method="MO") is False
    ), "MO different codes"
    assert (
        compare_codes(["86011", "86012"], ["86012", "86013"], method="MM") is True
    ), "MM overlapping codes"
    with pytest.raises(ValueError):
        compare_codes("86011", "86011", method="INVALID")
