"""Tests for code comparison functions."""

import pytest

from survey_assist_utils.evaluation.code_comparison import (
    INVALID_VALUES,
    cast_code_to_set,
    compare_codes,
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


def test_compare_oo_exact_match():
    """Test the OO comparison method for exact matches."""
    assert compare_codes("86011", "86011", method="OO") is True, "OO exact match"
    assert compare_codes(86011, ["86011"], method="OO") is True, "OO exact match (int)"
    assert (
        compare_codes(["86011"], ["86011"], method="OO") is True
    ), "OO exact match (list)"
    assert (
        compare_codes({"86011"}, {"86011"}, method="OO") is True
    ), "OO exact match (set)"
    assert compare_codes("86011", "86012", method="OO") is False, "OO different codes"
    assert (
        compare_codes(["86011"], ["86012"], method="OO") is False
    ), "OO different codes (list)"
    assert (
        compare_codes(["86011", "86012"], ["86011"], method="OO") is False
    ), "OO different codes (list)"
    assert compare_codes("", "86011", method="OO") is False


def test_compare_om_in_shortlist():
    """Test the OM comparison method for matches in shortlist."""
    assert (
        compare_codes("86011", ["86011", "86012"], method="OM") is True
    ), "OM exact match in list"
    assert (
        compare_codes("86012", ["86012"], method="OM") is True
    ), "OM exact match in list (single)"
    assert (
        compare_codes("86013", ["86011", "86012"], method="OM") is False
    ), "OM no match in list"
    assert (
        compare_codes(["86011"], ["86011", "86012"], method="OM") is True
    ), "OM exact match in list (single)"
    assert (
        compare_codes(["86013"], ["86011", "86012"], method="OM") is False
    ), "OM no match in list (single)"
    assert (
        compare_codes(["86011", "86012"], ["86011", "86012"], method="OM") is False
    ), "OM exact match in list (multiple)"
    assert (
        compare_codes([], ["86011"], method="OM") is False
    ), "OM no match in list (empty)"


def test_compare_mo_in_shortlist():
    """Test the MO comparison method for matches in shortlist."""
    assert (
        compare_codes(["86011", "86012"], "86011", method="MO") is True
    ), "MO exact match in list"
    assert (
        compare_codes(["86012"], "86012", method="MO") is True
    ), "MO exact match in list (single)"
    assert (
        compare_codes(["86013"], "86011", method="MO") is False
    ), "MO no match in list"
    assert (
        compare_codes(["86011", "86012"], ["86011"], method="MO") is True
    ), "MO exact match in list (single)"
    assert (
        compare_codes(["86011", "86012"], ["86013"], method="MO") is False
    ), "MO no match in list (single)"
    assert (
        compare_codes(["86011", "86012"], ["86011", "86012"], method="MO") is False
    ), "MO exact match in list (multiple)"
    assert (
        compare_codes(["86011"], [], method="MO") is False
    ), "MO no match in list (empty)"


def test_compare_mm_any_in_both():
    """Test the MM comparison method for any matches in both sets."""
    assert (
        compare_codes(["86011", "86012"], ["86012", "86013"], method="MM") is True
    ), "MM overlapping lists"
    assert (
        compare_codes(["86011", "86012"], ["86013", "86014"], method="MM") is False
    ), "MM no overlap"
    assert compare_codes({"86011"}, "86011", method="MM") is True, "MM exact match"
    assert (
        compare_codes("86011", ["86011", "86012"], method="MM") is True
    ), "MM exact match (single left)"
    assert (
        compare_codes(["86011", "86012"], "86012", method="MM") is True
    ), "MM exact match (single right)"
    assert (
        compare_codes([], ["86011"], method="MM") is False
    ), "MM no match (empty left)"
    assert (
        compare_codes(["86011"], None, method="MM") is False
    ), "MM no match (empty right)"
    assert (
        compare_codes("86011", range(86000, 87000), method="MM") is True
    ), "MM exact match (both single)"


def test_compare_codes_invalid_method():
    """Test the main compare_codes function with invalid methods."""
    with pytest.raises(ValueError):
        compare_codes("86011", "86011", method="INVALID")
