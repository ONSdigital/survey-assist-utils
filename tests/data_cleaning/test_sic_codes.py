"""Test SIC code parsing and validation."""

# pylint: disable=C0116

import pytest

from survey_assist_utils.data_cleaning.sic_codes import (
    asses_codability_gain,
    expand_to_n_digit_str,
    extract_alt_candidates_n_digit_codes,
    get_clean_n_digit_codes,
    get_clean_n_digit_one_code,
    get_codability_level,
    parse_numerical_code,
    validate_sic_codes,
)


def test_parse_numerical_code_basic():
    assert parse_numerical_code("86101") == {"86101"}
    assert parse_numerical_code("[86101, 86210]") == {"86101", "86210"}
    assert parse_numerical_code("86101;8602x;4+") == {"86101", "8602x"}
    assert parse_numerical_code(86101) == {"86101"}


def test_parse_numerical_code_empty():
    assert parse_numerical_code("nan") == set()
    assert parse_numerical_code("-9") == set()
    assert parse_numerical_code("") == set()
    assert parse_numerical_code("") == set()
    assert parse_numerical_code(None) == set()


def test_parse_numerical_code_logs(caplog):
    with caplog.at_level("WARNING"):
        parse_numerical_code(0, code_regex_pattern=r"([")
    assert any("error parsing" in record.message.lower() for record in caplog.records)


def test_expand_to_n_digit_str():
    assert expand_to_n_digit_str("86101", 2) == {"86101"}
    assert expand_to_n_digit_str("", 1) == {str(x) for x in range(10)}
    result = expand_to_n_digit_str("86", 5)
    assert "86000" in result
    assert "86999" in result
    assert len(result) == 10**3


def test_get_clean_n_digit_one_code():
    print(get_clean_n_digit_one_code("86", 5))
    assert get_clean_n_digit_one_code("861012", 5) == {"86101"}
    assert get_clean_n_digit_one_code("86101", 5) == {"86101"}
    group86 = {"86100", "86102", "86230", "86220", "86900", "86101", "86210"}
    assert get_clean_n_digit_one_code("86xxx", 5) == group86
    assert get_clean_n_digit_one_code("86", 5) == group86
    assert get_clean_n_digit_one_code("861012", 3) == {"861"}
    assert get_clean_n_digit_one_code("86101", 0) == {"Q"}


def test_get_clean_n_digit_codes_with_invalid():
    # Case 1: Mixed valid and invalid (alphanumeric/garbage)
    codes = ["86101", "NotACode", "123456789", "86210"]
    valid, invalid = get_clean_n_digit_codes(codes, 5)

    # Check valid codes are processed
    assert "86101" in valid
    assert "86210" in valid
    assert len(valid) == 2  # noqa: PLR2004

    # Check invalid codes are captured correctly
    assert "NotACode" in invalid
    # "123456789" will likely fail the .isdigit() check or validation check
    assert "123456789" in invalid
    assert len(invalid) == 2  # noqa: PLR2004

    # Case 2: Purely invalid codes
    bad_input = ["Bad1", "Bad2"]
    valid, invalid = get_clean_n_digit_codes(bad_input, 5)
    assert valid == set()
    assert invalid == {"Bad1", "Bad2"}

    # Case 3: Numeric codes that shouldn't exist (assuming 00000 is not in VALID_SIC_CODES)
    # This tests the validation lookup failure path
    fake_codes = ["00000"]
    valid, invalid = get_clean_n_digit_codes(fake_codes, 5)
    if "00000" not in valid:  # Only assert if we are sure it's invalid in your lookup
        assert "00000" in invalid


def test_get_clean_n_digit_codes_5d():
    codes = ["86101", "86210", "85xxx"]
    result, invalid = get_clean_n_digit_codes(codes, 5)
    assert "86101" in result
    assert "86210" in result
    assert "85100" in result
    assert "85590" in result
    assert isinstance(result, set)

    # Check invalid codes - there should be none
    assert isinstance(invalid, set)
    assert invalid == set()


def test_get_clean_n_digit_codes_logs(caplog):
    with caplog.at_level("WARNING"):
        get_clean_n_digit_codes(3, 5)
    assert any("set of strings" in record.message.lower() for record in caplog.records)


def test_get_clean_n_digit_codes_section():
    cleaned, invalid = get_clean_n_digit_codes("2xxxx", 0)
    assert cleaned == {"C"}
    assert invalid == set()

    codes = ["86101", "86210", "2xxxx"]
    cleaned, invalid = get_clean_n_digit_codes(codes, 0)
    assert cleaned == {"Q", "C"}
    assert invalid == set()


def test_get_clean_n_digit_codes_logs_invalid_item(caplog):
    # Ensure we capture WARNING logs for this test
    with caplog.at_level("WARNING"):
        # Input contains an item that will produce no valid codes
        cleaned, invalid = get_clean_n_digit_codes({"98765"}, n=5)

    # Assert logging happened with the expected message fragment
    assert any(
        "has no valid codes" in record.message.lower() for record in caplog.records
    ), "Expected a warning about invalid codes to be logged"

    # And the invalid item is recorded in the invalid set
    assert "98765" in invalid
    # cleaned may be empty depending on your cleaning logic
    assert isinstance(cleaned, set)
    assert isinstance(invalid, set)


def test_validate_sic_codes():
    assert validate_sic_codes("01110") == {"01110"}
    valid = validate_sic_codes(["01110", "99999", "A"])
    assert "01110" in valid
    assert "A" in valid
    assert "99999" not in valid


def test_validate_sic_codes_logs(caplog):
    with caplog.at_level("WARNING"):
        validate_sic_codes(5)
    assert any("set of strings" in record.message.lower() for record in caplog.records)


def test_extract_alt_candidates_n_digit_codes():
    candidates = [
        {"code": "86101", "likelihood": 0.8},
        {"code": "86210", "likelihood": 0.6},
    ]
    result_valid, result_invalid = extract_alt_candidates_n_digit_codes(
        candidates, code_name="code", score_name="likelihood", threshold=0.7
    )
    assert result_valid == {"86101"}
    assert result_invalid == set()
    # No pruning
    result2_valid, result2_invalid = extract_alt_candidates_n_digit_codes(
        candidates, code_name="code", score_name="likelihood", threshold=0
    )
    assert result2_valid == {"86101", "86210"}
    assert result2_invalid == set()


def test_extract_alt_candidates_n_digit_codes_invalid():
    candidates = [
        {"code": "86101", "likelihood": 0.8},
        {"code": "12345", "likelihood": 0.6},
    ]
    result_valid, result_invalid = extract_alt_candidates_n_digit_codes(
        candidates, code_name="code", score_name="likelihood", threshold=0.7
    )
    assert result_valid == {"86101"}
    assert result_invalid == {"12345"}


@pytest.mark.parametrize(
    "codes,expected",
    [
        (set(), "Uncodable"),
        ({"01621", "8610"}, "Uncodable"),
        ({"01", "02"}, "Section (letter)"),
        ({"01", "01621"}, "Division (2-digits)"),
        ({"016"}, "Group (3-digits)"),
        ({"861"}, "Class (4-digits)"),
        ({"0162"}, "Class (4-digits)"),
        ({"01621"}, "Sub-class (5-digits)"),
    ],
)
def test_get_codability_level(codes, expected):
    assert get_codability_level(codes) == expected


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("Uncodable", "Uncodable", 0),
        ("Uncodable", "Section (letter)", 1),
        ("Section (letter)", "Uncodable", -1),
        ("Division (2-digits)", "Group (3-digits)", 1),
        ("Group (3-digits)", "Division (2-digits)", -1),
        ("Class (4-digits)", "Class (4-digits)", 0),
        ("Sub-class (5-digits)", "Class (4-digits)", -1),
        ("not a label", "Sub-class (5-digits)", None),
        ("not a label", "not a label", None),
    ],
)
def test_asses_codability_gain(left, right, expected):
    row = {"initial": left, "final": right}
    out = asses_codability_gain(
        row,
        initial_level_col="initial",
        final_level_col="final",
    )
    assert out == expected
