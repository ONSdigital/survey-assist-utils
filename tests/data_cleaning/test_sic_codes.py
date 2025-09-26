"""Test SIC code parsing and validation."""

# pylint: disable=C0116

from survey_assist_utils.data_cleaning.sic_codes import (
    expand_to_n_digit_str,
    extract_alt_candidates_n_digit_codes,
    get_clean_n_digit_codes,
    get_clean_n_digit_one_code,
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


def test_get_clean_n_digit_codes_5d():
    codes = ["86101", "86210", "85xxx"]
    result = get_clean_n_digit_codes(codes, 5)
    assert "86101" in result
    assert "86210" in result
    assert "85100" in result
    assert "85590" in result
    assert isinstance(result, set)


def test_get_clean_n_digit_codes_logs(caplog):
    with caplog.at_level("WARNING"):
        get_clean_n_digit_codes(3, 5)
    assert any("set of strings" in record.message.lower() for record in caplog.records)


def test_get_clean_n_digit_codes_section():
    assert get_clean_n_digit_codes("2xxxx", 0) == {"C"}
    codes = ["86101", "86210", "2xxxx"]
    assert get_clean_n_digit_codes(codes, 0) == {"Q", "C"}


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
    result = extract_alt_candidates_n_digit_codes(
        candidates, code_name="code", score_name="likelihood", threshold=0.7
    )
    assert result == {"86101"}
    # No pruning
    result2 = extract_alt_candidates_n_digit_codes(
        candidates, code_name="code", score_name="likelihood", threshold=0
    )
    assert result2 == {"86101", "86210"}
