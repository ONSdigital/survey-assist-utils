"""Test lookup sic_code_section_list and associated helper functions."""

# pylint: disable=C0116

from survey_assist_utils.data_cleaning.sic_code_section_list import (
    SECTION_LOOKUP,
    VALID_SIC_CODES,
    generate_section_lookup,
    generate_valid_sic_codes,
)


def test_valid_sic_codes_notnone():
    assert isinstance(VALID_SIC_CODES, set), "VALID_SIC_CODES should be a set"
    assert len(VALID_SIC_CODES) > 0, "VALID_SIC_CODES should not be empty"


def test_section_lookup_notnone():
    assert isinstance(SECTION_LOOKUP, dict), "SECTION_LOOKUP should be a dictionary"
    assert len(SECTION_LOOKUP) > 0, "SECTION_LOOKUP should not be empty"


def test_generate_valid_sic_codes_basic():
    section_list = (
        ("A", "01110"),
        ("B", "05100"),
    )
    valid_codes = generate_valid_sic_codes(section_list)
    assert isinstance(valid_codes, set), "Valid codes is not a set"
    # Should include section letters
    assert "A" in valid_codes, "Missing section A"
    assert "B" in valid_codes, "Missing section B"
    # Should include all prefixes of codes
    assert "0" in valid_codes, "Missing code 0"
    assert "01" in valid_codes, "Missing code 01"
    assert "011" in valid_codes, "Missing code 011"
    assert "0111" in valid_codes, "Missing code 0111"
    assert "01110" in valid_codes, "Missing code 01110"
    assert "05" in valid_codes, "Missing code 05"
    assert "051" in valid_codes, "Missing code 051"
    assert "0510" in valid_codes, "Missing code 0510"
    assert "05100" in valid_codes, "Missing code 05100"


def test_generate_section_lookup_basic():
    section_list = (
        ("A", "01110"),
        ("A", "01120"),
        ("B", "05100"),
        ("C", "10110"),
    )
    lookup = generate_section_lookup(section_list)
    # Should map first two digits to section
    assert isinstance(lookup, dict), "Lookup is not a dictionary"
    assert len(lookup) == 3, "Lookup should have 3 entries"  # noqa: PLR2004
    assert lookup["01"] == "A"
    assert lookup["05"] == "B"
    assert lookup["10"] == "C"


def test_generate_valid_sic_codes_empty():
    assert not generate_valid_sic_codes(())


def test_generate_section_lookup_empty():
    assert not generate_section_lookup(())
