# pylint: disable=W0511
#!/usr/bin/env python
"""Script that calculates accuracy metrics for two prompts evaluation model.

Takes evaluation_data, test_type, and match_type as positional arguments.

Allows parsing --filter_unambiguous, --filter_ambiguous, and
--neglect_impossible as optional atguments.

Use:
    -h, --help to show help message.
"""
import re
from argparse import ArgumentParser as AP

import pandas as pd


def parse_clerical_code(candidates_str: str):
    """Converts the clerical coder responses from a
    stringified list to a proper list of strings.
    """
    if (
        pd.isna(candidates_str)
        or candidates_str == ""
        or str(candidates_str).lower() == "nan"
    ):
        return []

    try:
        # Extract all RagCandidate entries using regex
        pattern = r"([0-9]+x*X*)"
        matches = re.findall(pattern, str(candidates_str))  # pylint: disable=W0621

        return matches
    except Exception:  # pylint: disable=W0706 # TODO: introduce logging
        raise


def allocate_final_final_sic(row):
    """Handles the intermediate-result routing for what
    should be considered the true final sic code.
    """
    if row["unambiguously_codable"]:
        return row["initial_code"]
    if row["unambiguously_codable_final"]:
        return row["final_sic"]
    return row["higher_level_final_sic"]


def get_top_clerical_code(codes: list) -> str:
    """Extract the top clerical code from the parsed
    clerical codes.
    """
    if len(codes) == 0:
        return ""
    return codes[0]


def get_clean_5digit_str(input_str: str) -> str:
    """Converts a 5digit string to either a valid SIC code format
    or an empty string. E.g. '86011' -> '86011'; '86xxx' -> ''.
    """
    if len(input_str) != 5:  # noqa: PLR2004
        return ""

    pattern = r"^([0-9]{5})"
    matches = re.findall(pattern, input_str)  # pylint: disable=W0621
    if len(matches) == 1:
        return matches[0]
    return ""


def get_clean_5digit_list(input_list: list[str]) -> list[str]:
    """Converts a list of possible codes to a list containing only
    valid 5-digit SIC codes.
    E.g. ['86011', '86012', '85xxx'] -> ['86011', '86012'].
    """
    cleaned_list = [get_clean_5digit_str(i) for i in input_list]
    pruned_list = [i for i in cleaned_list if len(i) > 0]
    return pruned_list


def get_clean_2digit_str(input_str: str) -> str:
    """Converts a 5digit string to either a valid 2-digit SIC code
    format or an empty string.
    E.g. '86011' -> '86'; '86xxx' -> '86'; '-9' -> ''.
    """
    if len(input_str) != 5:  # noqa: PLR2004
        return ""

    pattern = r"^([0-9]{2})"
    matches = re.findall(pattern, input_str)  # pylint: disable=W0621
    if len(matches) == 1:
        return matches[0]
    return ""


def get_clean_2digit_list(input_list: list[str]) -> list[str]:
    """Converts a list of 2-digit possible codes to a list containing
    only unique, valid 2-digit SIC codes.
    E.g. ['86', '86', ''] -> ['86'].
    """
    cleaned_list = [get_clean_2digit_str(i) for i in input_list]
    pruned_list = [i for i in cleaned_list if len(i) > 0]
    unique_list = list(set(pruned_list))
    return unique_list


parser = AP()

parser.add_argument(
    "evaluation_data", type=str, help="relative path to the parquet dataset"
)

parser.add_argument(
    "test_type",
    type=str,
    help="test type: OO / MM / OM / MO (M=Many, O=One, format: CC-SA)",
)
parser.add_argument("match_type", type=str, help="match type: full / 2-digit")
parser.add_argument(
    "--filter_unambiguous",
    "-fua",
    action="store_true",
    default=False,
    help="add flag to only consider CC-reported unambiguously codable responses",
)
parser.add_argument(
    "--filter_ambiguous",
    "-fa",
    action="store_true",
    default=False,
    help="add flag to only consider CC-reported NOT unambiguously codable responses",
)
parser.add_argument(
    "--neglect_impossible",
    "-n",
    action="store_true",
    default=False,
    help="ignore rows where no n-digit clerical code is available when calculating accuracy",
)

args = parser.parse_args()

assert args.test_type in [  # noqa: S101
    "OO",
    "MM",
    "OM",
    "MO",
], "illegal value passed for test_type"
assert args.match_type in [  # noqa: S101
    "full",
    "2-digit",
], "illegal value passed for match_type"

# Load final-stage output DataFrame
try:
    my_dataframe = pd.read_parquet(args.evaluation_data)
except FileNotFoundError as e:
    print(f"no such file: {args.evaluation_data}")
    raise e

# Apply filtering (if specified)
if args.filter_unambiguous:
    my_dataframe = my_dataframe[my_dataframe["Unambiguous"]]
elif args.filter_ambiguous:
    my_dataframe = my_dataframe[~my_dataframe["Unambiguous"]]

# Parse clerical coder column to actual list of strings
my_dataframe["All_Clerical_codes_parsed"] = my_dataframe["All_Clerical_codes"].apply(
    parse_clerical_code
)
# Extract the top clerical code as new column, for ease of comparisons
my_dataframe["top_clerical_code"] = my_dataframe["All_Clerical_codes_parsed"].apply(
    get_top_clerical_code
)
# Extract the codes from the model's alt_candidates as a list in a new column
my_dataframe["alt_sic_candidate_parsed"] = my_dataframe["alt_sic_candidates"].apply(
    lambda x: [xi["code"] for xi in x] if len(x) > 0 else []
)


# Add on the initial sic code if unambiguously codable
def get_extended_alt_candidates(row: pd.Series) -> list[str]:
    """Extends list of alternative SIC code by the initial SIC code.

    Args:
        row (pd.Series): List of alternative SIC codes.

    Returns:
        list[str]: List of alternative SIC codes, including the initial code.
    """
    candidate_list = row["alt_sic_candidate_parsed"]
    if row["unambiguously_codable"]:
        candidate_list.append(row["initial_code"])
    return candidate_list


my_dataframe["alt_sic_candidate_parsed_extended"] = my_dataframe.apply(
    get_extended_alt_candidates, axis=1
)
# Allocate the 'actual' final model SIC code to be used in "-O" style comparisons
my_dataframe["final_final_sic"] = my_dataframe.apply(allocate_final_final_sic, axis=1)

# Create the *cleaned* data columns for easy comparisons (5d & 2d):
## clerical & model 'one' columns:
my_dataframe["final_final_sic_5d_clean"] = my_dataframe["final_final_sic"].apply(
    get_clean_5digit_str
)
my_dataframe["final_final_sic_2d_clean"] = my_dataframe["final_final_sic"].apply(
    get_clean_2digit_str
)
my_dataframe["top_clerical_code_5d_clean"] = my_dataframe["top_clerical_code"].apply(
    get_clean_5digit_str
)
my_dataframe["top_clerical_code_2d_clean"] = my_dataframe["top_clerical_code"].apply(
    get_clean_2digit_str
)
## clerical & model 'many' columns:
my_dataframe["alt_sic_candidates_5d_clean"] = my_dataframe[
    "alt_sic_candidate_parsed_extended"
].apply(get_clean_5digit_list)
my_dataframe["alt_sic_candidates_2d_clean"] = my_dataframe[
    "alt_sic_candidate_parsed_extended"
].apply(get_clean_2digit_list)
my_dataframe["All_clerical_5d_clean"] = my_dataframe["All_Clerical_codes_parsed"].apply(
    get_clean_5digit_list
)
my_dataframe["All_clerical_2d_clean"] = my_dataframe["All_Clerical_codes_parsed"].apply(
    get_clean_2digit_list
)


def compare_OO(clerical_col: str, model_col: str) -> bool:  # pylint: disable=C0103
    """Returns true where clerical coders and model agree exactly.
    Assumes cleaned input columns.
    Applicable to both 2-digit and 5-digit columns.
    If one is an empty string, returns False.
    """
    if clerical_col in ("-9", ""):
        return False
    if model_col in ("-9", ""):
        return False
    return clerical_col == model_col


def compare_OM(  # pylint: disable=C0103
    clerical_col: str, model_col: list[str]
) -> bool:
    """Returns true where clerical coder choice is in the model's shortlist.
    Assumes cleaned input columns.
    Applicable to both 2-digit and 5-digit columns.
    If clerical code is an empty string, returns False.
    If the model's shortlist is empty, returns False.
    """
    if clerical_col in ("-9", ""):
        return False
    if len(model_col) == 0:
        return False
    return clerical_col in model_col


def compare_MO(  # pylint: disable=C0103
    clerical_col: list[str], model_col: str
) -> bool:
    """Returns true where any clerical coder option matches model choice.
    Assumes cleaned input columns.
    Applicable to both 2-digit and 5-digit columns.
    If clerical code list is empty, returns False.
    If the model's top choice is empty string, returns False.
    """
    if model_col in ("-9", ""):
        return False
    if len(clerical_col) == 0:
        return False
    return model_col in clerical_col


def compare_MM(  # pylint: disable=C0103
    clerical_col: list[str], model_col: list[str]
) -> bool:
    """Returns true where any clerical coder choice is in the model's shortlist.
    Assumes cleaned input columns.
    Applicable to both 2-digit and 5-digit columns.
    If either list is empty, returns False.
    """
    if len(model_col) == 0:
        return False
    if len(clerical_col) == 0:
        return False
    return any(i in clerical_col for i in model_col)


# Define the configuration for the test
if args.test_type in ["OO", "MO"]:
    MODEL_LABEL_COL_PREFIX = "final_final_sic"

elif args.test_type in ["MM", "OM"]:
    MODEL_LABEL_COL_PREFIX = "alt_sic_candidates"

if args.test_type in ["OO", "OM"]:
    CLERICAL_LABEL_COL_PREFIX = "top_clerical_code"

elif args.test_type in ["MM", "MO"]:
    CLERICAL_LABEL_COL_PREFIX = "All_clerical"

SUFFIX = "_5d_clean" if args.match_type == "full" else "_2d_clean"
# if args.match_type == "full":
#     SUFFIX = "_5d_clean"
# else:
#     SUFFIX = "_2d_clean"

# Determine the columns which will be used for the assesment
MODEL_COL_NAME = MODEL_LABEL_COL_PREFIX + SUFFIX  # pylint: disable=E0606
CLERICAL_COL_NAME = CLERICAL_LABEL_COL_PREFIX + SUFFIX  # pylint: disable=E0606


# Define the row-wise applyable test function
def compare_row(row: pd.Series) -> bool:
    """Select the desired comparison method beteen codes selected by
    Clerical Coder with codes selected by LLM tool.

    Args:
        row (pd.Series): rows containing data form CC and LLM to compare.

    Raises:
        ValueError: Rises when incorrect comparison methos id selected.

    Returns:
        bool: True if comparison method finds the specified number of matches
            between CC and LLM results.
    """
    if args.test_type == "OO":
        return compare_OO(row[CLERICAL_COL_NAME], row[MODEL_COL_NAME])
    if args.test_type == "OM":
        return compare_OM(row[CLERICAL_COL_NAME], row[MODEL_COL_NAME])
    if args.test_type == "MO":
        return compare_MO(row[CLERICAL_COL_NAME], row[MODEL_COL_NAME])
    if args.test_type == "MM":
        return compare_MM(row[CLERICAL_COL_NAME], row[MODEL_COL_NAME])
    raise ValueError(
        f"Invalid input: '{args.test_type}'. Expected 'OO', 'OM', 'MO', or 'MM'."
    )


my_dataframe["results_column"] = my_dataframe.apply(compare_row, axis=1)
matches = my_dataframe["results_column"].sum()

# Handle CC recording '4+' etc. by neglecting impossible to match CC values in accuracy calc.
if args.neglect_impossible:
    if args.match_type == "2-digit":
        NEGLECT_COUNT = len(
            my_dataframe[my_dataframe["top_clerical_code_2d_clean"] == ""]
        )
    else:
        NEGLECT_COUNT = len(
            my_dataframe[my_dataframe["top_clerical_code_5d_clean"] == ""]
        )
else:
    NEGLECT_COUNT = 0

usable_count = len(my_dataframe) - NEGLECT_COUNT

if args.filter_unambiguous:
    print("\nOnly considering CC-recorded unambiguously codable records:")
elif args.filter_ambiguous:
    print("\nOnly considering CC-recorded NOT unambiguously codable records:")
else:
    print("\nConsidering ALL records")

if args.neglect_impossible:
    print(
        f"{NEGLECT_COUNT} records had no usable clerically coded answer, and are ignored in calculation"  # pylint: disable=C0301
    )

print(f"\ntest type: {args.test_type}")
print(f"accuracy {args.match_type}: {100*matches/usable_count:.4f}%")
print(f"matches {args.match_type}: {matches}")
print(f"non_matches {args.match_type}: {usable_count-matches}")
print(f"total considered: {usable_count}")
