"""Script that calculates accuracy metrics for survey assist evaluation model.

Takes evaluation_data and match_digits as positional arguments.

Optional arguments:
    -c <clerical_file>, --clerical_file <clerical_file> Path to the clerical codes file.
        If not provided, the main data file is expected to include clerical columns.
    -o, --old_one_prompt to expect data format from one_prompt pipeline.

Use:
    -h, --help to show help message.
"""

import logging
from argparse import ArgumentParser as AP

import pandas as pd

from survey_assist_utils.data_cleaning.prep_data import (
    prep_clerical_codes,
    prep_model_codes,
)
from survey_assist_utils.evaluation.metrics import (
    calc_simple_metrics,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


if __name__ == "__main__":
    parser = AP()

    parser.add_argument(
        "evaluation_data", type=str, help="relative path to the parquet dataset"
    )

    parser.add_argument("match_digits", type=str, help="match type: full / n-digit")

    parser.add_argument(
        "-c",
        "--clerical_file",
        type=str,
        default=None,
        help="Path to the clerical codes file. Optional.",
    )

    parser.add_argument(
        "--old_one_prompt",
        "-o",
        action="store_true",
        default=False,
        help="expect data format from one_prompt pipeline (column names)",
    )

    args = parser.parse_args()

    if not args.match_digits.startswith(
        ("full", "0-digit", "1-digit", "2-digit", "3-digit", "4-digit", "5-digit")
    ):
        raise ValueError("illegal value passed for match_digits")
    DIGITS = 5 if args.match_digits == "full" else int(args.match_digits[0])

    # Load final-stage output DataFrame
    try:
        my_dataframe = pd.read_parquet(args.evaluation_data)
        logger.info(
            "Loaded %d rows from %s for evaluation.",
            len(my_dataframe),
            args.evaluation_data,
        )
    except FileNotFoundError as e:
        logger.error("Could not read file: %s", args.evaluation_data)
        raise e

    clerical_df = my_dataframe
    # pylint: disable=C0103
    clerical_4plu_df = None
    # load clerical codes
    if args.clerical_file:
        try:
            clerical_df = pd.read_csv(args.clerical_file)
        except FileNotFoundError as e:
            logger.error("Could not read file: %s", args.clerical_file)
            raise e

        try:
            # Split on last dash and insert substring '_4plus' before file extension
            base, filename = args.clerical_file.rsplit("/", 1)

            fourplusfilename = f"{base}/Codes_for_4_plus_{filename}"
            clerical_4plu_df = pd.read_csv(fourplusfilename)
        except FileNotFoundError:
            logger.warning(
                "Could not read file: %s. Clerical codes for 4+ candidate set can't be used.",
                fourplusfilename,
            )

    clerical_codes = prep_clerical_codes(clerical_df, clerical_4plu_df, digits=DIGITS)

    model_codes = (
        prep_model_codes(
            my_dataframe,
            codes_col="final_sic_code",
            alt_codes_col="sic_candidates",
            out_col="sa_initial_codes",
            alt_codes_name="sic_code",
            threshold=0.7,
            digits=DIGITS,
        )
        if args.old_one_prompt
        else prep_model_codes(
            my_dataframe,
            codes_col="initial_code",
            alt_codes_col="alt_sic_candidates",
            out_col="sa_initial_codes",
            alt_codes_name="code",
            threshold=0,
            digits=DIGITS,
        )
    )

    combined_dataframe = model_codes.merge(clerical_codes, on="unique_id", how="inner")

    if not args.old_one_prompt:
        final_codes = prep_model_codes(
            my_dataframe,
            codes_col="initial_code",
            alt_codes_col="final_sic",
            out_col="sa_final_codes",
            digits=DIGITS,
        )
        combined_dataframe = combined_dataframe.merge(
            final_codes, on="unique_id", how="left"
        )

    evaluation_metrics = calc_simple_metrics(combined_dataframe)

    print(evaluation_metrics.report_metrics())
