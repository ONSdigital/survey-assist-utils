#!/usr/bin/env python
"""Script that calculates accuracy metrics for survey assist evaluation model.

Takes evaluation_data file as positional arguments.

Optional arguments:
    -n <number_of_digits>, --number_of_digits <number_of_digits> Required number of digits
        Expected one of S / 0 / 1 / 2 / 3 / 4 / 5.
    -c <clerical_file>, --clerical_file <clerical_file> Path to the clerical codes file.
        If not provided, the main data file is expected to include clerical columns.
    -w, --write_output If set, writes the evaluation metrics to a JSON file.

Use:
    -h, --help to show help message.
"""

import json
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

    parser.add_argument(
        "-n",
        "--number_of_digits",
        type=str,
        default=None,
        help="Number of digits:  0 / 1 / 2 / 3 / 4 / 5 / S. Optional.",
    )

    parser.add_argument(
        "-c",
        "--clerical_file",
        type=str,
        default=None,
        help="Path to the clerical codes file. Optional.",
    )

    parser.add_argument(
        "-w",
        "--write_output",
        action="store_true",
        default=False,
        help="If set, writes the evaluation metrics to a JSON file.",
    )

    args = parser.parse_args()

    if args.number_of_digits is not None and args.number_of_digits not in (
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "S",
    ):
        raise ValueError("illegal value passed for number_of_digits")

    DIGITS = (
        (0 if args.number_of_digits == "S" else int(args.number_of_digits))
        if args.number_of_digits
        else 5
    )

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

    model_codes = prep_model_codes(
        my_dataframe,
        codes_col="initial_code",
        alt_codes_col="alt_sic_candidates",
        out_col="sa_initial_codes",
        alt_codes_name="code",
        threshold=0,
        digits=DIGITS,
    )

    combined_dataframe = model_codes.merge(clerical_codes, on="unique_id", how="inner")

    try:
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
    except KeyError:
        logger.warning(
            "Final codes not found in the data. Skipping final code metrics."
        )

    evaluation_metrics = calc_simple_metrics(combined_dataframe)

    print(evaluation_metrics.report_metrics())

    if args.write_output:
        out_file = args.evaluation_data.replace(".parquet", f"_metrics_{DIGITS}d.json")
        if out_file.startswith("gs://"):
            # optional dependency on gsfs
            from gcsfs import GCSFileSystem

            fs = GCSFileSystem()
            with fs.open(out_file, "w") as f:
                json.dump(evaluation_metrics.as_dict(), f, indent=2)
        else:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(evaluation_metrics.as_dict(), f, indent=2)
        logger.info("Wrote evaluation metrics to %s", out_file)
