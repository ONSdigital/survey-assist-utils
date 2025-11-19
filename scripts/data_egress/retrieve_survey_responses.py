#!/usr/bin/env python
"""This script retrieves survey responses from a Google Firestore database,
processes the data by flattening nested structures, and saves the results
into a CSV file.

It processes the data in chunks to handle large datasets efficiently.
"""
import json
import os
from argparse import ArgumentParser as AP
from collections.abc import Generator, MutableMapping
from datetime import datetime

import pandas as pd
from firebase_admin import firestore, initialize_app
from google.cloud import storage

from survey_assist_utils.logging import (
    get_logger,
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")


def setup_logger():
    """Set up the logger."""
    logger_tool = get_logger("data_egress", level=LOG_LEVEL.upper())
    return logger_tool


def setup_parser() -> AP:
    """Sets up a CLI parser."""
    parser = AP(
        "Utility to retrieve survey responses from a Firestore database.\n"
        "python retrieve_survey_responses.py"
    )
    parser.add_argument("project_id", type=str, help="The Google Cloud project ID.")
    parser.add_argument("database_id", type=str, help="The Firestore database ID.")
    parser.add_argument("collection_name", type=str, help="The collection_name.")
    parser.add_argument(
        "output_name", type=str, help="The base of the name of the output folder."
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=5,
        help="The connection timeout in seconds.",
    )
    parser.add_argument(
        "--chunk_size",
        "-c",
        type=int,
        default=500,
        help="The number of documents to process in each chunk.",
    )
    return parser


def apply_custom_adjustments_results(
    flattened_dict: dict | MutableMapping,
):
    """Reformats flattened dictionaries based on patterns, targeted towards
    the survey_response collection.
    """
    # Handle removing 'responses_0' prefix as there are never multiple responses.
    for key in list(flattened_dict.keys()):
        if key.startswith("responses_0_"):
            flattened_dict[f"{key.removeprefix('responses_0_')}"] = flattened_dict[key]
            del flattened_dict[key]

    for key in list(flattened_dict.keys()):
        # Handle deletion of dict items as we go
        if key not in flattened_dict:
            continue
        # Restructure 'field: <name>, value: <val>' patterns to 'fieldname: value' pattern.
        if key.endswith("field"):
            attribute_base = key.removesuffix("field")
            attribute_name = flattened_dict[key]
            attribute_value = f"{attribute_base}value"
            flattened_dict[f"{attribute_base}{attribute_name}"] = flattened_dict[
                attribute_value
            ]
            del flattened_dict[key]
            del flattened_dict[attribute_value]
    # Add flag to be False when we detect an issue with the response.
    # E.g. those introduced via page refreshes.
    return flattened_dict


def flatten_dict(
    d: dict | MutableMapping, parent_key: str = "", sep: str = "_"
) -> dict:
    """Flattens a nested dictionary (or dict-like object).

    Nested dictionary keys are combined with a separator. Lists are expanded
    such that each element gets its own key with an index.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key for constructing new keys. Defaults to ''.
        sep (str): The separator to use between nested keys. Defaults to '_'.

    Returns:
        dict: The flattened dictionary.
    """
    items: list[tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        # Handle item being a dictionary:
        if isinstance(v, (MutableMapping, dict)):
            items.extend(flatten_dict(v, parent_key=new_key, sep=sep).items())
        # Handle item being a list:
        elif isinstance(v, (list, tuple)):
            if (
                not v
            ):  # Handle empty lists by acting as if there's a single (empty) element:
                items.append((f"{new_key}{sep}0", None))
            else:
                for i, item in enumerate(v):
                    # Create a key for each list item - e.g. 'responses_0'
                    list_key = f"{new_key}{sep}{i}"
                    if isinstance(item, (MutableMapping, dict)):
                        items.extend(flatten_dict(item, list_key, sep=sep).items())
                    else:
                        # If the item in the list is not a dictionary, just save it with its index.
                        items.append((list_key, item))
        # No need to handle as a special case if item is already a single value:
        else:
            items.append((new_key, v))
    return dict(items)


def connect_to_firestore(
    project_id: str,
    database_id: str,
    logger_tool,
    collection_name: str = "survey_results",
    timeout: int | float = 5,
):
    """Initializes a connection to a Firestore database and returns a collection.

    Args:
        project_id (str): The Google Cloud project ID.
        database_id (str): The Firestore database ID.
        logger_tool: The logger tool.
        collection_name (str): The name of the Firestore collection.
            Defaults to 'survey_results'.
        timeout (int | float): The HTTP timeout in seconds for the connection.
            Defaults to 5.

    Returns:
        google.cloud.firestore_v1.collection.CollectionReference: A reference to
            the specified collection.

    Raises:
        ValueError: If `timeout` is not a positive number.
        ConnectionError: If the connection to Firestore fails or the
            specified collection is not found.
    """
    if timeout <= 0:
        logger_tool.error("`timeout` must be positive and non-zero.")
        raise ValueError("`timeout` must be positive and non-zero.")

    # Initialize Firestore connection
    app_options = {
        "projectId": project_id,
        "httpTimeout": timeout,
    }
    logger_tool.debug("Connecting to Firestore...")
    app = initialize_app(options=app_options)
    db = firestore.client(app=app, database_id=database_id)
    logger_tool.debug("Firestore client link established.")

    # non-intrusive test to verify connection
    logger_tool.debug("Testing connection to Firestore...")
    try:
        if collection_name not in [collection.id for collection in db.collections()]:
            logger_tool.error(f"'{collection_name}' collection not found.")
            raise ValueError(
                f"'{collection_name}' collection not found, or not accessible."
            )
    except Exception as e:
        logger_tool.error(f"Error when connecting to Firestore: {e}")
        raise ConnectionError(f"Error when connecting to Firestore: {e}") from e
    logger_tool.debug("Connection to Firestore successful.")
    return db.collection(collection_name)


def chunker(
    db_collection, chunk_size: int = 1000, collection_name: str = "survey_results"
) -> Generator[list[dict], None, None]:
    """Yields batches of documents from a Firestore collection.

    This generator function fetches documents from a Firestore collection in batches,
    ordered by 'time_end' to process large collections without loading everything into
    memory.

    Args:
        db_collection: The Firestore collection reference to query.
        chunk_size (int): The number of documents to retrieve in each chunk.
            Defaults to 1000.
        collection_name (str): the name of the collection being processed.
            Used to determine which adjustments / reformatting heuristics to apply.
            Defaults to 'survey_results'.

    Yields:
        list[dict]: A list of flattened dictionaries, where each dictionary
            represents a survey response document.
    """
    if chunk_size <= 0:
        raise ValueError("`chunk_size` must be positive and non-zero.")
    query = db_collection
    last_doc = None
    while True:
        try:
            current_query = (
                query.order_by("time_end").order_by("__name__").limit(chunk_size)
            )
            if current_query.count().get()[0][0].value == 0:
                raise AttributeError(
                    "'time_end' is not an attribute of the query, ordering by ID instead."
                )
        except AttributeError:
            current_query = query.order_by("__name__").limit(chunk_size)
        if last_doc:
            current_query = current_query.start_after(last_doc)

        docs = list(current_query.stream())
        if len(docs) == 0:
            break

        last_doc = docs[-1]
        flattened_dicts = [flatten_dict({"id": doc.id} | doc.to_dict()) for doc in docs]
        if collection_name == "survey_results":
            yield [
                apply_custom_adjustments_results(flattened_dict)
                for flattened_dict in flattened_dicts
            ]
        else:
            yield flattened_dicts


def prepare_output_directory(output_base: str, gcp: bool = False) -> str:
    """Creates the (local) output directory if it doesn't exist.

    Args:
        output_base (str): The base name of the output folder.
        gcp (bool): Whether a GCP Bucket is used for output storage.
            Defaults to False.

    Returns:
        str: The path to the output directory.
    """
    # If we're using local storage, we need to ensure directories exist before we write files within
    # them.
    # If we're using a GCP Bucket, the directories will be created automatically if needed.
    output_directory = f"{output_base}_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}"
    if not gcp:
        os.makedirs(os.path.dirname(f"{output_directory}/"), exist_ok=True)
    return output_directory


def process_and_save_survey_results(  # noqa: PLR0913 # pylint: disable=R0913,R0917,R0914
    project_id: str,
    database_id: str,
    logger_tool,
    output_directory: str,
    collection_name: str = "survey_results",
    timeout: int | float = 30,
    chunk_size: int = 1000,
) -> dict:
    """Connects to Firestore, processes survey results, and saves them to parquet files.

    This function orchestrates the full process of fetching survey data in
    chunks, flattening each document, and saving them to parquet files in batches.

    Args:
        project_id (str): The Google Cloud project ID.
        database_id (str): The Firestore database ID.
        logger_tool: The logger tool.
        collection_name (str): The name of the Firestore collection to process.
        output_directory (str): The name of the folder to store the output files.
        chunk_size (int): The number of documents to process in each chunk.
        timeout (int): The connection timeout in seconds.
    """
    logger_tool.debug("Attempting to retrieve collection information from Firestore...")
    survey_results_collection = connect_to_firestore(
        project_id, database_id, logger_tool, collection_name, timeout
    )
    logger_tool.debug("Collection information retrieved from Firestore successfully.")
    logger_tool.debug("Beggining to process survey outputs in batches...")
    total_chunks = 0
    for chunk_id, results_chunk in enumerate(
        chunker(survey_results_collection, chunk_size, collection_name)
    ):
        logger_tool.debug(f"Processing batch {chunk_id}...")
        total_chunks += 1
        df = pd.DataFrame(results_chunk)
        logger_tool.debug(f"Batch {chunk_id} loaded into DataFrame.")
        logger_tool.debug(f"Saving processed batch {chunk_id} to intermediate file...")
        df.to_parquet(
            f"{output_directory}/chunk_{chunk_id}.parquet",
            index=False,
        )
        logger_tool.debug(f"Saved processed batch {chunk_id} to intermediate file.")
    survey_collection_metadata = {
        "number_of_chunks": total_chunks,
        "chunk_size": chunk_size,
        "final_row_id": df.iloc[-1]["id"],
    }
    if collection_name == "survey_results":
        survey_collection_metadata["final_row_timestamp"] = df.iloc[-1][
            "time_start"
        ].strftime("%Y_%m_%d__%H_%M_%S_%f")
    return survey_collection_metadata


if __name__ == "__main__":
    cli_parser = setup_parser()
    args = cli_parser.parse_args()
    logger = setup_logger()
    use_gcp_storage = args.output_name.startswith("gs://")
    output_name_base = args.output_name.removesuffix(".csv")
    OUTPUT_DIRECTORY_NAME = prepare_output_directory(output_name_base, use_gcp_storage)
    metadata = process_and_save_survey_results(
        args.project_id,
        args.database_id,
        logger,
        OUTPUT_DIRECTORY_NAME,
        args.collection_name,
        args.timeout,
        args.chunk_size,
    )
    if use_gcp_storage:
        logger.debug("Initialising connection to GCP Bucket...")
        client = storage.Client()
        bucket = client.bucket(
            OUTPUT_DIRECTORY_NAME.removeprefix("gs://").split("/")[0]
        )
        blob = bucket.blob(
            f"{'/'.join(OUTPUT_DIRECTORY_NAME.removeprefix('gs://').split('/')[1:])}/metadata.json"
        )
        logger.debug("Writing the metadata file to GCP Bucket...")
        blob.upload_from_string(json.dumps(metadata), content_type="application/json")
        logger.debug("Metadata uploaded successfully")

    else:
        logger.debug("Writing the metadata file to local storage...")
        with open(f"{OUTPUT_DIRECTORY_NAME}/metadata.json", "w", encoding="utf8") as f:
            json.dump(metadata, f)
        logger.debug("Metadata written successfully")
