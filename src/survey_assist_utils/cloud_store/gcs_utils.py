"""Utilities for interacting with Google Cloud Storage (GCS).

This module provides helper functions to upload and download files
between the local filesystem and GCS buckets using the Google Cloud
Storage Python client library.

Functions:
    - upload_to_gcs(local_path, gcs_uri): Uploads a local file to a specified GCS URI.
    - download_from_gcs(gcs_uri, local_path): Downloads a file from a GCS URI to a local path.
"""

import logging
from urllib.parse import urlparse

from google.cloud import storage


def upload_to_gcs(local_path, gcs_uri):
    """Upload a local file to Google Cloud Storage (GCS)."""
    parsed = urlparse(gcs_uri)
    bucket_name = parsed.netloc
    blob_path = parsed.path.lstrip("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    blob.upload_from_filename(local_path)
    logging.info("Uploaded intermediate results to gs://%s/%s", bucket_name, blob_path)


def download_from_gcs(gcs_uri, local_path):
    """Download a file from GCS to a local path."""
    parsed = urlparse(gcs_uri)
    bucket_name = parsed.netloc
    blob_path = parsed.path.lstrip("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    logging.info(
        "Downloaded GCS file gs://%s/%s to %s", bucket_name, blob_path, local_path
    )
