from google.cloud import storage
import os
from datetime import datetime
import numpy as np
os.environ["GCLOUD_PROJECT"] = "noaa-passive-bioacoustic"


def get_total_size(bucket_name, source_folder):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_folder)
    total = sum(1 for _ in blobs)
    return total, blobs

def get_file_names(bucket_name, source_folder):
    """Downloads a folder from a Google Cloud Storage bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Convert iterator to a list to avoid exhaustion
    blobs = list(bucket.list_blobs(prefix=source_folder))
    file_names = [x.name for x in blobs if x.name.endswith('.flac') or x.name.endswith('.wav')]
    return file_names

def download_file(bucket_name, source_file, destination_file_name):
    """Downloads a folder from a Google Cloud Storage bucket."""

    # Initialize the Google Cloud Storage client
    storage_client = storage.Client()

    # Get the bucket object
    bucket = storage_client.get_bucket(bucket_name)

    # Get the blob (file object) from the bucket
    blob = bucket.blob(source_file)

    # Download the file to the local machine
    blob.download_to_filename(destination_file_name)


def download_folder(bucket_name, source_folder, destination_folder, start_index=0):
    """Downloads a folder from a Google Cloud Storage bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Convert iterator to a list to avoid exhaustion
    blobs = list(bucket.list_blobs(prefix=source_folder))
    total = len(blobs)

    counter = 0
    download_size = 0

    for blob in blobs:
        if counter < start_index:
            counter += 1  # Ensure counter updates when skipping
            continue
        print('File name: ', blob.name)
        start_time = np.datetime64(datetime.strptime(' '.join(blob.name.split('.')[0].split('_')[-2:]), "%Y%m%d %H%M%S"), 's')
        if start_time < np.datetime64('2019-01-01'):
            continue

        # Define the local path
        local_path = os.path.join(destination_folder, blob.name)
        print(blob.name, counter)

        # Create local directories if they don't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download the blob
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")
        download_size += os.path.getsize(local_path)

        counter += 1

        # Stop conditions
        if counter >= 1:
            break
        if download_size > 550_000_000_000:
            break
        if counter >= total:
            break

    return counter, total