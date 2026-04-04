import boto3
from dagster import asset, AssetExecutionContext
import os

@asset
def upload_csv_to_minio(context: AssetExecutionContext):
    # Upload CSV files from local data/dataset to MinIO landing bucket using boto3.
    # MinIO configuration
    endpoint_url = os.getenv("AWS_S3_ENDPOINT")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_DEFAULT_REGION")
    bucket_name = "landing"

    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )

    # List of CSV files to upload
    csv_files = [
        "data/dataset/ratings.csv",
        "data/dataset/movies_metadata.csv",
        "data/dataset/links.csv"
    ]

    for csv_file in csv_files:
        if os.path.exists(csv_file):
            file_name = os.path.basename(csv_file)
            s3_client.upload_file(csv_file, bucket_name, file_name)
            context.log.info(f"Uploaded {csv_file} to s3://{bucket_name}/{file_name}")
        else:
            context.log.warning(f"File {csv_file} does not exist, skipping.")

    return "Upload completed"
