import boto3
import os
import zipfile


S3_BUCKET = "mlops-l9-bucket"
S3_KEY = "model.zip"
LOCAL_DIR = "../models"
ZIP_PATH = os.path.join(LOCAL_DIR, "model.zip")

def download_zip_from_s3(bucket, key, output_path):
    s3 = boto3.client("s3")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    s3.download_file(bucket, key, output_path)

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def cleanup(zip_path):
    if os.path.exists(zip_path):
        os.remove(zip_path)

if __name__ == "__main__":
    download_zip_from_s3(S3_BUCKET, S3_KEY, ZIP_PATH)
    extract_zip(ZIP_PATH, LOCAL_DIR)
    cleanup(ZIP_PATH)