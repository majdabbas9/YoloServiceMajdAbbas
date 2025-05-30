import logging
import boto3
from botocore.exceptions import ClientError
import os
from datetime import datetime, timezone



def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name,bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True
def download_file(bucket_name, s3_key, object_name=None):
    # Download the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.download_file(bucket_name, s3_key, object_name)
    except ClientError as e:
        logging.error(e)
        return e
    return True
def delete_file(bucket_name, s3_key):
    # Delete the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
    except ClientError as e:
        logging.error(e)
        return False
    return True