'''
This .py file is for creating the fixtures

Author: Vitor Abdo
Date: March/2023
'''

# import necessary packages
import pytest
import pandas as pd
import boto3
import logging
from decouple import config

# config
BUCKET_NAME = config('BUCKET_NAME')
AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')
AWS_REGION = config('AWS_REGION')


@pytest.fixture(scope='session')
def data():
    '''Fixture to generate data to our tests'''

    # Create a session with AWS credentials
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

    # Create a client instance for S3
    s3_client = session.client('s3')
    logging.info('S3 authentication was created successfully.')

    # Define the paths for the raw and processed layers
    bucket_directory = f'dataset.csv'

    # Read the raw data from S3, selecting only desired columns
    obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=bucket_directory)
    df = pd.read_csv(obj['Body'])
    logging.info('Data from s3 folder was fetched successfully.')
    return df
