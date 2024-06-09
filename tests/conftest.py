'''
This .py file is for creating the fixtures

Author: Vitor Abdo
Date: June/2024
'''

# import necessary packages
import pytest
import pandas as pd
import boto3
import logging
import os

# config
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')


@pytest.fixture(scope='session')
def data():
    '''Fixture to generate data to our tests'''

    # Create a session with AWS credentials
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name='us-east-1'
    )

    # Create a client instance for S3
    s3_client = session.client('s3')
    logging.info('S3 authentication was created successfully.')

    # Read the raw data from S3, selecting only desired columns
    obj = s3_client.get_object(Bucket='ct-fraud-data-bucket', Key='dataset.csv')
    df = pd.read_csv(obj['Body'])
    logging.info('Data from s3 folder was fetched successfully.')
    return df
