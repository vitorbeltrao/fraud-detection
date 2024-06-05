'''
Script to manage the entire ML model pipeline

Author: Vitor Abdo
Date: May/2024
'''

# import necessary packages
import logging
import boto3
import yaml
import os
import pandas as pd
from components.train_model import train_model
from components.test_model import evaluate_model

# config
BUCKET_NAME_DATA = os.environ['BUCKET_NAME_DATA']

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')


def lambda_handler(event, context):
    '''
    AWS Lambda function handler for training and evaluating a
    machine learning model using data from an S3 bucket.
    '''

    # create a client instance for S3
    s3_client = boto3.client('s3')
    logging.info('S3 authentication was created successfully.')

    # get the dataset to feed the train and test function
    obj = s3_client.get_object(Bucket=BUCKET_NAME_DATA, Key='dataset.csv')
    dataset = pd.read_csv(obj['Body'])
    logging.info('Dataset loaded successfully.\n')

    # load yaml file configuration
    with open('rf_config.yaml', 'r') as file:
        rf_config = yaml.safe_load(file)
    logging.info('yaml file loaded successfully.\n')

    # call the train pipeline function
    logging.info('Starting the training pipeline...\n')
    train_model(
        dataset=dataset,
        test_size=0.15,
        label_column='fraude',
        cv=5,
        scoring=['balanced_accuracy', 'f1', 'neg_brier_score'],
        refit='f1',
        rf_config=rf_config
    )
    logging.info('The train pipeline finished successfully.\n')

    # call the test pipeline function
    logging.info('Starting the test pipeline...\n')
    evaluate_model(
        dataset,
        test_size=0.15,
        label_column='fraude'
    )
    logging.info('The test pipeline finished successfully.\n')
