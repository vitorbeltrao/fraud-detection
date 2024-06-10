'''
Script to manage the entire ML inference pipeline

Author: Vitor Abdo
Date: June/2024
'''

# import necessary packages
import json
import pickle
import os
import boto3
import logging
import datetime
import pandas as pd

# config
BUCKET_NAME_DATA = os.environ['BUCKET_NAME_DATA']
BUCKET_NAME_MODEL = os.environ['BUCKET_NAME_MODEL']
DYNAMO_TABLE_TRAIN_MODEL = os.environ['DYNAMO_TABLE_TRAIN_MODEL']

logging.basicConfig(
    level=logging.INFO,
    force=True,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')


def lambda_handler(event, context):
    '''
    AWS Lambda function handler for retrieving a machine learning model from an S3 bucket,
    making predictions with it using data from an HTTP request, and storing logs.
    '''
    # Get the current date
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # create a client instance for S3
    s3_client = boto3.client('s3')
    logging.info('S3 authentication was created successfully.')
    
    # Load the request body as JSON
    payload = json.loads(event['body'])
    logging.info('Payload loaded successfully: %s', payload)

    # load the last trained model registered
    logging.info('Loading the last model registered...')
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(DYNAMO_TABLE_TRAIN_MODEL)
    answer = table.scan()
    logging.info('DynamoDB table scan performed successfully.')

    if 'Items' in answer and len(answer['Items']) > 0:
        # order by id in a reverse order
        tag_value = sorted(
            answer['Items'],
            key=lambda x: x['id'],
            reverse=True)[0]['tag']
        logging.info(f'The last tag value: {tag_value}.')
    else:
        return {
            'statusCode': 404,
            'body': json.dumps('Model not found')
        }

    # Load the model from the registry bucket
    final_model = s3_client.get_object(
        Bucket=BUCKET_NAME_MODEL,
        Key=f'pickles/extracted_at={current_date}/model_{tag_value}.pkl')
    final_model = pickle.loads(final_model['Body'].read())
    logging.info('Model loaded successfully.')

    # Make predictions
    preds = final_model.predict(pd.DataFrame([payload]))[0]
    logging.info('Prediction made successfully: %s', preds)

    return {
        'statusCode': 200,
        'body': json.dumps(str(preds))
    }
