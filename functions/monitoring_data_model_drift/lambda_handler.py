'''
Script to manage the entire ML model and data drift

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from components.train_model import train_model
from components.data_drift_check import generate_evidently_reports
from components.model_drift_check import get_column_values, final_model_drift_verify

# config
BUCKET_NAME_DATA = os.environ['BUCKET_NAME_DATA']
BUCKET_NAME_MODEL = os.environ['BUCKET_NAME_MODEL']
BUCKET_NAME_DATA_DRIFT = os.environ['BUCKET_NAME_DATA_DRIFT']
DYNAMO_TABLE_TRAIN_MODEL = os.environ['DYNAMO_TABLE_TRAIN_MODEL']
DYNAMO_TABLE_TEST_MODEL = os.environ['DYNAMO_TABLE_TEST_MODEL']
SNS_TOPIC_ARN = os.environ['SNS_TOPIC_ARN']

logging.basicConfig(
    level=logging.INFO,
    force=True,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')


def lambda_handler(event, context):
    '''
    AWS Lambda function handler for checking model and data drift
    '''
    # Get the current date
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # create client instances for S3 and SNS
    s3_client = boto3.client('s3')
    sns_client = boto3.client('sns')
    logging.info('S3 and SNS authentication were created successfully.\n')

    ########################### DATA DRIFT ###########################
    # concatenate all historical datasets
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME_DATA)
    dataframes = []
    for obj in response.get('Contents', []):
        key = obj['Key']
        if key.startswith('dataset') and key.endswith('.csv'):
            logging.info(f'Reading {key} from S3 bucket.')
            csv_obj = s3_client.get_object(Bucket=BUCKET_NAME_DATA, Key=key)
            current_df = pd.read_csv(csv_obj['Body'])
            dataframes.append(current_df)
            logging.info(f'Dataset {key} loaded successfully.')
    unified_dataframe = pd.concat(dataframes, ignore_index=True)
    logging.info('All datasets have been concatenated into a single dataframe.')

    # fetch reference dataset
    obj = s3_client.get_object(Bucket=BUCKET_NAME_DATA, Key='reference.csv')
    reference_df = pd.read_csv(obj['Body'])
    logging.info('Dataset loaded successfully.\n')

    # create data drift report
    selected_columns = [
        'score_1', 'entrega_doc_1', 'entrega_doc_2', 
        'entrega_doc_3', 'pais', 'score_4', 'score_9', 
        'score_10', 'valor_compra', 'fraude'] # columns that we gonna check data drift
    generate_evidently_reports(unified_dataframe, reference_df, selected_columns, BUCKET_NAME_DATA_DRIFT)
    logging.info('Finish model drift step successfully.\n')

    ########################### MODEL DRIFT ###########################
    # load the last trained model registered
    logging.info('Loading the last model registered...')
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(DYNAMO_TABLE_TRAIN_MODEL)
    answer = table.scan()

    if 'Items' in answer and len(answer['Items']) > 0:
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
    
    final_model = s3_client.get_object(
        Bucket=BUCKET_NAME_MODEL,
        Key=f'pickles/extracted_at={current_date}/model_{tag_value}.pkl')
    final_model = pickle.loads(final_model['Body'].read())
    logging.info('Model loaded successfully.')

    # get the test set
    _, test_set = train_test_split(
        unified_dataframe, test_size=0.15, random_state=42)

    # read test dataset
    X_test = test_set.drop(['fraude'], axis=1)
    y_test = test_set['fraude']

    # making inference on test set
    final_predictions = final_model.predict(X_test)

    # scoring the results
    logging.info('Scoring in test set...')
    last_f1_test_score = f1_score(y_test, final_predictions)

    # fetch historical test metrics from DynamoDB
    logging.info('Start getting the historical test metrics from DynamoDB...')
    hist_f1_test_scores = get_column_values(DYNAMO_TABLE_TEST_MODEL, 'test_f1')
    logging.info('Historical metrics collected successfully.')

    # testing model drift
    logging.info('Start testing model drift...')
    model_drift_result = final_model_drift_verify(hist_f1_test_scores, last_f1_test_score)
    logging.info('Model drift tested successfully.')

    # Check if model drift occurred
    if model_drift_result:
        logging.info('Model drift detected. Publishing to SNS topic.')
        message = {
            'default': 'Model drift detected. Please check and retrain the model.',
            'email': 'Model drift detected. The model needs to be retrained and redeployed.',
            'lambda': 'Model drift detected, triggering retraining and redeployment.'
        }
        sns_response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=json.dumps(message),
            Subject='Model Drift Alert',
            MessageStructure='json'
        )
        logging.info('SNS topic published successfully.')
        logging.info(f'SNS response: {sns_response}')

    return {
        'statusCode': 200,
        'body': json.dumps(model_drift_result)
    }
