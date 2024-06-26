'''
This file is for testing the final model
with the "prod" tag in the test data

Author: Vitor Abdo
Date: May/2024
'''

# Import necessary packages
import logging
import pickle
import boto3
import json
import datetime
import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, brier_score_loss, confusion_matrix, roc_auc_score, RocCurveDisplay
from decimal import Decimal
os.makedirs('/tmp/matplotlib', exist_ok=True)
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib.pyplot as plt

# config
BUCKET_NAME_MODEL = os.environ['BUCKET_NAME_MODEL']
DYNAMO_TABLE_TRAIN_MODEL = os.environ['DYNAMO_TABLE_TRAIN_MODEL']
DYNAMO_TABLE_TEST_MODEL = os.environ['DYNAMO_TABLE_TEST_MODEL']

logging.basicConfig(
    level=logging.INFO,
    force=True,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')


def confusion_matrix_plot(y_true, y_pred, output_image_path) -> None:
    '''
    Generate and export a confusion matrix plot.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        output_image_path (str): The path to save the output image.
    '''
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    # Create a figure and axis
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Adjust the layout to prevent cutting off elements
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(output_image_path)
    plt.show()


def roc_auc_plot(model, X_test, y_test, output_image_path) -> None:
    '''
    Generate and export a ROC AUC curve plot.

    Args:
        model: The trained model.
        X_test (array-like or DataFrame): Test features.
        y_test (array-like): True labels for the test set.
        output_image_path (str): The path to save the output image.
    '''
    # Create a figure
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    # Plot the ROC AUC curve using the model and test data
    roc_disp = RocCurveDisplay.from_estimator(
        model, X_test, y_test, ax=ax, alpha=0.8)
    roc_disp.plot(ax=ax, alpha=0.8)

    # Adjust the layout to prevent cutting off elements
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(output_image_path)
    plt.show()


def evaluate_model(
        dataset: pd.DataFrame,
        test_size: float,
        label_column: str) -> None:
    '''
    Evaluate a machine learning model using a test dataset and log the results.

    This function loads the most recently registered model from an S3 bucket,
    evaluates its performance on a provided test set, generates evaluation
    metrics and plots, and logs the results to an AWS DynamoDB table.

    Args:
        dataset : pd.DataFrame
            The dataset containing features and the target variable.
        test_size : float
            The proportion of the dataset to include in the test split.
        label_column : str
            The name of the column in the dataset that contains the target variable.

    Returns:
        None
        This function does not return any value. It logs the results to DynamoDB
        and saves evaluation plots to S3.
    '''

    # Get the current date
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # create a client instance for S3
    s3_client = boto3.client('s3')
    logging.info('S3 authentication was created successfully.')

    # load the last trained model registered
    logging.info('Loading the last model registered...')
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(DYNAMO_TABLE_TRAIN_MODEL)
    answer = table.scan()

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

    final_model = s3_client.get_object(
        Bucket=BUCKET_NAME_MODEL,
        Key=f'pickles/extracted_at={current_date}/model_{tag_value}.pkl')
    final_model = pickle.loads(final_model['Body'].read())
    logging.info('Model loaded successfully.')

    # get the test set
    _, test_set = train_test_split(
        dataset, test_size=test_size, random_state=42)

    # read test dataset
    X_test = test_set.drop([label_column], axis=1)
    y_test = test_set[label_column]

    # making inference on test set
    final_predictions = final_model.predict(X_test)

    # scoring the results
    logging.info('Scoring in test set...')
    test_accuracy = balanced_accuracy_score(y_test, final_predictions)
    test_f1 = f1_score(y_test, final_predictions)
    test_brier = brier_score_loss(y_test, final_predictions)
    test_roc_auc = roc_auc_score(y_test, final_predictions)

    # displaying the confusion matrix
    output_confusion_matrix_image_path = f'confusion_matrix_{tag_value}.png'
    images_confusion_matrix_directory = f'images/extracted_at={current_date}/{output_confusion_matrix_image_path}'
    confusion_matrix_plot(
        y_test,
        final_predictions,
        '/tmp/' + output_confusion_matrix_image_path)
    s3_client.upload_file(
        '/tmp/' + output_confusion_matrix_image_path,
        BUCKET_NAME_MODEL,
        images_confusion_matrix_directory)
    logging.info('Confusion matrix image was inserted into bucket.')

    # displaying the roc auc curve
    output_rocauc_image_path = f'rocauc_{tag_value}.png'
    images_rocauc_directory = f'images/extracted_at={current_date}/{output_rocauc_image_path}'
    roc_auc_plot(
        final_model,
        X_test,
        y_test,
        '/tmp/' + output_rocauc_image_path)
    s3_client.upload_file(
        '/tmp/' + output_rocauc_image_path,
        BUCKET_NAME_MODEL,
        images_rocauc_directory)
    logging.info('Roc curve image was inserted into bucket.')

    # save the model register in dynamodb
    dynamodb = boto3.resource('dynamodb')
    dynamo_table = dynamodb.Table(DYNAMO_TABLE_TEST_MODEL)

    # insert the item with necessary fields to monitor model drift
    s3_url_confusion_matrix = f"https://{BUCKET_NAME_MODEL}.s3.amazonaws.com/images/extracted_at={current_date}/{output_confusion_matrix_image_path}"
    s3_url_rocauc = f"https://{BUCKET_NAME_MODEL}.s3.amazonaws.com/images/extracted_at={current_date}/{output_rocauc_image_path}"

    dynamo_table.put_item(Item={
        'id': int(pd.Timestamp.now().timestamp()),
        'publication_date': pd.Timestamp.now().isoformat(),
        'test_balanced_accuracy': Decimal(str(test_accuracy)),
        'test_f1': Decimal(str(test_f1)),
        'test_brier': Decimal(str(test_brier)),
        'test_roc_auc': Decimal(str(test_roc_auc)),
        'confusion_matrix_url': s3_url_confusion_matrix,
        'rocauc_url': s3_url_rocauc,
    })
    logging.info('The final test model was inserted into dynamo table.')
