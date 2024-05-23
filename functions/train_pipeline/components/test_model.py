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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from decouple import config
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, brier_score_loss, confusion_matrix, roc_auc_score, RocCurveDisplay

# config
BUCKET_NAME_MODEL = config('BUCKET_NAME_MODEL')
AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')
AWS_REGION = config('AWS_REGION')

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')


def confusion_matrix_plot(y_true, y_pred, output_image_path):
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
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # Adjust the layout to prevent cutting off elements
    plt.tight_layout()
    
    # Save the plot as an image file
    plt.savefig(output_image_path)
    plt.show()


def roc_auc_plot(model, X_test, y_test, output_image_path):
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
    roc_disp = RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, alpha=0.8)
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
    
    # create a session with AWS credentials
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

    # create a client instance for S3
    s3_client = session.client('s3')
    logging.info('S3 authentication was created successfully.')

    # load the last trained model registered
    logging.info('Loading the last model registered...')
    dynamo_table_name = 'model-register'
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(dynamo_table_name)
    answer = table.scan()

    if 'Items' in answer and len(answer['Items']) > 0:
        # order by id in a reverse order
        tag_value = sorted(answer['Items'], key=lambda x: x['id'], reverse=True)[0]['tag']
        logging.info(f'The last tag value: {tag_value}.')
    else:
        return {
            'statusCode': 404,
            'body': json.dumps('Model not found')
            }  
    
    final_model = s3_client.get_object(Bucket='bucket-registro-ct', Key=f'pickles/model_{tag_value}.pkl')
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
    images_confusion_matrix_directory = f'images/{output_confusion_matrix_image_path}'
    confusion_matrix_plot(
        y_test,
        final_predictions,
        output_confusion_matrix_image_path)
    s3_client.upload_file(
        output_confusion_matrix_image_path,
        BUCKET_NAME_MODEL,
        images_confusion_matrix_directory)
    logging.info('Confusion matrix image was inserted into bucket.')

    # displaying the roc auc curve
    output_rocauc_image_path = f'rocauc_{tag_value}.png'
    images_rocauc_directory = f'images/{output_rocauc_image_path}'
    roc_auc_plot(
        final_model,
        X_test,
        y_test,
        output_confusion_matrix_image_path)
    s3_client.upload_file(
        output_rocauc_image_path,
        BUCKET_NAME_MODEL,
        images_rocauc_directory)
    logging.info('Roc curve image was inserted into bucket.')