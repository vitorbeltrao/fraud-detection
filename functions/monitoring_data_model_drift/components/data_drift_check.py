'''
This file is to verify and monitor the data drift,
at the end it generates a report on the data drift
and one on the stability of the data.

Author: Vitor Abdo
Date: Jun/2024
'''

# import necessary packages
import pandas as pd
import logging
import boto3
import datetime
import tempfile
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *
from evidently.test_suite import TestSuite
from evidently.tests import *


logging.basicConfig(
    level=logging.INFO,
    force=True,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')


def generate_evidently_reports(
        current_dataset: pd.DataFrame,
        reference_dataset: pd.DataFrame,
        selected_columns: list,
        bucket_name: str) -> None:
    '''
    Generate data drift and stability reports using Evidently for selected columns and upload to S3.

    Parameters:
        current_dataset (pd.DataFrame): The current dataset.
        reference_dataset (pd.DataFrame): The reference dataset.
        selected_columns (list): List of column names to include in the reports.
        bucket_name (str): The name of the S3 bucket where reports will be uploaded.

    Returns:
        None
    '''
    # Get the current date
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # Initialize the S3 client
    s3_client = boto3.client('s3')

    # Select specified columns from the datasets
    current_dataset_selected = current_dataset[selected_columns]
    reference_dataset_selected = reference_dataset[selected_columns]

    # Generate evidently data drift report
    drift_report = Report(metrics=[
        DataDriftPreset(),
    ])

    logging.info('Running data drift report...')
    drift_report.run(
        reference_data=reference_dataset_selected,
        current_data=current_dataset_selected)

    # Save to a temporary file and upload to S3
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        drift_report.save_html(tmp_file.name)
        s3_client.upload_file(
            tmp_file.name,
            bucket_name,
            f'data_drift_reports/data_drift_report_{current_date}.html')
    logging.info(
        'Data drift report uploaded to S3 at data_drift_reports/data_drift_report_%s.html',
        current_date)

    # Generate evidently data stability report
    stability_tests = TestSuite(tests=[
        TestNumberOfColumnsWithMissingValues(),
        TestNumberOfRowsWithMissingValues(),
        TestNumberOfConstantColumns(),
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns(),
        TestColumnsType(),
        TestNumberOfDriftedColumns(),
    ])

    logging.info('Running data stability report...')
    stability_tests.run(
        reference_data=reference_dataset_selected,
        current_data=current_dataset_selected)

    # Save to a temporary file and upload to S3
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        stability_tests.save_html(tmp_file.name)
        s3_client.upload_file(
            tmp_file.name,
            bucket_name,
            f'data_drift_reports/data_stability_report_{current_date}.html')
    logging.info(
        'Data stability report uploaded to S3 at data_drift_reports/data_stability_report_%s.html',
        current_date)
