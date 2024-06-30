'''
This file is for doing some checks on model drift.

Author: Vitor Abdo
Date: Jun/2024
'''

# Import necessary packages
import boto3
import numpy as np


def get_column_values(table_name, column_name) -> list:
    '''
    Retrieve all values from a specified column in a DynamoDB table.

    This function scans the entire DynamoDB table and collects all values
    from a specified column, handling different DynamoDB data types.

    Args:
        table_name (str): The name of the DynamoDB table to scan.
        column_name (str): The name of the column to retrieve values from.

    Returns:
        list: A list of values from the specified column.
    '''
    # Initialize a DynamoDB resource
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)

    values = []

    # Perform initial scan of the table
    response = table.scan()
    items = response.get('Items', [])

    # Process the items retrieved from the initial scan
    for item in items:
        if column_name in item:
            value = item[column_name]
            if isinstance(value, dict):
                # Handle different DynamoDB data types stored as dictionaries
                if 'S' in value:
                    values.append(value['S'])  # String
                elif 'N' in value:
                    values.append(float(value['N']))  # Number
                elif 'B' in value:
                    values.append(value['B'])  # Binary
            else:
                # Directly add the value if it's not a dict
                values.append(value)

    # Continue scanning if there are more items to be retrieved
    while 'LastEvaluatedKey' in response:
        response = table.scan(
            ExclusiveStartKey=response['LastEvaluatedKey']
        )
        items = response.get('Items', [])

        # Process the items retrieved from subsequent scans
        for item in items:
            if column_name in item:
                value = item[column_name]
                if isinstance(value, dict):
                    # Handle different DynamoDB data types stored as
                    # dictionaries
                    if 'S' in value:
                        values.append(value['S'])  # String
                    elif 'N' in value:
                        values.append(float(value['N']))  # Number
                    elif 'B' in value:
                        values.append(value['B'])  # Binary
                else:
                    # Directly add the value if it's not a dict
                    values.append(value)

    return values


def raw_comparison_test(old_metric: float, new_metric: float) -> bool:
    '''
    Perform a raw comparison to detect model drift.

    This function checks if the new evaluation metric is worse than the historical metric.
    If the new metric is worse, it indicates model drift.

    Args:
        old_metric (float): The old minimum value recorded in the historical metrics.
        new_metric (float): The new value to compare.

    Returns:
        bool: True if model drift occurred (new metric is worse), False otherwise.
    '''
    return new_metric < old_metric


def parametric_significance_test(
        hist_metrics: list,
        new_metric: float) -> bool:
    '''
    Perform a parametric significance test to detect model drift.

    This function checks if the new evaluation metric is more than two standard deviations
    lower than the mean of all previous metrics. If the new metric is worse by this standard,
    it indicates model drift.

    Args:
        hist_metrics (list): A list of historical metric values recorded.
        new_metric (float): The new metric value to compare.

    Returns:
        bool: True if model drift occurred (new metric is significantly worse), False otherwise.
    '''
    mean_hist = np.mean(hist_metrics)
    std_hist = np.std(hist_metrics)
    return new_metric < mean_hist - (2 * std_hist)


def non_parametric_outlier_test(hist_metrics: list, new_metric: float) -> bool:
    '''
    Perform a non-parametric outlier test to detect model drift.

    This function uses the interquartile range (IQR) to determine if the new
    evaluation metric is an outlier. A metric is considered an extreme value if it is:

    1. More than 1.5 IQRs above the 75th percentile (a high outlier).
    2. More than 1.5 IQRs below the 25th percentile (a low outlier).

    If the new metric is a low outlier, it indicates model drift.

    Args:
        hist_metrics (list): A list of historical metric values.
        new_metric (float): The new metric value to compare.

    Returns:
        bool: True if model drift occurred (new metric is a low outlier), False otherwise.
    '''
    q75, q25 = np.quantile(hist_metrics, 0.75), np.quantile(hist_metrics, 0.25)
    iqr = q75 - q25
    return new_metric < q25 - (iqr * 1.5)


def final_model_drift_verify(hist_metrics: list, new_metric: float) -> bool:
    '''
    Verify model drift using three tests: raw comparison, parametric significance, and non-parametric outlier.

    This function checks for model drift by performing three different tests. If at least two of the tests indicate
    model drift, the function concludes that model drift has occurred.

    Args:
        hist_metrics (list): A list of historical metric values.
        new_metric (float): The new metric value to compare.

    Returns:
        bool: True if model drift occurred (at least two tests indicate drift), False otherwise.
    '''
    first_test = raw_comparison_test(min(hist_metrics), new_metric)
    second_test = parametric_significance_test(hist_metrics, new_metric)
    third_test = non_parametric_outlier_test(hist_metrics, new_metric)

    # Check if at least two tests indicate model drift
    if (first_test and second_test) or (
            first_test and third_test) or (second_test and third_test):
        return True
    else:
        return False
