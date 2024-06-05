'''
This file is for training, saving, tracking the best model and
get the feature importance for model

Author: Vitor Abdo
Date: May/2024
'''

# import necessary packages
import logging
import os
import pickle
import boto3
import json
import timeit
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.calibration import calibration_curve
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# config
BUCKET_NAME_MODEL = os.environ['BUCKET_NAME_MODEL']
DYNAMO_TABLE_TRAIN_MODEL = os.environ['DYNAMO_TABLE_TRAIN_MODEL']
qualitative_feat = ['score_1', 'entrega_doc_1']
entrega_doc_2_feat = ['entrega_doc_2']
entrega_doc_3_feat = ['entrega_doc_3']
pais_feat = ['pais']
quantitative_continue_feat = ['score_4', 'score_9', 'score_10', 'valor_compra']

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')


class CountryMapper(BaseEstimator, TransformerMixin):
    '''
    A custom transformer for mapping country codes to specified categories.

    This transformer maps the country codes 'BR', 'AR', 'UY', and 'US' to their respective
    values and maps all other country codes to 'outros'.
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def map_countries(country):
            if country == 'BR':
                return 'BR'
            elif country == 'AR':
                return 'AR'
            elif country == 'UY':
                return 'UY'
            elif country == 'US':
                return 'US'
            else:
                return 'outros'

        X['pais'] = X['pais'].apply(map_countries)
        return X

    def get_feature_names_out(self, X):
        return ['pais']


def get_inference_pipeline() -> Pipeline:
    '''
    Create and return a machine learning inference pipeline.

    This function constructs a pipeline for data preprocessing and model training.
    It preprocesses both qualitative and quantitative features, applies appropriate
    transformations, and sets up a RandomForest classifier with undersampling for handling
    class imbalance.

    Returns:
        tuple: A tuple containing the final pipeline and the list of processed feature names.
    '''

    # preprocessing pipeline for the features
    qualitative_preproc = make_pipeline(
        SimpleImputer(strategy='most_frequent'))

    entrega_doc_2_preproc = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='null'),
        OneHotEncoder(drop='first'))

    entrega_doc_3_preproc = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(drop='if_binary'))

    pais_preproc = make_pipeline(
        CountryMapper(),
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(drop='first'))

    quantitative_continue_preproc = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler())

    # apply the respective transformations with columntransformer method
    preprocessor = ColumnTransformer([
        ('entrega_doc_1_feat', qualitative_preproc, qualitative_feat),
        ('entrega_doc_2_feat', entrega_doc_2_preproc, entrega_doc_2_feat),
        ('entrega_doc_3_feat', entrega_doc_3_preproc, entrega_doc_3_feat),
        ('pais_feat', pais_preproc, pais_feat),
        ('quantitative_continue_feat', quantitative_continue_preproc, quantitative_continue_feat)],
        remainder='drop')

    # combine all the processed feature names into a single list
    processed_features = qualitative_feat + entrega_doc_2_feat + \
        entrega_doc_3_feat + pais_feat + quantitative_continue_feat

    # instantiate the final model
    final_model = Pipeline(
        steps=[('preprocessor', preprocessor),
               ('under', RandomUnderSampler(random_state=42)),
               ('ml_model', RandomForestClassifier(random_state=42))])

    return final_model, processed_features


def feature_importance_rf_plot(model, preprocessor, output_image_path) -> None:
    '''
    Generate and export a feature importance plot for the random forest model.

    Args:
        model (RandomForestClassifier): The trained RandomForest model.
        preprocessor (ColumnTransformer): The preprocessor used in the pipeline.
        output_image_path (str): The path to save the output image.

    Returns:
        None
    '''
    # Importance based on each tree
    global_exp = pd.DataFrame()

    for i in range(model.n_estimators):
        global_exp[f'tree_{i+1}'] = model.estimators_[i].feature_importances_

    # Set the feature names as indices of the DataFrame
    global_exp.index = preprocessor.get_feature_names_out()

    # Calculate the mean importance of features across all trees
    global_exp['importance'] = global_exp.mean(axis=1)

    # Sort the DataFrame by the 'importance' column in descending order and
    # plot it
    global_exp_sorted = global_exp.sort_values(
        by='importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(global_exp_sorted.index, global_exp_sorted['importance'])
    plt.ylabel("Feature importance")
    plt.title("Feature importance - Global explanations")
    plt.xticks(rotation=90)

    # Adjust the layout to prevent cutting off elements
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(output_image_path)
    plt.show()


def plot_calibration_curve(best_model, X, y, output_image_path, n_bins=10) -> None:
    '''
    Plot and save the calibration curve for the provided model.

    Args:
        best_model: The best trained model (e.g., the result of grid_search.best_estimator_).
        X: Feature matrix for making predictions.
        y: True labels.
        output_image_path: The path to save the output image.
        n_bins: Number of bins for the calibration curve (default: 10).

    Returns:
        None
    '''
    # Predict probabilities with the best model
    y_probs = best_model.predict_proba(X)[:, 1]

    # Calculate the calibration curve
    prob_true, prob_pred = calibration_curve(y, y_probs, n_bins=n_bins)

    # Plot the calibration curve
    plt.figure(figsize=(10, 7))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Frequency')
    plt.title('Calibration Curve')
    plt.legend()

    # Save the plot as an image file
    plt.savefig(output_image_path)
    plt.show()


def train_model(
        dataset: str,
        test_size: float,
        label_column: str,
        cv: int,
        scoring: list,
        refit: str,
        rf_config: dict) -> dict:
    '''
    Train a machine learning model using grid search and cross-validation,
    then save the model and related artifacts to AWS S3.

    Args:
        dataset (str): The dataset to be used for training.
        test_size (float): The proportion of the dataset to include in the test split.
        label_column (str): The name of the column to be used as the label.
        cv (int): The number of cross-validation folds.
        scoring (list): A list of scoring metrics to use.
        refit (str): The metric to refit the model on.
        rf_config (dict): Configuration for the Random Forest model, including the parameter grid.

    Returns:
        dict: A dictionary with the status code and message of the training result.
    '''
    # Get the current date
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # create a client instance for S3
    s3_client = boto3.client('s3')
    logging.info('S3 authentication was created successfully.')

    # divide data into train and test to avoid data snooping bias
    train_set, _ = train_test_split(
        dataset, test_size=test_size, random_state=42)

    # get the final model
    model, _ = get_inference_pipeline()

    # select only the features that we are going to use
    X = train_set.drop([label_column], axis=1)
    y = train_set[label_column]

    # extract param_grid from rf_config
    param_grid = rf_config['param_grid']

    # training and apply grid search with cross-validation
    logging.info('Training the model...')
    starttime = timeit.default_timer()
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        refit=refit,
        return_train_score=True)
    grid_search.fit(X, y)
    timing = timeit.default_timer() - starttime
    logging.info(f'The execution time of the model was:{timing}')

    # displaying the results of the grid search for all metrics
    logging.info('Scoring the training model...')
    results = pd.DataFrame(grid_search.cv_results_)
    metrics = ['balanced_accuracy', 'f1', 'neg_brier_score']
    best_metrics = {}

    print("\nBest metrics for each calculated metric:\n")
    for metric in metrics:
        best_train_metric = results[f'mean_train_{metric}'][grid_search.best_index_]
        best_val_metric = results[f'mean_test_{metric}'][grid_search.best_index_]
        print(f"Best training {metric}: {best_train_metric:.4f}")
        print(f"Best validation {metric}: {best_val_metric:.4f}")

        best_metrics[f'train_{metric}'] = best_train_metric
        best_metrics[f'validation_{metric}'] = best_val_metric

    # get the final model
    final_model = grid_search.best_estimator_

    # generate a random SHA for the model
    final_model_sha = os.urandom(16).hex()

    # displaying the model interpretation
    logging.info('Displaying model interpretation...')
    best_ml_model = grid_search.best_estimator_.named_steps['ml_model']
    preprocessor = grid_search.best_estimator_.named_steps['preprocessor']
    output_feature_importance_image_path = f'feature_importance_{final_model_sha}.png'

    images_feature_importance_directory = f'images/extracted_at={current_date}/{output_feature_importance_image_path}'
    feature_importance_rf_plot(
        best_ml_model,
        preprocessor,
        output_feature_importance_image_path)
    s3_client.upload_file(
        output_feature_importance_image_path,
        BUCKET_NAME_MODEL,
        images_feature_importance_directory)
    logging.info('Model interpretation image was inserted into bucket.')

    # displaying the model calibration
    logging.info('Displaying model calibration...')
    output_calibration_curve_image_path = f'calibration_curve_{final_model_sha}.png'
    images_calibration_curve_directory = f'images/extracted_at={current_date}/{output_calibration_curve_image_path}'

    plot_calibration_curve(
        final_model,
        X,
        y,
        output_calibration_curve_image_path)
    s3_client.upload_file(
        output_calibration_curve_image_path,
        BUCKET_NAME_MODEL,
        images_calibration_curve_directory)
    logging.info('Model calibration image was inserted into bucket.')

    # save the model in s3 bucket
    logging.info('Putting the final trained model into dynamo table...')
    s3_client.put_object(
        Bucket=BUCKET_NAME_MODEL, 
        Key=f'pickles/extracted_at={current_date}/model_{final_model_sha}.pkl',
        Body=pickle.dumps(final_model), 
        ContentType='application/octet-stream')
    logging.info('Model pickle file was inserted into bucket.')

    # save the model register in dynamodb
    dynamodb = boto3.resource('dynamodb')
    dynamo_table = dynamodb.Table(DYNAMO_TABLE_TRAIN_MODEL)

    # insert the item with necessary fields to monitor model drift
    s3_url_feature_importance = f"https://{BUCKET_NAME_MODEL}.s3.amazonaws.com/images/extracted_at={current_date}/{output_feature_importance_image_path}"
    s3_url_model_calibration = f"https://{BUCKET_NAME_MODEL}.s3.amazonaws.com/images/extracted_at={current_date}/{output_calibration_curve_image_path}"

    dynamo_table.put_item(Item={
        'id': int(pd.Timestamp.now().timestamp()),
        'publication_date': pd.Timestamp.now().isoformat(),
        'tag': final_model_sha,
        'train_time': timing,
        'metrics': json.dumps(best_metrics),
        'feature_importance_url': s3_url_feature_importance,
        'model_calibration_url': s3_url_model_calibration,
        'hyperparameters': json.dumps(param_grid)
    })
    logging.info('The final trained model was inserted into dynamo table.')

    return {
        'statusCode': 200,
        'body': 'Model trained and inserted successfully.'
    }
