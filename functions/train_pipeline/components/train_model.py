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
import pandas as pd
import matplotlib.pyplot as plt
from decouple import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV, train_test_split


# config
qualitative_feat = ['score_1', 'entrega_doc_1']
entrega_doc_2_feat = ['entrega_doc_2']
entrega_doc_3_feat = ['entrega_doc_3']
pais_feat = ['pais']
quantitative_continue_feat = ['score_4', 'score_9', 'score_10', 'valor_compra']
BUCKET_NAME_MODEL = config('BUCKET_NAME_MODEL')
AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')
AWS_REGION = config('AWS_REGION')


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
    processed_features = qualitative_feat + entrega_doc_2_feat + entrega_doc_3_feat + pais_feat + quantitative_continue_feat

    # instantiate the final model
    final_model = Pipeline(
            steps=[('preprocessor', preprocessor),
                   ('under', RandomUnderSampler(random_state=42)),
                   ('ml_model', RandomForestClassifier(random_state=42))])
    
    return final_model, processed_features


def feature_importance_rf_plot(model, preprocessor, output_image_path):
    '''
    Generate and export a feature importance plot for the random forest model.

    Parameters:
    model (RandomForestClassifier): The trained RandomForest model.
    preprocessor (ColumnTransformer): The preprocessor used in the pipeline.
    output_image_path (str): The path to save the output image.
    '''
    # Importance based on each tree
    global_exp = pd.DataFrame()

    for i in range(model.n_estimators):
        global_exp[f'tree_{i+1}'] = model.estimators_[i].feature_importances_

    # Set the feature names as indices of the DataFrame
    global_exp.index = preprocessor.get_feature_names_out()

    # Calculate the mean importance of features across all trees
    global_exp['importance'] = global_exp.mean(axis=1)

    # Sort the DataFrame by the 'importance' column in descending order and plot it
    global_exp_sorted = global_exp.sort_values(by='importance', ascending=False)
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


def train_model(
        dataset: str,
        test_size: int,
        label_column: str,
        cv: int,
        scoring: list,
        refit: str,
        rf_config: dict) -> None:
    '''Function to train the model, tune the hyperparameters
    and save the best final model

    :param train_set: (str)
    Path to the wandb leading to the training dataset

    :param label_column: (str)
    Column name of the dataset to be trained that will be the label

    :param cv: (int)
    Determines the cross-validation split strategy

    :param scoring: (list)
    Strategy to evaluate the performance of the model of
    cross-validation in the validation set

    :param rf_config: (dict)
    Dict with the values of the hyperparameters of the adopted model
    '''
    # create a session with AWS credentials
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

    # create a client instance for S3
    s3_client = session.client('s3')
    logging.info('S3 authentication was created successfully.')

    # divide data into train and test to avoid data snooping bias
    train_set, _ = train_test_split(dataset, test_size=test_size, random_state=42)

    # get the final model
    model, _ = get_inference_pipeline()

    # select only the features that we are going to use
    X = train_set.drop([label_column], axis=1)
    y = train_set[label_column]

    # hyperparameter interval to be tested
    param_grid = {
        'ml_model__n_estimators': [150, 250, 350],
        'ml_model__n_estimators__max_depth': [8, 9, None],
        'under__sampling_strategy': ['auto', 0.5, 0.7, 0.8, 0.9, 1.0]
    }

    # training and apply grid search with cross-validation
    logging.info('Training...')
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
    logging.info('Scoring...')
    results = pd.DataFrame(grid_search.cv_results_)
    metrics = ['balanced_accuracy', 'f1', 'neg_brier_score']
    best_metrics = {}

    print("\nBest metrics for each calculated metric:\n")
    for metric in metrics:
        best_train_metric = results[f'mean_train_{metric}'][grid_search.best_index_]
        best_val_metric = results[f'mean_test_{metric}'][grid_search.best_index_]
        print(f"Best training {metric}: {best_train_metric:.4f}")
        print(f"Best vaidation {metric}: {best_val_metric:.4f}")

        best_metrics[f'train_{metric}'] = best_train_metric
        best_metrics[f'test_{metric}'] = best_val_metric

    # get the final model
    final_model = grid_search.best_estimator_

    # generate a random SHA for the model
    final_model_sha = os.urandom(16).hex()

    # displaying the model interpretation
    logging.info('Displaying model interpretation...')
    best_ml_model = grid_search.best_estimator_.named_steps['ml_model']
    preprocessor = grid_search.best_estimator_.named_steps['preprocessor']
    output_feature_importance_image_path = f'feature_importance_{final_model_sha}.png'

    images_directory = f'images/{output_feature_importance_image_path}'
    feature_importance_rf_plot(best_ml_model, preprocessor, output_feature_importance_image_path)
    s3_client.upload_file(output_feature_importance_image_path, BUCKET_NAME_MODEL, images_directory)
    logging.info('Model interpretation image was inserted into bucket.')

    # displaying the model calibration
    logging.info('Displaying model calibration...')


    logging.info('Model calibration image was inserted into bucket.')

    # save the model in s3 bucket
    logging.info('Putting the final model into dynamo table...')
    s3_client.put_object(
        Bucket=BUCKET_NAME_MODEL, Key=f'pickles/model_{final_model_sha}.pkl',
        Body=pickle.dumps(final_model), ContentType='application/octet-stream')
    logging.info('Model pickle file was inserted into bucket.')
    
    # save the model register in dynamodb
    dynamodb = boto3.resource('dynamodb')
    dynamo_table = dynamodb.Table('model-register')

    # insert the item with necessary fields to monitore model drift
    s3_url_feature_importance = f"https://{BUCKET_NAME_MODEL}.s3.amazonaws.com/images/{output_feature_importance_image_path}"
    s3_url_model_calibration = f"https://{BUCKET_NAME_MODEL}.s3.amazonaws.com/images/{output_image_path}"

    dynamo_table.put_item(Item={
        'id': int(pd.Timestamp.now().timestamp()),
        'publication_date': pd.Timestamp.now().isoformat(),
        'tag': final_model_sha,
        'train_time': timing,
        'metrics': json.dumps(best_metrics),
        'feature_importance_url': s3_url_feature_importance,
        'model_calibration_url': s3_url_model_calibration,
    })
    logging.info('The final model was inserted into dynamo table.')

    return {
        'statusCode': 200,
        'body': 'Model trained and inserted successful.'
    }
    