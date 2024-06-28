# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The fraud detection system project is based on a machine learning model that predicts whether a bank transaction is a fraud or not. In this case, we created a model that will make this prediction, so that the company can take some action to prevent the fraud, and avoid losses.

We used a dataset provided by the company to train the model. The independent variables and the dependent variable (target) were provided so that this task could be done. We train and validate the model with a part of the data set (training and validation data) and evaluate the performance of the model in production with the test data. 

We used a [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), class from scikit-learn in version 1.4.2. To arrive at the final version, we used the aforementioned model and tuned the hyperparameters with the [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) class. The optimal hyperparameters were:

* n_estimators: 150 
* max_depth: 8

The rest of the hyperparameters were kept by default.

Model is saved as a pickle file stored in the [s3 bucket in AWS](https://aws.amazon.com/s3/). To use the latest model available in production, you must download it via AWS, like this: `

```
final_model = s3_client.get_object(
    Bucket=BUCKET_NAME_MODEL,
    Key=f'pickles/extracted_at={current_date}/model_{tag_value}.pkl')
    final_model = pickle.loads(final_model['Body'].read())
```

Model training and validation was done in component [train_pipeline/components/train_model.py](https://github.com/vitorbeltrao/fraud-detection/blob/main/functions/train_pipeline/components/train_model.py).

Model testing was done in component [train_pipeline/components/test_model.py](https://github.com/vitorbeltrao/fraud-detection/blob/main/functions/train_pipeline/components/test_model.py).

## Intended Use

This fraud detection model is designed to help financial institutions identify potentially fraudulent transactions in real-time. By accurately predicting and flagging suspicious activities, the model enables banks to take immediate action to prevent financial losses and protect their customers. The primary use of this model is for financial institutions, security analysts, and researchers in the field of cybersecurity and finance. It is intended to enhance the security of banking operations and improve the overall trust in financial systems.

## Training Data

The dataset was provided by the company. For more information, contact the author of the project. The means of communication are in the [README](https://github.com/vitorbeltrao/fraud-detection/blob/main/README.md).

The model was trained on a training dataset. The training dataset represents 85% of the total dataset.

First, we split the dataset into an array of independent variables (X) and the target vector (y); Then we trained the pipeline using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) with cross validation using 5 folds.

The pipeline used for model training and validation consists of several key steps:

1. Preprocessing:

* Custom Transformer: A custom transformer, CountryMapper, is used to map country codes to specified categories. It maps 'BR', 'AR', 'UY', and 'US' to their respective values and all other country codes to 'outros'.
* Qualitative Features: Preprocessing of qualitative features using SimpleImputer with the strategy 'most_frequent'.
* entrega_doc_2 Feature: This feature undergoes preprocessing using SimpleImputer with a constant fill value and OneHotEncoder with drop='first'.
* entrega_doc_3 Feature: Preprocessing of this feature involves SimpleImputer with the 'most_frequent' strategy and OneHotEncoder with drop='if_binary'.
* Country Feature: The country feature is processed using the custom CountryMapper, followed by SimpleImputer with 'most_frequent' strategy and OneHotEncoder with drop='first'.
* Quantitative Features: Continuous quantitative features are processed using SimpleImputer with 'median' strategy and StandardScaler.

2. Column Transformation:

The ColumnTransformer method is applied to transform the respective features using the preprocessing pipelines defined above. This results in the preprocessing of different types of features appropriately before feeding them into the model.

3. Model Training:

The final model is a pipeline that includes the preprocessing steps and the machine learning model. Here, RandomUnderSampler is used to handle class imbalance, and a RandomForestClassifier is used for classification.

4. Saving the Model:

The best estimator from the pipeline is saved in a pickle file for deployment in production.

## Evaluation Data

The data was trained and evaluated on 85% of all available data, via cross-validation. All performance metrics were stored in a [DynamoDB](https://aws.amazon.com/dynamodb/). We are currently collecting the accuracy, F1 score and brier score metrics.

The data was tested on the remaining 15% ​​of the total dataset, where we never had contact with that dataset. For this, we downloaded the entire inference pipeline stored in the pickle file in the last step and tested it on the test dataset. The evaluation metrics, you can see in the topic below.

Also, regarding model evaluation, we implant a model evaluation historical record into the test data, this evaluation of historical data in test data is being stored in a DynamoDB. All of this, in order to verify **model drift**, that is, if the model starts to perform poorly in the future, we will be aware of this in advance and consequently take the necessary actions such as retraining this model. We are currently collecting the accuracy, F1 score and brier score metrics.

![Recording model scores](https://github.com/vitorbeltrao/risk_assessment/blob/main/Images/Recording_model_scores.png?raw=true)

## Metrics

**Train and validation metrics:** 

* The balanced accuracy score for the training set is 0.752, and for the validation set, it is 0.733.
* The F1 score for the training set is 0.341, and for the validation set, it is 0.321.
* The negative Brier score for the training set is -0.093, and for the validation set, it is -0.095.

**Test metrics:** 

* The balanced accuracy score is 0.734.
* The F1 score is 0.330.
* The negative Brier score is -0.117.

## Ethical Considerations

Remembering that the model should be used only for didactic purposes, not for professional purposes.

## Caveats and Recommendations

The dataset is a outdated sample and cannot adequately be used as a statistical representation of the population. It is recommended to use the dataset for training purpose on ML classification or related problems.