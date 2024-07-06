# Fraud Detection System - v0.0.1

## Table of Contents

1. [Project Description](#description)
2. [Files Description](#files)
3. [Using the API](#api)
4. [Model and Data Diagnostics](#diagnostics)
5. [Orchestration](#orchestration)
6. [Licensing and Authors](#licensingandauthors)
***

## Project Description <a name="description"></a>

In the current financial landscape, banks must not only acquire new customers but also safeguard their existing customer base from fraudulent activities to prevent financial losses and maintain trust. The aim of this project is to predict fraudulent transactions in real-time, enabling banks to take necessary actions to prevent these frauds and protect their customers. This project is designed for students, academics, or research purposes.

To carry out the project, it was necessary an architecture that met the low budget, in addition to being functional and scalable. Therefore, we use the architecture below:

![Fraud Detection Architecture](https://github.com/vitorbeltrao/Pictures/blob/main/fraud_data_architecture.jpg?raw=true)
***

## Files Description <a name="files"></a>

In "fraud-detection" repository we have:

* **.github**: Inside this folder, we have the github actions workflows, responsible for the project's CI/CD process.

* **continuous_training_system**: Inside this folder, we have the entire infrastructure as code of the pipeline training stage, which you can see in the architecture image. This file is responsible for creating all instances in AWS for this continuous training step.

* **functions**: Inside this folder are the model's training and inference codes. Everything related to Python codes, containerization with Docker, virtual environments and lambda functions is inside this folder.

* **inference_system**: Inside this folder, we have the entire infrastructure as code of the pipeline inference stage, which you can see in the architecture image. This file is responsible for creating all instances in AWS for this inference step.

* **model_data_drift_system**: Inside this folder, we have the entire infrastructure as code of the pipeline inference stage, which you can see in the architecture image. This file is responsible for creating all instances in AWS for this model and data drift step.

* **monitoring_system**: Inside this folder we have the entire infrastructure as code for system monitoring of lambda functions, which you can see in the architecture image. This file is responsible for creating all instances in AWS for this system monitoring step.

* **prototype_notebooks**: Here are the notebooks used for prototyping and experimenting with machine learning models, as well as the exploratory analyzes done to obtain information and insights from the dataset.

* **repo-image**: Inside this folder, we have all the infrastructure as code for creating ECR instances on AWS, to store our docker images.

* **tests**: Folder that contains all the unit tests of the functions and the data tests (which serve to check whether the data continues to follow the same structure and distribution). This is all tested in the CI/CD stage. The system will only go to production if it passes all these tests.

* **deploy.sh**: File that orchestrates the sequence that must be followed when deploying the system. It is called in the CI/CD step as the last step.

* **justfile**: File used to write in a more elegant way the commands necessary to carry out some actions necessary to create the necessary parts of the project.

* **model_card.md file**: Documentation of the created machine learning model.
***

## Using the API <a name="api"></a>

To test and send a request to the created API, I used [postman](https://www.postman.com/).

![Postman inference](https://github.com/vitorbeltrao/Pictures/blob/main/fraud_detection_postman.png?raw=true)
***

## Model and Data Diagnostics <a name="diagnostics"></a>

### Model Drift

To check the model score and check the model drift, we are doing it by the following process:

![The model scoring process](https://github.com/vitorbeltrao/risk_assessment/blob/main/Images/The_model_scoring_proces.png?raw=true)

Model scoring should happen at regular intervals. You should read fresh data, make predictions with a deployed model, and measure the model's errors by comparing predicted and actual values.

If your model begins to perform worse than it had before, then you're a victim of model drift. When model drift occurs, you need to retrain and re-deploy your model.

The file containing the detailed evaluation metrics, including the historical records of the test data evaluation, to verify the model drift, are being detailed in the [model card](https://github.com/vitorbeltrao/risk_assessment/blob/main/model_card.md).

In the *functions/monitoring_data_model_drift* folder, we create the scripts that check the model drift through three functions: *Raw Comparison Test*, *Parametric Significance Test* and *Non-Parametric Outlier Test*. For more information on what each of these tests does, visit the respective folder with the scripts and see the documentation for the functions.

Finally, after testing these three functions that verify the model drift, we choose by voting whether the model suffered model drift or not, that is, if two of these functions show model drift, then we have model drift and vice versa. If we don't have model drift, then we keep the current model in production; if we have model drift then we must retrain and re-deploy the model. 

**Our system automatically fires a trigger with an SNS Topic to automatically retrain the model!**

**Our system automatically sends an email to the person in charge, in case the model drift occurs!**

### Data Drift

Data drift also happens at regular intervals. We have a reference dataset, which was the first dataset that we trained, validated and tested the model before going into production and everything went well on that dataset. Over time, more data enters the data lake and the idea here is to compare the entire historical dataset (reference + new data coming in regularly) with the reference dataset that was the first one we trained.

To make this comparison, we used an open source library that is [Evidently](https://www.evidentlyai.com/). For more information read the documentation at the highlighted link. Finally, we generate HTML files with the entire report on the data drift for the user stored in a S3 bucket.
***

## Orchestration <a name="orchestration"></a>

The orchestration of this project is done automatically by lambdas, with the help of the **EventBridge: Scheduler instance** and **SNS Topic**.

**With that, we have a model working almost 100% automatically without much manual intervention on the part of those responsible. Of course, it is recommended that sometimes those responsible take a look at the reports, to verify that everything is fine**.
***

## Licensing and Author <a name="licensingandauthors"></a>

Vítor Beltrão - Data Scientist / Machine Learning Engineer

Reach me at: 

- vitorbeltraoo@hotmail.com

- [linkedin](https://www.linkedin.com/in/vitorbeltrao/)

- [github](https://github.com/vitorbeltrao)

- [medium](https://medium.com/@vitorbeltrao300)

Licensing: [MIT LICENSE](https://github.com/vitorbeltrao/fraud-detection/blob/main/LICENSE)