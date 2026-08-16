# Bank Marketing Classification Using Machine Learning

## Machine Learning Assignment 2
## Problem Statement

The aim of this assignment is to build classification models for predicting
whether a bank customer will subscribe to a term deposit or not.

For this purpose, I used the Bank Marketing dataset from the UCI Machine
Learning Repository. I trained and tested five classification models on the
same dataset and compared their performance using Accuracy, AUC, Precision,
Recall, F1 Score and MCC.

I also developed a Streamlit application where test data can be uploaded and
the performance of different trained models can be checked interactively.


## Dataset

The project uses the UCI Bank Marketing dataset.

The dataset contains information related to direct marketing campaigns of a Portuguese banking institution.

### Target Variable

The target variable is:

- `y`

It indicates whether the customer subscribed to a term deposit.

- `yes` = Customer subscribed
- `no` = Customer did not subscribe

The `duration` attribute was removed during data preparation.



## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the Bank Marketing dataset.
2. Removed the `duration` column.
3. Separated input features and target variable.
4. Identified numerical and categorical features.
5. Applied preprocessing to numerical and categorical variables.
6. Converted categorical variables using One-Hot Encoding.
7. Split the dataset into training and testing sets.
8. Saved the fitted preprocessing pipeline for use by the Streamlit application.

After preprocessing:

- Training samples: 36,168
- Testing samples: 9,043
- Number of original input features: 15
- Number of processed features: 50



## Machine Learning Models

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier



## Evaluation Metrics

The models were evaluated using the following six metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Confusion matrices and classification reports were also generated.



## Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC Score |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.893288 | 0.771749 | 0.663158 | 0.178639 | 0.281459 | 0.306530 |
| Decision Tree | 0.830919 | 0.613178 | 0.298201 | 0.328922 | 0.312809 | 0.217011 |
| KNN | 0.887427 | 0.702749 | 0.552910 | 0.197543 | 0.291086 | 0.283266 |
| Gaussian Naive Bayes | 0.845184 | 0.751441 | 0.370846 | 0.464083 | 0.412259 | 0.327093 |
| Random Forest | 0.895389 | 0.788788 | 0.637255 | 0.245747 | 0.354707 | 0.351844 |



## Observations

From the results, Random Forest gave the best overall performance among the five models. It achieved the highest Accuracy, AUC and MCC score.

The Random Forest results were:

- Accuracy: 0.895389
- AUC: 0.788788
- Precision: 0.637255
- Recall: 0.245747
- F1 Score: 0.354707
- MCC Score: 0.351844

However, Gaussian Naive Bayes gave better Recall and F1 Score compared to the other models. This shows that the best model can depend on which evaluation metric is considered.

Overall, Random Forest performed better when Accuracy, AUC and MCC are considered together.



## Streamlit Web Application

An interactive Streamlit application was developed for the project.

The application allows the user to:

- View the performance comparison of all five machine learning models.
- Select a classification model.
- Upload the test dataset in CSV format.
- Generate predictions using the selected trained model.
- View Accuracy, AUC, Precision, Recall, F1 Score and MCC Score.
- View the confusion matrix.
- View the classification report.
- View individual prediction results and subscription probabilities.



## Project Structure

ML_ASSIGNMENT_2/

- `01_data_preparation.py` - Data preprocessing and model training program
- `app.py` - Streamlit web application
- `requirements.txt` - Python package dependencies
- `test_data.csv` - Test dataset used for model evaluation
- `DATA/` - Dataset files
- `model/` - Saved preprocessing pipeline and trained models
- `RESULTS/` - Model comparison results

### Saved Model Files

The `model` folder contains:

- `preprocessor.pkl`
- `logistic_regression.pkl`
- `decision_tree.pkl`
- `knn.pkl`
- `naive_bayes.pkl`
- `random_forest.pkl`


## How to Run the Application

### Step 1: Install the required libraries

```bash
pip install -r requirements.txt
```

### Step 2: Run the Streamlit application

After installing the required libraries, run:

```bash
streamlit run app.py
```

The application will open in the web browser.

## Deployment

The application was also deployed using Streamlit Community Cloud. The deployed application can be used to
select any of the five trained models, upload the test dataset and view the prediction and evaluation results.
