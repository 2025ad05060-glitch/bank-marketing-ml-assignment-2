# Bank Marketing Classification Using Machine Learning

## Machine Learning Assignment 2

This project implements and compares multiple machine learning classification algorithms on the UCI Bank Marketing dataset. The objective is to predict whether a bank customer will subscribe to a term deposit.

A Streamlit web application is also developed to allow users to select a trained machine learning model, upload test data, and view the corresponding prediction and evaluation results.

---

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

---

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

---

## Machine Learning Models

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

---

## Evaluation Metrics

The models were evaluated using the following six metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Confusion matrices and classification reports were also generated.

---

## Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC Score |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.893288 | 0.771749 | 0.663158 | 0.178639 | 0.281459 | 0.306530 |
| Decision Tree | 0.830919 | 0.613178 | 0.298201 | 0.328922 | 0.312809 | 0.217011 |
| KNN | 0.887427 | 0.702749 | 0.552910 | 0.197543 | 0.291086 | 0.283266 |
| Gaussian Naive Bayes | 0.845184 | 0.751441 | 0.370846 | 0.464083 | 0.412259 | 0.327093 |
| Random Forest | 0.895389 | 0.788788 | 0.637255 | 0.245747 | 0.354707 | 0.351844 |

---

## Observations

The Random Forest Classifier achieved the highest overall Accuracy, AUC and MCC score among the five evaluated models.

Random Forest achieved:

- Accuracy: 0.895389
- AUC: 0.788788
- Precision: 0.637255
- Recall: 0.245747
- F1 Score: 0.354707
- MCC Score: 0.351844

Gaussian Naive Bayes achieved the highest Recall and F1 Score among the evaluated models.

Therefore, there is no single model that dominates every evaluation metric. Random Forest provides the strongest overall performance according to Accuracy, AUC and MCC, whereas Gaussian Naive Bayes detects a larger proportion of the positive class, as indicated by its higher Recall.

---

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

---

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

---

## How to Run the Application

### 1. Install the required libraries

```bash
pip install -r requirements.txt
