import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

df = pd.read_csv("data/bank-full.csv", sep=";")

print("=" * 70)
print("BANK MARKETING - DATA PREPARATION")
print("=" * 70)


# ============================================================
# STEP 2: REMOVE DURATION
# ============================================================

# 'duration' is excluded because it is known only during/after
# the current customer contact and can cause information leakage.

df = df.drop(columns=["duration"])

print("\n1. Removed column: duration")
print("Remaining columns:", df.shape[1])


# ============================================================
# STEP 3: SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})

print("\n2. FEATURES AND TARGET")
print("Number of features:", X.shape[1])
print("Target values:")
print(y.value_counts())


# ============================================================
# STEP 4: IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()

print("\n3. NUMERICAL FEATURES")
print(numerical_features)

print("\n4. CATEGORICAL FEATURES")
print(categorical_features)


# ============================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n5. TRAIN-TEST SPLIT")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# STEP 6: PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ============================================================
# STEP 7: FIT PREPROCESSING ON TRAINING DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# ============================================================
# STEP 8: DISPLAY FINAL DATA SHAPE
# ============================================================

print("\n6. PROCESSED DATA")
print("Training data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)

print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETED SUCCESSFULLY")
print("=" * 70)


# ============================================================
# STEP 9: LOGISTIC REGRESSION MODEL
# ============================================================

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

print("\n" + "=" * 70)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 70)


# Create Logistic Regression model
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# Train the model
logistic_model.fit(X_train_processed, y_train)

print("\nModel training completed successfully.")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

y_pred = logistic_model.predict(X_test_processed)

# Probability of positive class (y = 1)
y_prob = logistic_model.predict_proba(X_test_processed)[:, 1]


# ============================================================
# CALCULATE EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nLOGISTIC REGRESSION EVALUATION METRICS")
print("-" * 50)

print(f"Accuracy  : {accuracy:.6f}")
print(f"AUC Score : {auc:.6f}")
print(f"Precision : {precision:.6f}")
print(f"Recall    : {recall:.6f}")
print(f"F1 Score  : {f1:.6f}")
print(f"MCC Score : {mcc:.6f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX")
print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No", "Yes"]
))


print("=" * 70)
print("LOGISTIC REGRESSION COMPLETED SUCCESSFULLY")
print("=" * 70)









# ============================================================
# STEP 10: DECISION TREE CLASSIFIER
# ============================================================

from sklearn.tree import DecisionTreeClassifier

print("\n" + "=" * 70)
print("MODEL 2: DECISION TREE CLASSIFIER")
print("=" * 70)


# Create Decision Tree model
decision_tree_model = DecisionTreeClassifier(
    random_state=42
)


# Train the model
decision_tree_model.fit(X_train_processed, y_train)

print("\nModel training completed successfully.")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

dt_pred = decision_tree_model.predict(X_test_processed)

# Probability of positive class
dt_prob = decision_tree_model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# CALCULATE EVALUATION METRICS
# ============================================================

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_auc = roc_auc_score(y_test, dt_prob)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)
dt_mcc = matthews_corrcoef(y_test, dt_pred)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nDECISION TREE EVALUATION METRICS")
print("-" * 50)

print(f"Accuracy  : {dt_accuracy:.6f}")
print(f"AUC Score : {dt_auc:.6f}")
print(f"Precision : {dt_precision:.6f}")
print(f"Recall    : {dt_recall:.6f}")
print(f"F1 Score  : {dt_f1:.6f}")
print(f"MCC Score : {dt_mcc:.6f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

dt_cm = confusion_matrix(y_test, dt_pred)

print("\nCONFUSION MATRIX")
print(dt_cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_test,
    dt_pred,
    target_names=["No", "Yes"]
))


print("=" * 70)
print("DECISION TREE COMPLETED SUCCESSFULLY")
print("=" * 70)





# ============================================================
# STEP 11: K-NEAREST NEIGHBORS CLASSIFIER
# ============================================================

from sklearn.neighbors import KNeighborsClassifier

print("\n" + "=" * 70)
print("MODEL 3: K-NEAREST NEIGHBORS (KNN)")
print("=" * 70)


# Create KNN model
# k = 5 means the 5 nearest neighbors are considered
knn_model = KNeighborsClassifier(
    n_neighbors=5
)


# Train the model
knn_model.fit(X_train_processed, y_train)

print("\nModel training completed successfully.")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

knn_pred = knn_model.predict(X_test_processed)

# Probability of positive class
knn_prob = knn_model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# CALCULATE EVALUATION METRICS
# ============================================================

knn_accuracy = accuracy_score(y_test, knn_pred)
knn_auc = roc_auc_score(y_test, knn_prob)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)
knn_mcc = matthews_corrcoef(y_test, knn_pred)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nKNN EVALUATION METRICS")
print("-" * 50)

print(f"Accuracy  : {knn_accuracy:.6f}")
print(f"AUC Score : {knn_auc:.6f}")
print(f"Precision : {knn_precision:.6f}")
print(f"Recall    : {knn_recall:.6f}")
print(f"F1 Score  : {knn_f1:.6f}")
print(f"MCC Score : {knn_mcc:.6f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

knn_cm = confusion_matrix(y_test, knn_pred)

print("\nCONFUSION MATRIX")
print(knn_cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_test,
    knn_pred,
    target_names=["No", "Yes"]
))


print("=" * 70)
print("KNN COMPLETED SUCCESSFULLY")
print("=" * 70)







# ============================================================
# STEP 12: GAUSSIAN NAIVE BAYES CLASSIFIER
# ============================================================

from sklearn.naive_bayes import GaussianNB

print("\n" + "=" * 70)
print("MODEL 4: GAUSSIAN NAIVE BAYES")
print("=" * 70)


# Create Gaussian Naive Bayes model
nb_model = GaussianNB()


# Train the model
nb_model.fit(X_train_processed, y_train)

print("\nModel training completed successfully.")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

nb_pred = nb_model.predict(X_test_processed)

# Probability of positive class
nb_prob = nb_model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# CALCULATE EVALUATION METRICS
# ============================================================

nb_accuracy = accuracy_score(y_test, nb_pred)
nb_auc = roc_auc_score(y_test, nb_prob)
nb_precision = precision_score(y_test, nb_pred)
nb_recall = recall_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred)
nb_mcc = matthews_corrcoef(y_test, nb_pred)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nNAIVE BAYES EVALUATION METRICS")
print("-" * 50)

print(f"Accuracy  : {nb_accuracy:.6f}")
print(f"AUC Score : {nb_auc:.6f}")
print(f"Precision : {nb_precision:.6f}")
print(f"Recall    : {nb_recall:.6f}")
print(f"F1 Score  : {nb_f1:.6f}")
print(f"MCC Score : {nb_mcc:.6f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

nb_cm = confusion_matrix(y_test, nb_pred)

print("\nCONFUSION MATRIX")
print(nb_cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_test,
    nb_pred,
    target_names=["No", "Yes"]
))


print("=" * 70)
print("NAIVE BAYES COMPLETED SUCCESSFULLY")
print("=" * 70)














# ============================================================
# STEP 13: RANDOM FOREST CLASSIFIER
# ============================================================

from sklearn.ensemble import RandomForestClassifier

print("\n" + "=" * 70)
print("MODEL 5: RANDOM FOREST CLASSIFIER")
print("=" * 70)


# Create Random Forest model
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# Train the model
random_forest_model.fit(
    X_train_processed,
    y_train
)

print("\nModel training completed successfully.")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

rf_pred = random_forest_model.predict(
    X_test_processed
)

# Probability of positive class
rf_prob = random_forest_model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# CALCULATE EVALUATION METRICS
# ============================================================

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_mcc = matthews_corrcoef(y_test, rf_pred)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nRANDOM FOREST EVALUATION METRICS")
print("-" * 50)

print(f"Accuracy  : {rf_accuracy:.6f}")
print(f"AUC Score : {rf_auc:.6f}")
print(f"Precision : {rf_precision:.6f}")
print(f"Recall    : {rf_recall:.6f}")
print(f"F1 Score  : {rf_f1:.6f}")
print(f"MCC Score : {rf_mcc:.6f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

rf_cm = confusion_matrix(
    y_test,
    rf_pred
)

print("\nCONFUSION MATRIX")
print(rf_cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_test,
    rf_pred,
    target_names=["No", "Yes"]
))


print("=" * 70)
print("RANDOM FOREST COMPLETED SUCCESSFULLY")
print("=" * 70)




# ============================================================
# STEP 14: SAVE MODELS, PREPROCESSOR, RESULTS AND TEST DATA
# ============================================================

import os
import joblib


# Create folders if they do not already exist
os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

joblib.dump(
    preprocessor,
    "model/preprocessor.pkl"
)

print("\nPreprocessor saved successfully.")


# ============================================================
# SAVE TRAINED MODELS
# ============================================================

joblib.dump(
    logistic_model,
    "model/logistic_regression.pkl"
)

joblib.dump(
    decision_tree_model,
    "model/decision_tree.pkl"
)

joblib.dump(
    knn_model,
    "model/knn.pkl"
)

joblib.dump(
    nb_model,
    "model/naive_bayes.pkl"
)

joblib.dump(
    random_forest_model,
    "model/random_forest.pkl"
)

print("All trained models saved successfully.")


# ============================================================
# CREATE MODEL COMPARISON TABLE
# ============================================================

comparison_results = pd.DataFrame({
    "ML Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Gaussian Naive Bayes",
        "Random Forest"
    ],

    "Accuracy": [
        accuracy,
        dt_accuracy,
        knn_accuracy,
        nb_accuracy,
        rf_accuracy
    ],

    "AUC": [
        auc,
        dt_auc,
        knn_auc,
        nb_auc,
        rf_auc
    ],

    "Precision": [
        precision,
        dt_precision,
        knn_precision,
        nb_precision,
        rf_precision
    ],

    "Recall": [
        recall,
        dt_recall,
        knn_recall,
        nb_recall,
        rf_recall
    ],

    "F1 Score": [
        f1,
        dt_f1,
        knn_f1,
        nb_f1,
        rf_f1
    ],

    "MCC Score": [
        mcc,
        dt_mcc,
        knn_mcc,
        nb_mcc,
        rf_mcc
    ]
})


# Round results to 6 decimal places
comparison_results = comparison_results.round(6)


# Display comparison table
print("\n" + "=" * 90)
print("FINAL MODEL COMPARISON")
print("=" * 90)

print(comparison_results.to_string(index=False))


# Save comparison results
comparison_results.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully.")


# ============================================================
# CREATE TEST DATA CSV
# ============================================================

test_data = X_test.copy()

# Add the actual target values back
test_data["y"] = y_test.map({
    0: "no",
    1: "yes"
})


# Save test data
test_data.to_csv(
    "test_data.csv",
    index=False
)

print("test_data.csv saved successfully.")


print("\n" + "=" * 70)
print("ALL PROJECT FILES SAVED SUCCESSFULLY")
print("=" * 70)
