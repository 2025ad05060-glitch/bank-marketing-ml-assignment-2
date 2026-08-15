import streamlit as st
import pandas as pd
import joblib

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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# LOAD SAVED PREPROCESSOR AND MODELS
# ============================================================

@st.cache_resource
def load_models():

    preprocessor = joblib.load("model/preprocessor.pkl")

    models = {
        "Logistic Regression":
            joblib.load("model/logistic_regression.pkl"),

        "Decision Tree":
            joblib.load("model/decision_tree.pkl"),

        "K-Nearest Neighbors (KNN)":
            joblib.load("model/knn.pkl"),

        "Gaussian Naive Bayes":
            joblib.load("model/naive_bayes.pkl"),

        "Random Forest":
            joblib.load("model/random_forest.pkl")
    }

    return preprocessor, models


preprocessor, models = load_models()


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Bank Marketing Classification App")

st.write(
    "Machine Learning Assignment 2 — "
    "Prediction of Term Deposit Subscription"
)

st.info(
    "This application compares five classification models "
    "using the UCI Bank Marketing dataset."
)


# ============================================================
# MODEL COMPARISON RESULTS
# ============================================================

st.header("📊 Model Performance Comparison")

try:

    comparison_df = pd.read_csv(
        "RESULTS/model_comparison.csv"
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

except FileNotFoundError:

    st.warning(
        "model_comparison.csv was not found in the RESULTS folder."
    )


# ============================================================
# MODEL SELECTION
# ============================================================

st.header("🤖 Select Machine Learning Model")

selected_model_name = st.selectbox(
    "Choose a classification model:",
    list(models.keys())
)

selected_model = models[selected_model_name]

st.success(
    f"Selected model: {selected_model_name}"
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("📁 Upload Test Data")

st.write(
    "Upload a CSV file containing the Bank Marketing "
    "test dataset."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


# ============================================================
# PROCESS UPLOADED DATA
# ============================================================

if uploaded_file is not None:

    try:

        test_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")

        st.dataframe(
            test_df.head(10),
            use_container_width=True
        )


        # ====================================================
        # CHECK TARGET COLUMN
        # ====================================================

        if "y" not in test_df.columns:

            st.error(
                "The uploaded CSV must contain the target "
                "column named 'y'."
            )

        else:

            # Separate features and target
            X_uploaded = test_df.drop(
                columns=["y"]
            )

            y_uploaded = test_df["y"].map(
                {
                    "no": 0,
                    "yes": 1
                }
            )


            # =================================================
            # VALIDATE TARGET VALUES
            # =================================================

            if y_uploaded.isnull().any():

                st.error(
                    "The 'y' column must contain only "
                    "'yes' and 'no' values."
                )

            else:

                # =============================================
                # PREPROCESS DATA
                # =============================================

                X_processed = preprocessor.transform(
                    X_uploaded
                )


                # =============================================
                # MAKE PREDICTIONS
                # =============================================

                predictions = selected_model.predict(
                    X_processed
                )

                probabilities = (
                    selected_model.predict_proba(
                        X_processed
                    )[:, 1]
                )


                # =============================================
                # CALCULATE METRICS
                # =============================================

                accuracy = accuracy_score(
                    y_uploaded,
                    predictions
                )

                auc = roc_auc_score(
                    y_uploaded,
                    probabilities
                )

                precision = precision_score(
                    y_uploaded,
                    predictions,
                    zero_division=0
                )

                recall = recall_score(
                    y_uploaded,
                    predictions,
                    zero_division=0
                )

                f1 = f1_score(
                    y_uploaded,
                    predictions,
                    zero_division=0
                )

                mcc = matthews_corrcoef(
                    y_uploaded,
                    predictions
                )


                # =============================================
                # DISPLAY METRICS
                # =============================================

                st.header(
                    "📈 Evaluation Metrics"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Accuracy",
                        f"{accuracy:.4f}"
                    )

                    st.metric(
                        "Precision",
                        f"{precision:.4f}"
                    )

                with col2:

                    st.metric(
                        "AUC Score",
                        f"{auc:.4f}"
                    )

                    st.metric(
                        "Recall",
                        f"{recall:.4f}"
                    )

                with col3:

                    st.metric(
                        "F1 Score",
                        f"{f1:.4f}"
                    )

                    st.metric(
                        "MCC Score",
                        f"{mcc:.4f}"
                    )


                # =============================================
                # CONFUSION MATRIX
                # =============================================

                st.header(
                    "🔢 Confusion Matrix"
                )

                cm = confusion_matrix(
                    y_uploaded,
                    predictions
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=[
                        "Actual No",
                        "Actual Yes"
                    ],
                    columns=[
                        "Predicted No",
                        "Predicted Yes"
                    ]
                )

                st.dataframe(
                    cm_df,
                    use_container_width=True
                )


                # =============================================
                # CLASSIFICATION REPORT
                # =============================================

                st.header(
                    "📋 Classification Report"
                )

                report = classification_report(
                    y_uploaded,
                    predictions,
                    target_names=[
                        "No",
                        "Yes"
                    ],
                    output_dict=True,
                    zero_division=0
                )

                report_df = pd.DataFrame(
                    report
                ).transpose()

                st.dataframe(
                    report_df,
                    use_container_width=True
                )


                # =============================================
                # DISPLAY PREDICTIONS
                # =============================================

                st.header(
                    "🔍 Prediction Results"
                )

                prediction_output = test_df.copy()

                prediction_output[
                    "Predicted_y"
                ] = [
                    "yes" if value == 1 else "no"
                    for value in predictions
                ]

                prediction_output[
                    "Subscription_Probability"
                ] = probabilities

                st.dataframe(
                    prediction_output.head(50),
                    use_container_width=True
                )


                st.success(
                    "Prediction and evaluation completed successfully!"
                )


    except Exception as e:

        st.error(
            f"Error while processing the uploaded file: {e}"
        )


# ============================================================
# INSTRUCTIONS
# ============================================================

else:

    st.info(
        "Please upload test_data.csv to evaluate "
        "the selected model."
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("ℹ️ Project Information")

st.write(
    """
    Dataset: UCI Bank Marketing Dataset

    Target Variable:
    y — whether the customer subscribed to a term deposit.

    Models implemented:
    Logistic Regression, Decision Tree, KNN,
    Gaussian Naive Bayes and Random Forest.

    Evaluation metrics:
    Accuracy, AUC, Precision, Recall, F1 Score
    and Matthews Correlation Coefficient (MCC).
    """
)
