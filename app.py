import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.title("ML Model Evaluation & Deployment Dashboard")
st.sidebar.header("User Input Controls")

# 1. Dataset Upload Option
uploaded_file = st.sidebar.file_uploader("Upload your test data (CSV)", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Test Data Preview")
    st.dataframe(test_df.head())

    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    # 2. Model Selection Dropdown
    model_name = st.sidebar.selectbox(
        "Select Classification Model", 
        ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"]
    )

    # Load respective model
    file_mapping = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "kNN": "kinn.pkl" if False else "k_nn.pkl", # Adjust based on exact filename saved
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest": "random_forest.pkl"
    }
    
    # Simple mapping load check
    try:
        filename = file_mapping[model_name]
        model = joblib.load(filename)
        
        if model_name in ["Logistic Regression", "kNN"]:
            scaler = joblib.load("scaler.pkl")
            X_test_input = scaler.transform(X_test)
        else:
            X_test_input = X_test

        y_pred = model.predict(X_test_input)

        # 3. Display Evaluation Metrics
        st.subheader(f"Performance Metrics for: {model_name}")
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(acc, 4))
        col2.metric("Precision", round(prec, 4))
        col3.metric("Recall", round(rec, 4))

        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", round(f1, 4))
        col5.metric("MCC Score", round(mcc, 4))

        # 4. Confusion Matrix & Classification Report
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    except Exception as e:
        st.error(f"Error loading model or running prediction: {e}. Make sure all .pkl files are uploaded to GitHub.")
else:
    st.info("Please upload your `test_data.csv` file using the sidebar to proceed.")
