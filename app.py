"""
Machine Learning Assignment 2 - Streamlit Web Application
Interactive app for spam classification using multiple ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Spam Classification App")

@st.cache_data
def load_evaluation_results():
    """Load pre-computed evaluation results"""
    try:
        with open('model/evaluation_results.json', 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        st.error("Evaluation results not found. Please train models first.")
        return None

@st.cache_resource
def load_model(model_name):
    """Load a trained model"""
    model_file_map = {
        'Logistic Regression': 'model/logistic_regression.pkl',
        'Decision Tree': 'model/decision_tree.pkl',
        'K-Nearest Neighbors': 'model/k_nearest_neighbors.pkl',
        'Naive Bayes': 'model/naive_bayes.pkl',
        'Random Forest': 'model/random_forest.pkl',
        'XGBoost': 'model/xgboost.pkl'
    }

    try:
        model = joblib.load(model_file_map[model_name])
        scaler = joblib.load('model/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error(f"Model file not found for {model_name}")
        return None, None

def plot_confusion_matrix(cm, model_name):
    """Create confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Spam', 'Spam'],
                yticklabels=['Not Spam', 'Spam'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(f'Confusion Matrix - {model_name}')
    return fig

def main():
    """Main application function"""

    # Title
    st.title("Spam Email Classification System")
    st.write("Machine Learning Assignment 2")

    st.markdown("---")

    # Model selection in main area (more visible)
    st.subheader("Select Classification Model")
    model_name = st.selectbox(
        "Choose a model:",
        (
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        )
    )

    st.markdown("---")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Model Performance", "Make Predictions", "Model Comparison"])

    # Tab 1: Model Performance
    with tab1:
        st.header(f"Performance Metrics - {model_name}")

        results = load_evaluation_results()

        if results and model_name in results:
            model_results = results[model_name]
            metrics = model_results['metrics']

            # Display metrics in columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                st.metric("AUC Score", f"{metrics['auc']:.4f}")

            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
                st.metric("Recall", f"{metrics['recall']:.4f}")

            with col3:
                st.metric("F1 Score", f"{metrics['f1']:.4f}")
                st.metric("MCC Score", f"{metrics['mcc']:.4f}")

            st.markdown("---")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = np.array(model_results['confusion_matrix'])
            fig = plot_confusion_matrix(cm, model_name)
            st.pyplot(fig)

            # Classification Report
            st.subheader("Classification Report")
            clf_report = model_results['classification_report']

            report_df = pd.DataFrame({
                'Class': ['Not Spam (0)', 'Spam (1)'],
                'Precision': [clf_report['0']['precision'], clf_report['1']['precision']],
                'Recall': [clf_report['0']['recall'], clf_report['1']['recall']],
                'F1-Score': [clf_report['0']['f1-score'], clf_report['1']['f1-score']],
                'Support': [clf_report['0']['support'], clf_report['1']['support']]
            })

            st.dataframe(report_df, use_container_width=True)

    # Tab 2: Make Predictions
    with tab2:
        st.header("Make Predictions on Test Data")

        st.write("Upload a CSV file with test data to make predictions.")
        st.write("The CSV should have the same 56 features as the training data.")

        st.info("**For testing:** A sample test file is available at `data/test_data.csv` (10 samples)")

        uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

        if uploaded_file is not None:
            try:
                test_data = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Shape: {test_data.shape}")

                with st.expander("Preview Data"):
                    st.dataframe(test_data.head())

                model, scaler = load_model(model_name)

                if model is not None and scaler is not None:
                    has_label = 'label' in test_data.columns

                    if has_label:
                        X_test = test_data.drop('label', axis=1)
                        y_test = test_data['label']
                    else:
                        X_test = test_data
                        y_test = None

                    if st.button("Predict"):
                        with st.spinner("Making predictions..."):
                            X_test_scaled = scaler.transform(X_test)
                            predictions = model.predict(X_test_scaled)

                            if hasattr(model, 'predict_proba'):
                                probabilities = model.predict_proba(X_test_scaled)
                            else:
                                probabilities = None

                            st.success("Predictions completed!")

                            results_df = pd.DataFrame({
                                'Sample': range(1, len(predictions) + 1),
                                'Prediction': ['Spam' if p == 1 else 'Not Spam' for p in predictions]
                            })

                            if probabilities is not None:
                                results_df['Confidence'] = probabilities.max(axis=1)

                            if has_label:
                                results_df['Actual'] = ['Spam' if y == 1 else 'Not Spam' for y in y_test]
                                results_df['Correct'] = predictions == y_test.values

                            st.dataframe(results_df, use_container_width=True)

                            if has_label:
                                accuracy = (predictions == y_test.values).mean()
                                st.write(f"**Prediction Accuracy:** {accuracy:.4f}")

                                cm = confusion_matrix(y_test, predictions)
                                fig = plot_confusion_matrix(cm, model_name)
                                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please ensure your CSV has the correct format with 56 features.")

    # Tab 3: Model Comparison
    with tab3:
        st.header("Compare All Models")

        results = load_evaluation_results()

        if results:
            comparison_data = []
            for model, model_results in results.items():
                metrics = model_results['metrics']
                comparison_data.append({
                    'Model': model,
                    'Accuracy': metrics['accuracy'],
                    'AUC': metrics['auc'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1 Score': metrics['f1'],
                    'MCC': metrics['mcc']
                })

            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

            st.subheader("Model Performance Comparison")
            st.dataframe(comparison_df, use_container_width=True)

            # Simple bar chart
            st.subheader("Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(comparison_df['Model'], comparison_df['Accuracy'])
            ax.set_xlabel('Accuracy')
            ax.set_xlim([0, 1])
            plt.tight_layout()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
