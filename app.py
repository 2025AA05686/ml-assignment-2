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
st.set_page_config(
    page_title="Spam Classification App",
    page_icon="📧",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

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
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Spam', 'Spam'],
                yticklabels=['Not Spam', 'Spam'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(f'Confusion Matrix - {model_name}')
    return fig

def display_metrics(metrics, model_name):
    """Display evaluation metrics in a nice format"""
    st.markdown(f"### Evaluation Metrics for {model_name}")

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

def main():
    """Main application function"""

    # Header
    st.markdown('<p class="main-header">📧 Spam Email Classification System</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning Assignment 2 - Interactive Model Evaluation</p>',
                unsafe_allow_html=True)

    st.markdown("---")

    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    st.sidebar.markdown("Choose a classification model to evaluate")

    model_name = st.sidebar.selectbox(
        "Select Model:",
        (
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        )
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About Dataset")
    st.sidebar.info(
        "**Spambase Dataset**\n\n"
        "- Features: 56\n"
        "- Instances: 4,601\n"
        "- Classes: Binary (Spam/Not Spam)\n"
        "- Source: UCI Machine Learning Repository"
    )

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "🔮 Make Predictions", "📈 Model Comparison", "ℹ️ Dataset Info"])

    # Tab 1: Model Performance
    with tab1:
        st.header(f"Performance Metrics - {model_name}")

        # Load evaluation results
        results = load_evaluation_results()

        if results and model_name in results:
            model_results = results[model_name]

            # Display metrics
            display_metrics(model_results['metrics'], model_name)

            st.markdown("---")

            # Display confusion matrix and classification report side by side
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Confusion Matrix")
                cm = np.array(model_results['confusion_matrix'])
                fig = plot_confusion_matrix(cm, model_name)
                st.pyplot(fig)

            with col2:
                st.subheader("Classification Report")
                clf_report = model_results['classification_report']

                # Create a formatted dataframe from classification report
                report_df = pd.DataFrame({
                    'Class': ['Not Spam', 'Spam', 'Macro Avg', 'Weighted Avg'],
                    'Precision': [
                        clf_report['0']['precision'],
                        clf_report['1']['precision'],
                        clf_report['macro avg']['precision'],
                        clf_report['weighted avg']['precision']
                    ],
                    'Recall': [
                        clf_report['0']['recall'],
                        clf_report['1']['recall'],
                        clf_report['macro avg']['recall'],
                        clf_report['weighted avg']['recall']
                    ],
                    'F1-Score': [
                        clf_report['0']['f1-score'],
                        clf_report['1']['f1-score'],
                        clf_report['macro avg']['f1-score'],
                        clf_report['weighted avg']['f1-score']
                    ],
                    'Support': [
                        clf_report['0']['support'],
                        clf_report['1']['support'],
                        clf_report['macro avg']['support'],
                        clf_report['weighted avg']['support']
                    ]
                })

                # Style the dataframe
                st.dataframe(
                    report_df.style.format({
                        'Precision': '{:.4f}',
                        'Recall': '{:.4f}',
                        'F1-Score': '{:.4f}',
                        'Support': '{:.0f}'
                    }),
                    hide_index=True,
                    use_container_width=True
                )

    # Tab 2: Make Predictions
    with tab2:
        st.header("Make Predictions on Test Data")

        st.markdown("""
        Upload a CSV file with test data to make predictions.

        **Note:** Due to Streamlit Community Cloud limitations, upload only a small test dataset.
        The CSV should have the same 56 features as the training data (excluding the 'label' column).
        """)

        uploaded_file = st.file_uploader(
            "Upload Test Dataset (CSV)",
            type=["csv"],
            help="Upload a CSV file with features only (no label column)"
        )

        if uploaded_file is not None:
            try:
                # Load the uploaded data
                test_data = pd.read_csv(uploaded_file)

                st.success(f"File uploaded successfully! Shape: {test_data.shape}")

                # Display preview
                with st.expander("Preview Data (first 5 rows)"):
                    st.dataframe(test_data.head())

                # Load model and scaler
                model, scaler = load_model(model_name)

                if model is not None and scaler is not None:
                    # Check if data has label column
                    has_label = 'label' in test_data.columns

                    if has_label:
                        X_test = test_data.drop('label', axis=1)
                        y_test = test_data['label']
                    else:
                        X_test = test_data
                        y_test = None

                    # Make predictions
                    if st.button("Predict", type="primary"):
                        with st.spinner("Making predictions..."):
                            # Scale features
                            X_test_scaled = scaler.transform(X_test)

                            # Predict
                            predictions = model.predict(X_test_scaled)

                            # Get probabilities if available
                            if hasattr(model, 'predict_proba'):
                                probabilities = model.predict_proba(X_test_scaled)
                            else:
                                probabilities = None

                            # Display results
                            st.success("Predictions completed!")

                            # Create results dataframe
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

                            # If labels are available, show accuracy
                            if has_label:
                                accuracy = (predictions == y_test.values).mean()
                                st.metric("Prediction Accuracy", f"{accuracy:.4f}")

                                # Show confusion matrix
                                cm = confusion_matrix(y_test, predictions)
                                fig = plot_confusion_matrix(cm, model_name)
                                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your CSV has the correct format with 56 features.")

    # Tab 3: Model Comparison
    with tab3:
        st.header("Compare All Models")

        results = load_evaluation_results()

        if results:
            # Create comparison dataframe
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

            # Display comparison table
            st.subheader("Model Performance Comparison")
            st.dataframe(
                comparison_df.style.format({
                    'Accuracy': '{:.4f}',
                    'AUC': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1 Score': '{:.4f}',
                    'MCC': '{:.4f}'
                }).background_gradient(cmap='RdYlGn', subset=['Accuracy', 'AUC', 'F1 Score', 'MCC']),
                hide_index=True,
                use_container_width=True
            )

            # Visualization
            st.subheader("Visual Comparison")

            metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()

            for idx, metric in enumerate(metrics_to_plot):
                ax = axes[idx]
                data = comparison_df.sort_values(metric, ascending=True)
                ax.barh(data['Model'], data[metric], color='steelblue')
                ax.set_xlabel(metric)
                ax.set_title(f'{metric} Comparison')
                ax.set_xlim([0, 1])

                # Add value labels
                for i, v in enumerate(data[metric]):
                    ax.text(v + 0.01, i, f'{v:.3f}', va='center')

            plt.tight_layout()
            st.pyplot(fig)

    # Tab 4: Dataset Information
    with tab4:
        st.header("Dataset Information")

        st.markdown("""
        ### Spambase Dataset

        **Source:** UCI Machine Learning Repository

        **Description:**
        The Spambase dataset is a collection of email messages classified as spam or not spam (ham).
        The dataset contains 4,601 email instances with 57 attributes (56 features + 1 label).

        **Features (56 total):**
        - **Word Frequencies (48 features):** Percentage of words in the email that match a specific word
        - **Character Frequencies (6 features):** Percentage of characters in the email that match specific characters
        - **Capital Run Lengths (3 features):** Statistics about sequences of consecutive capital letters
          - Average length of uninterrupted sequences of capital letters
          - Length of longest uninterrupted sequence of capital letters
          - Total number of capital letters in the email

        **Target Variable:**
        - **label:** Binary classification (0 = Not Spam, 1 = Spam)

        **Dataset Statistics:**
        - Total Instances: 4,601
        - Not Spam: 2,788 (60.6%)
        - Spam: 1,813 (39.4%)
        - Features: 56
        - Missing Values: 0

        **Train/Test Split:**
        - Training Set: 80% (3,680 instances)
        - Test Set: 20% (921 instances)
        - Stratified split to maintain class distribution

        **Preprocessing:**
        - Feature scaling using StandardScaler
        - No missing value imputation required
        """)

        # Display sample feature names
        with st.expander("Sample Feature Names"):
            sample_features = [
                "word_freq_make", "word_freq_address", "word_freq_all",
                "word_freq_3d", "word_freq_our", "word_freq_over",
                "word_freq_remove", "word_freq_internet", "word_freq_order",
                "char_freq_!", "char_freq_$", "char_freq_#",
                "capital_run_length_average", "capital_run_length_longest",
                "capital_run_length_total"
            ]
            st.code("\n".join(sample_features))

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "Machine Learning Assignment 2 | M.Tech (AIML/DSE) | BITS Pilani"
        "</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
