"""
Machine Learning Assignment 2 - Model Training Script
Trains 6 classification models on the spambase dataset and evaluates them.
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath='data/spambase.csv'):
    """Load and prepare the dataset"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)

    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']

    print(f"Dataset shape: {df.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Class distribution:\n{y.value_counts()}")

    return X, y

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split data into train/test sets and scale features"""
    print("\nSplitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all required evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1': f1_score(y_true, y_pred, average='binary'),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

    # AUC score requires probability predictions
    if y_pred_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    else:
        metrics['auc'] = 0.0

    return metrics

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all models and evaluate them"""

    # Define all models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }

    results = {}
    trained_models = {}

    print("\n" + "="*80)
    print("Training and Evaluating Models")
    print("="*80)

    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print("-" * 40)

        # Train model
        print("  Training...")
        model.fit(X_train, y_train)

        # Make predictions
        print("  Predicting...")
        y_pred = model.predict(X_test)

        # Get probability predictions for AUC
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = None

        # Calculate metrics
        print("  Calculating metrics...")
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Get classification report
        clf_report = classification_report(y_test, y_pred, output_dict=True)

        # Store results
        results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': clf_report
        }

        trained_models[model_name] = model

        # Print metrics
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  MCC:       {metrics['mcc']:.4f}")

    return results, trained_models

def save_models_and_results(trained_models, results, scaler):
    """Save trained models and evaluation results"""
    print("\n" + "="*80)
    print("Saving Models and Results")
    print("="*80)

    # Save each model
    for model_name, model in trained_models.items():
        filename = model_name.lower().replace(' ', '_').replace('-', '_') + '.pkl'
        filepath = f'model/{filename}'
        joblib.dump(model, filepath)
        print(f"Saved: {filepath}")

    # Save scaler
    joblib.dump(scaler, 'model/scaler.pkl')
    print("Saved: model/scaler.pkl")

    # Save results as JSON
    # Convert numpy types to Python types for JSON serialization
    results_serializable = {}
    for model_name, model_results in results.items():
        results_serializable[model_name] = {
            'metrics': {k: float(v) for k, v in model_results['metrics'].items()},
            'confusion_matrix': model_results['confusion_matrix'],
            'classification_report': model_results['classification_report']
        }

    with open('model/evaluation_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print("Saved: model/evaluation_results.json")

    return results_serializable

def create_results_dataframe(results):
    """Create a DataFrame with all model results for easy viewing"""
    data = []
    for model_name, model_results in results.items():
        metrics = model_results['metrics']
        data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'AUC': metrics['auc'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'MCC': metrics['mcc']
        })

    df = pd.DataFrame(data)
    df = df.sort_values('Accuracy', ascending=False)

    return df

def main():
    """Main execution function"""
    print("="*80)
    print("ML Assignment 2 - Model Training and Evaluation")
    print("="*80)

    # Load data
    X, y = load_and_prepare_data('data/spambase.csv')

    # Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)

    # Train and evaluate
    results, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Save everything
    results_serializable = save_models_and_results(trained_models, results, scaler)

    # Create and display results table
    print("\n" + "="*80)
    print("Model Comparison Summary")
    print("="*80)
    results_df = create_results_dataframe(results_serializable)
    print(results_df.to_string(index=False))

    # Save results table to CSV
    results_df.to_csv('model/model_comparison.csv', index=False)
    print("\nSaved: model/model_comparison.csv")

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
