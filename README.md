# Spam Email Classification System

## Machine Learning Assignment 2
**Name:** Pritish Joshi
**ID:** 2025AA05686
**M.Tech (AIML) - BITS Pilani**

**Live App:** https://2025aa05686-spam-detector.streamlit.app/

---

## Problem Statement

Develop a machine learning solution for email spam classification using multiple classification algorithms. The objectives are:

1. Build and train 6 different classification models on a spam email dataset
2. Evaluate each model using multiple performance metrics (Accuracy, AUC, Precision, Recall, F1 Score, MCC)
3. Create an interactive Streamlit web application to demonstrate the models
4. Deploy the application on Streamlit Community Cloud
5. Compare model performances and provide insights

---

## Dataset Description

### Spambase Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/spambase)

The Spambase dataset is a binary classification dataset containing email messages labeled as spam or legitimate (ham). It consists of emails collected for spam detection research.

**Dataset Characteristics:**
- **Total Instances:** 4,601 emails
- **Total Features:** 56 numerical features
- **Target Variable:** Binary (0 = Not Spam, 1 = Spam)
- **Class Distribution:**
  - Not Spam (Class 0): 2,788 instances (60.6%)
  - Spam (Class 1): 1,813 instances (39.4%)
- **Missing Values:** None

**Feature Categories:**

1. **Word Frequency Features (48 features):**
   - Percentage of words in the email matching specific keywords
   - Examples: word_freq_make, word_freq_address, word_freq_free, word_freq_business
   - Range: 0 to 100 (percentage)

2. **Character Frequency Features (6 features):**
   - Percentage of specific characters in the email
   - Characters: ; ( [ ! $ #
   - Range: 0 to 100 (percentage)

3. **Capital Letter Statistics (3 features):**
   - capital_run_length_average: Average length of consecutive capital letters
   - capital_run_length_longest: Longest sequence of capital letters
   - capital_run_length_total: Total number of capital letters

**Data Preprocessing:**
- Train-Test Split: 80-20 (Stratified)
- Feature Scaling: StandardScaler applied
- Training Set: 3,680 instances
- Test Set: 921 instances

---

## Models Used

Six classification algorithms were implemented and evaluated:

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-------|------|
| Logistic Regression | 0.9283 | 0.9706 | 0.9207 | 0.8953 | 0.9078 | 0.8494 |
| Decision Tree | 0.9088 | 0.9064 | 0.8760 | 0.8953 | 0.8856 | 0.8099 |
| K-Nearest Neighbors | 0.9077 | 0.9538 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| Naive Bayes | 0.8328 | 0.9374 | 0.7146 | 0.9587 | 0.8188 | 0.6946 |
| Random Forest (Ensemble) | 0.9435 | 0.9844 | 0.9430 | 0.9118 | 0.9272 | 0.8814 |
| XGBoost (Ensemble) | 0.9457 | 0.9860 | 0.9311 | 0.9311 | 0.9311 | 0.8863 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Demonstrates strong performance with 92.83% accuracy. Achieves excellent AUC (0.9706), indicating good ability to distinguish between spam and legitimate emails. High precision (0.9207) minimizes false positives. The balanced F1 score (0.9078) and high MCC (0.8494) confirm robust overall performance. Serves as a solid baseline model with good interpretability. |
| Decision Tree | Achieves good accuracy (90.88%) but shows slightly lower performance compared to ensemble methods. Has the lowest AUC score (0.9064) among all models. Maintains reasonable precision (0.8760) and recall (0.8953). The simpler tree structure may lead to some overfitting on training data. Excels in interpretability which helps understand decision boundaries. |
| K-Nearest Neighbors | Delivers competitive performance with 90.77% accuracy. Shows strong AUC (0.9538) indicating good ranking ability. Balanced precision (0.8861) and recall (0.8788) suggest consistent performance across both classes. Computationally expensive during prediction as it requires distance calculations. Performance is dependent on the choice of k (k=5 used here). |
| Naive Bayes | Shows the lowest overall accuracy (83.28%) but exhibits the highest recall (0.9587), making it excellent at catching spam emails with minimal false negatives. The trade-off is lower precision (0.7146), resulting in more false positives. This model is useful when missing spam is more costly than incorrectly flagging legitimate emails. Extremely fast and works well with high-dimensional data. |
| Random Forest (Ensemble) | Demonstrates excellent performance with 94.35% accuracy, ranking second overall. The ensemble approach significantly improves upon the single Decision Tree. Outstanding AUC (0.9844) and precision (0.9430) indicate superior ability to correctly identify spam while minimizing false alarms. Strong recall (0.9118) and F1 score (0.9272) show balanced performance. High MCC (0.8814) confirms robust correlation. |
| XGBoost (Ensemble) | Best performing model overall with highest accuracy (94.57%), AUC (0.9860), and MCC (0.8863). Achieves perfect balance with equal precision and recall (0.9311), resulting in the highest F1 score (0.9311). The gradient boosting algorithm iteratively corrects errors, leading to superior predictive performance. Excellent generalization with minimal overfitting. Recommended as the primary model for spam detection. |

---

## Repository Structure

```
ml-assignment-2/
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/
│   ├── spambase.csv           # Main dataset (4,601 instances)
│   └── test_data.csv          # Sample test file (10 instances)
└── model/
    ├── train_models.py        # Model training script
    ├── *.pkl                  # Trained models (6 models + scaler)
    ├── evaluation_results.json # All evaluation metrics
    └── model_comparison.csv   # Model comparison table
```

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ml-assignment-2
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train models (optional - pre-trained models included):
   ```bash
   python model/train_models.py
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Open browser at `http://localhost:8501`

---

## Streamlit Application Features

The web application includes all required features:

1. **Model Selection Dropdown** - Choose from 6 classification models
2. **Dataset Upload (CSV)** - Upload test data for predictions
3. **Evaluation Metrics Display** - View all 6 metrics for selected model
4. **Confusion Matrix** - Visual representation of model performance
5. **Classification Report** - Detailed performance breakdown
6. **Model Comparison** - Compare all models side by side

### Testing the Application

A sample test file is provided at `data/test_data.csv` containing 10 email samples (5 spam, 5 not spam) for quick testing.

To test:
1. Run the Streamlit app
2. Navigate to "Make Predictions" tab
3. Upload `data/test_data.csv`
4. Click "Predict" to see results

---

## Model Training Details

### Algorithms Implemented:

1. **Logistic Regression** - Linear classifier with L2 regularization, max_iter=1000
2. **Decision Tree** - Tree-based classifier with Gini criterion
3. **K-Nearest Neighbors** - Instance-based learning with k=5 neighbors
4. **Naive Bayes** - Gaussian Naive Bayes classifier
5. **Random Forest** - Ensemble of 100 decision trees
6. **XGBoost** - Gradient boosting with 100 estimators

### Evaluation Metrics:

- **Accuracy:** Overall correctness of predictions
- **AUC:** Area Under ROC Curve - model's discrimination ability
- **Precision:** Ratio of true positives to predicted positives
- **Recall:** Ratio of true positives to actual positives
- **F1 Score:** Harmonic mean of precision and recall
- **MCC:** Matthews Correlation Coefficient - balanced measure

---

## Deployment

The application is deployed on Streamlit Community Cloud.

**Live Application:** https://2025aa05686-spam-detector.streamlit.app/

**Deployment Steps:**
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Sign in with GitHub account
4. Click "New App" and select repository
5. Choose branch (main) and app file (app.py)
6. Click "Deploy"

---

## Results Summary

**Best Model:** XGBoost with 94.57% accuracy and 0.9860 AUC score

**Key Findings:**
- Ensemble methods (XGBoost and Random Forest) outperform individual classifiers
- All models achieve AUC > 0.90, indicating strong discriminative ability
- Naive Bayes has highest recall (95.87%) - best for catching all spam
- XGBoost achieves best balance across all metrics

---

## Author

**Pritish Joshi**
**ID:** 2025AA05686
M.Tech (AIML)
BITS Pilani
Machine Learning - Assignment 2

---

## Acknowledgments

- BITS Pilani for course instruction
- UCI Machine Learning Repository for the Spambase dataset

---

**Last Updated:** February 2026
