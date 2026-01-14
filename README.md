# Spam Email Classification System

## Machine Learning Assignment 2
**M.Tech (AIML/DSE) - BITS Pilani**

---

## Problem Statement

Develop a comprehensive machine learning solution for email spam classification using multiple classification algorithms. The objective is to:

1. Build and train 6 different classification models on a spam email dataset
2. Evaluate each model using multiple performance metrics (Accuracy, AUC, Precision, Recall, F1 Score, MCC)
3. Create an interactive web application using Streamlit for model demonstration
4. Deploy the application on Streamlit Community Cloud for public access
5. Compare model performances and provide insights on their effectiveness

The solution demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation, deployment, and interactive visualization.

---

## Dataset Description

### Spambase Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/spambase)

**Overview:**
The Spambase dataset is a classic binary classification dataset containing email messages labeled as spam or legitimate (ham). This dataset was created from a collection of spam and non-spam emails.

**Dataset Characteristics:**
- **Total Instances:** 4,601 emails
- **Total Features:** 56 (continuous/discrete numerical features)
- **Target Variable:** Binary (0 = Not Spam, 1 = Spam)
- **Class Distribution:**
  - Not Spam (Class 0): 2,788 instances (60.6%)
  - Spam (Class 1): 1,813 instances (39.4%)
- **Missing Values:** None

**Feature Categories:**

1. **Word Frequency Features (48 features):**
   - Percentage of words in the email that match specific keywords
   - Examples: `word_freq_make`, `word_freq_address`, `word_freq_free`, `word_freq_business`
   - Range: 0 to 100 (percentage)

2. **Character Frequency Features (6 features):**
   - Percentage of characters in the email that match specific special characters
   - Features: `char_freq_;`, `char_freq_(`, `char_freq_[`, `char_freq_!`, `char_freq_$`, `char_freq_#`
   - Range: 0 to 100 (percentage)

3. **Capital Letter Statistics (3 features):**
   - `capital_run_length_average`: Average length of uninterrupted sequences of capital letters
   - `capital_run_length_longest`: Length of longest uninterrupted sequence of capital letters
   - `capital_run_length_total`: Total number of capital letters in the email

**Data Preprocessing:**
- Train-Test Split: 80-20 (Stratified to maintain class distribution)
- Feature Scaling: StandardScaler applied to all features
- Training Set: 3,680 instances
- Test Set: 921 instances

---

## Models Used

Six different classification algorithms were implemented and evaluated on the Spambase dataset:

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-------|------|
| **Logistic Regression** | 0.9283 | 0.9706 | 0.9207 | 0.8953 | 0.9078 | 0.8494 |
| **Decision Tree** | 0.9088 | 0.9064 | 0.8760 | 0.8953 | 0.8856 | 0.8099 |
| **K-Nearest Neighbors** | 0.9077 | 0.9538 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| **Naive Bayes** | 0.8328 | 0.9374 | 0.7146 | 0.9587 | 0.8188 | 0.6946 |
| **Random Forest** | 0.9435 | 0.9844 | 0.9430 | 0.9118 | 0.9272 | 0.8814 |
| **XGBoost** | 0.9457 | 0.9860 | 0.9311 | 0.9311 | 0.9311 | 0.8863 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Demonstrates strong performance across all metrics with 92.83% accuracy. The model achieves excellent AUC (0.9706), indicating superior ability to distinguish between spam and legitimate emails. High precision (0.9207) minimizes false positives, making it reliable for real-world deployment. The balanced F1 score (0.9078) and high MCC (0.8494) confirm robust overall performance. Logistic Regression serves as a solid baseline model with good interpretability. |
| **Decision Tree** | Achieves good accuracy (90.88%) but shows slightly lower performance compared to ensemble methods. The model has the lowest AUC score (0.9064) among all models, suggesting limited probabilistic discrimination capability. While it maintains reasonable precision (0.8760) and recall (0.8953), the simpler tree structure may lead to overfitting on training data. Decision Trees excel in interpretability but sacrifice some predictive power. Best suited for understanding feature importance and decision boundaries. |
| **K-Nearest Neighbors** | Delivers competitive performance with 90.77% accuracy. The model shows strong AUC (0.9538), indicating good ranking ability for spam detection. Balanced precision (0.8861) and recall (0.8788) suggest consistent performance across both classes. However, KNN is computationally expensive during prediction as it requires distance calculations for all training samples. The model's performance is highly dependent on the choice of k (k=5 used here) and distance metrics. Memory-intensive for large datasets. |
| **Naive Bayes** | Shows the lowest overall accuracy (83.28%) among all models but exhibits the highest recall (0.9587), making it excellent at catching spam emails with minimal false negatives. The trade-off is lower precision (0.7146), resulting in more false positives. This model is particularly useful when missing spam is more costly than incorrectly flagging legitimate emails. Despite lower accuracy, it maintains a strong AUC (0.9374). Naive Bayes is extremely fast, requires minimal training data, and works well with high-dimensional data. Ideal for real-time applications where catching all spam is critical. |
| **Random Forest** | Demonstrates excellent performance with 94.35% accuracy, ranking second overall. The ensemble approach significantly improves upon the single Decision Tree. Outstanding AUC (0.9844) and precision (0.9430) indicate superior ability to correctly identify spam while minimizing false alarms. The model achieves strong recall (0.9118) and F1 score (0.9272), showing balanced performance. High MCC (0.8814) confirms robust correlation between predictions and actual values. Random Forest provides feature importance rankings and is resilient to overfitting. Excellent choice for production systems requiring high accuracy and reliability. |
| **XGBoost** | **Best performing model overall** with the highest accuracy (94.57%), AUC (0.9860), and MCC (0.8863). Achieves perfect balance with equal precision and recall (0.9311), resulting in the highest F1 score (0.9311). The gradient boosting algorithm iteratively corrects errors from previous models, leading to superior predictive performance. Excellent generalization capability with minimal overfitting. XGBoost's regularization techniques and efficient implementation make it ideal for production deployment. Recommended as the primary model for spam detection applications. Provides feature importance and handles missing values effectively. |

---

## Key Insights

### Overall Model Ranking (by Accuracy):
1. **XGBoost** (94.57%) - Best overall performance
2. **Random Forest** (94.35%) - Close second with strong ensemble performance
3. **Logistic Regression** (92.83%) - Excellent baseline with good interpretability
4. **Decision Tree** (90.88%) - Good performance, highly interpretable
5. **K-Nearest Neighbors** (90.77%) - Competitive but computationally expensive
6. **Naive Bayes** (83.28%) - Best for high recall scenarios

### Key Findings:

- **Ensemble methods** (XGBoost and Random Forest) significantly outperform individual classifiers
- **XGBoost** achieves the best balance across all metrics, making it the recommended model for deployment
- **Naive Bayes**, despite lower accuracy, has the highest recall (95.87%), making it valuable for scenarios where missing spam is costly
- All models achieve AUC > 0.90, indicating strong discriminative ability
- **Logistic Regression** provides excellent performance with the added benefit of model interpretability
- The dataset's high-dimensional nature (56 features) favors algorithms that can handle feature interactions effectively

---

## Repository Structure

```
ml-assignment-2/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── data/
│   └── spambase.csv               # Dataset
└── model/
    ├── train_models.py            # Model training script
    ├── logistic_regression.pkl    # Trained Logistic Regression model
    ├── decision_tree.pkl          # Trained Decision Tree model
    ├── k_nearest_neighbors.pkl    # Trained KNN model
    ├── naive_bayes.pkl            # Trained Naive Bayes model
    ├── random_forest.pkl          # Trained Random Forest model
    ├── xgboost.pkl                # Trained XGBoost model
    ├── scaler.pkl                 # Feature scaler
    ├── evaluation_results.json    # Model evaluation metrics
    └── model_comparison.csv       # Model comparison table
```

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ml-assignment-2
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train models (optional - pre-trained models included):**
   ```bash
   python model/train_models.py
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser:**
   Navigate to `http://localhost:8501`

---

## Streamlit Application Features

The interactive web application includes the following features:

### 1. **Model Performance Dashboard**
- Select from 6 different classification models
- View comprehensive evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Interactive confusion matrix visualization
- Detailed classification report

### 2. **Make Predictions**
- Upload custom test datasets (CSV format)
- Get real-time predictions from selected model
- View prediction confidence scores
- Compare predictions with actual labels (if available)

### 3. **Model Comparison**
- Side-by-side comparison of all 6 models
- Color-coded performance heatmap
- Visual bar charts for each metric
- Sortable comparison table

### 4. **Dataset Information**
- Comprehensive dataset documentation
- Feature descriptions and statistics
- Data preprocessing details

---

## Technologies Used

- **Python 3.8+**
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning algorithms and metrics
- **XGBoost** - Gradient boosting framework
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **joblib** - Model serialization

---

## Model Training Details

### Algorithms Implemented:

1. **Logistic Regression**
   - Regularization: Default L2
   - Max iterations: 1000
   - Solver: lbfgs

2. **Decision Tree Classifier**
   - Criterion: Gini impurity
   - Splitter: Best
   - Random state: 42

3. **K-Nearest Neighbors**
   - Number of neighbors: 5
   - Distance metric: Euclidean
   - Weights: Uniform

4. **Naive Bayes**
   - Type: Gaussian Naive Bayes
   - Prior probabilities: Learned from data

5. **Random Forest**
   - Number of estimators: 100
   - Criterion: Gini
   - Random state: 42

6. **XGBoost**
   - Number of estimators: 100
   - Learning rate: Default (0.3)
   - Random state: 42
   - Eval metric: logloss

### Evaluation Metrics:

- **Accuracy:** Overall correctness of predictions
- **AUC (Area Under ROC Curve):** Model's ability to distinguish between classes
- **Precision:** Ratio of true positives to predicted positives (minimizes false alarms)
- **Recall:** Ratio of true positives to actual positives (minimizes missed spam)
- **F1 Score:** Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient):** Balanced measure considering all confusion matrix elements

---

## Deployment

The application is deployed on **Streamlit Community Cloud**, providing free hosting for Streamlit applications.

### Deployment Steps:
1. Push code to GitHub repository
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub account
4. Click "New App" and select repository
5. Choose branch (main) and app file (app.py)
6. Click "Deploy"

The app automatically deploys and updates with each GitHub push.

---

## Usage Instructions

### Running Locally:
```bash
# Start the application
streamlit run app.py

# The app will open in your default browser at http://localhost:8501
```

### Using the Application:

1. **Select a Model:** Use the sidebar to choose from 6 classification models
2. **View Performance:** Navigate to "Model Performance" tab to see detailed metrics
3. **Make Predictions:** Upload a CSV file in the "Make Predictions" tab
4. **Compare Models:** Check the "Model Comparison" tab for side-by-side analysis
5. **Learn About Data:** Visit "Dataset Info" tab for dataset documentation

---

## Test Data Format

For making predictions, upload a CSV file with the following format:

- **56 feature columns** (same as training data)
- Optional `label` column for evaluation
- Feature names must match the original dataset

Example test data structure:
```
word_freq_make,word_freq_address,...,capital_run_length_total,label
0.21,0.28,...,1028,1
0.06,0.00,...,2259,0
```

---

## Performance Summary

The project successfully demonstrates:

- ✅ Implementation of 6 diverse classification algorithms
- ✅ Comprehensive evaluation using 6 different metrics
- ✅ Interactive web application with all required features
- ✅ Model comparison and visualization capabilities
- ✅ Cloud deployment on Streamlit Community Cloud
- ✅ Clean, well-documented codebase
- ✅ Production-ready machine learning pipeline

**Best Model:** XGBoost with 94.57% accuracy and 0.9860 AUC score

---

## Future Enhancements

- Implement hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Add cross-validation for more robust performance estimates
- Include additional models (SVM, Neural Networks)
- Add feature importance visualization
- Implement A/B testing framework for model comparison
- Create API endpoints for programmatic access
- Add email text preprocessing for raw email input
- Implement model retraining pipeline with new data

---

## Author

**BITS Pilani Student**
M.Tech (AIML/DSE)
Machine Learning - Assignment 2

---

## License

This project is created for academic purposes as part of the Machine Learning course at BITS Pilani.

---

## Acknowledgments

- BITS Pilani for course instruction and resources
- UCI Machine Learning Repository for the Spambase dataset
- Streamlit team for the excellent web framework
- scikit-learn and XGBoost communities for robust ML libraries

---

## Contact

For questions or feedback regarding this project, please refer to the course instructor or teaching assistants.

---

**Last Updated:** January 2026
