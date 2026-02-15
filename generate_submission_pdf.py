"""
Generate PDF Submission for ML Assignment 2
Creates a professional PDF document with all required submission elements
"""

from fpdf import FPDF
import os

class SubmissionPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_page()
        self.set_auto_page_break(auto=True, margin=15)
        
    def header_section(self):
        self.set_font('Helvetica', 'B', 18)
        self.cell(0, 12, 'Machine Learning Assignment 2', ln=True, align='C')
        self.set_font('Helvetica', '', 12)
        self.cell(0, 8, 'M.Tech (AIML/DSE) - BITS Pilani', ln=True, align='C')
        self.cell(0, 8, 'Spam Email Classification System', ln=True, align='C')
        self.ln(5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(8)
        
    def section_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, ln=True, fill=True)
        self.ln(3)
        
    def add_link_section(self, title, url, description=""):
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 8, title, ln=True)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 255)
        self.cell(0, 6, url, ln=True, link=url)
        self.set_text_color(0, 0, 0)
        if description:
            self.set_font('Helvetica', 'I', 9)
            self.multi_cell(0, 5, description)
        self.ln(3)
        
    def add_paragraph(self, text):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(2)
        
    def add_subsection(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, title, ln=True)
        self.ln(2)
        
    def add_bullet(self, text, indent=5):
        self.set_font('Helvetica', '', 10)
        self.set_x(self.get_x() + indent)
        self.multi_cell(0, 5, f"- {text}")
        
    def add_table(self, headers, data, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
            
        # Header row
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(200, 200, 200)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 8, header, border=1, fill=True, align='C')
        self.ln()
        
        # Data rows
        self.set_font('Helvetica', '', 8)
        for row in data:
            max_height = 8
            for i, cell in enumerate(row):
                self.cell(col_widths[i], max_height, str(cell), border=1, align='C')
            self.ln()

def create_submission_pdf():
    pdf = SubmissionPDF()
    
    # Header
    pdf.header_section()
    
    # Section 1: GitHub Repository Link
    pdf.section_title("1. GitHub Repository Link")
    pdf.add_link_section(
        "Repository URL:",
        "https://github.com/2025AA05686/ml-assignment-2",
        "Contains complete source code, requirements.txt, README.md, and trained models"
    )
    
    # Section 2: Live Streamlit App
    pdf.section_title("2. Live Streamlit App Link")
    pdf.add_link_section(
        "Deployed Application:",
        "https://2025aa05686-spam-detector.streamlit.app/",
        "Interactive web application for spam classification using 6 ML models"
    )
    
    # Section 3: Screenshot Section
    pdf.section_title("3. BITS Virtual Lab Screenshot")
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 6, "Assignment executed on BITS Virtual Lab as required:")
    pdf.ln(3)

    # Add screenshot image if it exists
    screenshot_path = "screenshots/bits_lab_screenshot.png"
    if os.path.exists(screenshot_path):
        try:
            # Calculate image dimensions to fit on page
            page_width = 190  # Usable page width
            pdf.image(screenshot_path, x=10, w=page_width)
            pdf.ln(5)
        except Exception as e:
            pdf.set_font('Helvetica', 'I', 10)
            pdf.set_fill_color(255, 200, 200)
            pdf.multi_cell(0, 8, f"[Error loading screenshot: {str(e)}]", fill=True, align='C')
    else:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_fill_color(255, 255, 200)
        pdf.multi_cell(0, 8, "[Screenshot file not found at screenshots/bits_lab_screenshot.png]", fill=True, align='C')
    pdf.ln(5)
    
    # Section 4: README Content
    pdf.section_title("4. README.md Content")
    
    # Problem Statement
    pdf.add_subsection("Problem Statement")
    pdf.add_paragraph(
        "Develop a machine learning solution for email spam classification using multiple classification algorithms. "
        "The objectives are:\n"
        "1. Build and train 6 different classification models on a spam email dataset\n"
        "2. Evaluate each model using multiple performance metrics (Accuracy, AUC, Precision, Recall, F1 Score, MCC)\n"
        "3. Create an interactive Streamlit web application to demonstrate the models\n"
        "4. Deploy the application on Streamlit Community Cloud\n"
        "5. Compare model performances and provide insights"
    )
    
    # Dataset Description
    pdf.add_subsection("Dataset Description")
    pdf.add_paragraph("Spambase Dataset")
    pdf.add_paragraph("Source: UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/spambase)")
    pdf.add_paragraph(
        "The Spambase dataset is a binary classification dataset containing email messages labeled as spam or legitimate (ham). "
        "It consists of emails collected for spam detection research."
    )
    pdf.ln(2)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, "Dataset Characteristics:", ln=True)
    pdf.add_bullet("Total Instances: 4,601 emails")
    pdf.add_bullet("Total Features: 56 numerical features")
    pdf.add_bullet("Target Variable: Binary (0 = Not Spam, 1 = Spam)")
    pdf.add_bullet("Class Distribution: Not Spam (60.6%), Spam (39.4%)")
    pdf.add_bullet("Missing Values: None")
    pdf.ln(2)
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, "Feature Categories:", ln=True)
    pdf.add_bullet("Word Frequency Features (48 features): Percentage of words matching specific keywords")
    pdf.add_bullet("Character Frequency Features (6 features): Percentage of specific characters (; ( [ ! $ #)")
    pdf.add_bullet("Capital Letter Statistics (3 features): Average, longest, and total capital letter counts")
    pdf.ln(2)
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, "Data Preprocessing:", ln=True)
    pdf.add_bullet("Train-Test Split: 80-20 (Stratified)")
    pdf.add_bullet("Feature Scaling: StandardScaler applied")
    pdf.add_bullet("Training Set: 3,680 instances")
    pdf.add_bullet("Test Set: 921 instances")
    pdf.ln(3)
    
    # Model Comparison Table
    pdf.add_page()
    pdf.section_title("Models Used - Comparison Table")
    
    headers = ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    data = [
        ["Logistic Regression", "0.9283", "0.9706", "0.9207", "0.8953", "0.9078", "0.8494"],
        ["Decision Tree", "0.9088", "0.9064", "0.8760", "0.8953", "0.8856", "0.8099"],
        ["K-Nearest Neighbors", "0.9077", "0.9538", "0.8861", "0.8788", "0.8824", "0.8065"],
        ["Naive Bayes", "0.8328", "0.9374", "0.7146", "0.9587", "0.8188", "0.6946"],
        ["Random Forest (Ensemble)", "0.9435", "0.9844", "0.9430", "0.9118", "0.9272", "0.8814"],
        ["XGBoost (Ensemble)", "0.9457", "0.9860", "0.9311", "0.9311", "0.9311", "0.8863"],
    ]
    col_widths = [45, 22, 22, 24, 22, 22, 22]
    pdf.add_table(headers, data, col_widths)
    pdf.ln(8)
    
    # Model Observations
    pdf.section_title("Model Performance Observations")
    
    observations = [
        ("Logistic Regression", 
         "Demonstrates strong performance with 92.83% accuracy. Achieves excellent AUC (0.9706), indicating good ability to distinguish between spam and legitimate emails. High precision (0.9207) minimizes false positives. The balanced F1 score (0.9078) and high MCC (0.8494) confirm robust overall performance. Serves as a solid baseline model with good interpretability."),
        
        ("Decision Tree", 
         "Achieves good accuracy (90.88%) but shows slightly lower performance compared to ensemble methods. Has the lowest AUC score (0.9064) among all models. Maintains reasonable precision (0.8760) and recall (0.8953). The simpler tree structure may lead to some overfitting on training data. Excels in interpretability which helps understand decision boundaries."),
        
        ("K-Nearest Neighbors", 
         "Delivers competitive performance with 90.77% accuracy. Shows strong AUC (0.9538) indicating good ranking ability. Balanced precision (0.8861) and recall (0.8788) suggest consistent performance across both classes. Computationally expensive during prediction as it requires distance calculations. Performance is dependent on the choice of k (k=5 used here)."),
        
        ("Naive Bayes", 
         "Shows the lowest overall accuracy (83.28%) but exhibits the highest recall (0.9587), making it excellent at catching spam emails with minimal false negatives. The trade-off is lower precision (0.7146), resulting in more false positives. This model is useful when missing spam is more costly than incorrectly flagging legitimate emails. Extremely fast and works well with high-dimensional data."),
        
        ("Random Forest (Ensemble)", 
         "Demonstrates excellent performance with 94.35% accuracy, ranking second overall. The ensemble approach significantly improves upon the single Decision Tree. Outstanding AUC (0.9844) and precision (0.9430) indicate superior ability to correctly identify spam while minimizing false alarms. Strong recall (0.9118) and F1 score (0.9272) show balanced performance."),
        
        ("XGBoost (Ensemble)", 
         "Best performing model overall with highest accuracy (94.57%), AUC (0.9860), and MCC (0.8863). Achieves perfect balance with equal precision and recall (0.9311), resulting in the highest F1 score. The gradient boosting algorithm iteratively corrects errors, leading to superior predictive performance. Excellent generalization with minimal overfitting. Recommended as the primary model for spam detection."),
    ]
    
    for model_name, observation in observations:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 7, model_name, ln=True)
        pdf.set_font('Helvetica', '', 9)
        pdf.multi_cell(0, 5, observation)
        pdf.ln(3)
    
    # Repository Structure
    pdf.add_page()
    pdf.section_title("Repository Structure")
    pdf.set_font('Courier', '', 9)
    structure = """ml-assignment-2/
|-- app.py                      # Streamlit web application
|-- requirements.txt            # Python dependencies
|-- README.md                   # Project documentation
|-- data/
|   |-- spambase.csv           # Main dataset (4,601 instances)
|   |-- test_data.csv          # Sample test file (10 instances)
|-- model/
    |-- train_models.py        # Model training script
    |-- *.pkl                  # Trained models (6 models + scaler)
    |-- evaluation_results.json # All evaluation metrics
    |-- model_comparison.csv   # Model comparison table"""
    pdf.multi_cell(0, 5, structure)
    pdf.ln(5)
    
    # Streamlit App Features
    pdf.set_font('Helvetica', '', 10)
    pdf.section_title("Streamlit Application Features")
    pdf.add_bullet("Model Selection Dropdown - Choose from 6 classification models")
    pdf.add_bullet("Dataset Upload (CSV) - Upload test data for predictions")
    pdf.add_bullet("Evaluation Metrics Display - View all 6 metrics for selected model")
    pdf.add_bullet("Confusion Matrix - Visual representation of model performance")
    pdf.add_bullet("Classification Report - Detailed performance breakdown")
    pdf.add_bullet("Model Comparison - Compare all models side by side")
    pdf.ln(5)
    
    # Results Summary
    pdf.section_title("Results Summary")
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 7, "Best Model: XGBoost with 94.57% accuracy and 0.9860 AUC score", ln=True)
    pdf.ln(2)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 7, "Key Findings:", ln=True)
    pdf.add_bullet("Ensemble methods (XGBoost and Random Forest) outperform individual classifiers")
    pdf.add_bullet("All models achieve AUC > 0.90, indicating strong discriminative ability")
    pdf.add_bullet("Naive Bayes has highest recall (95.87%) - best for catching all spam")
    pdf.add_bullet("XGBoost achieves best balance across all metrics")
    
    # Save PDF
    output_path = "ML_Assignment_2_Submission.pdf"
    pdf.output(output_path)
    print(f"PDF generated successfully: {output_path}")
    return output_path

if __name__ == "__main__":
    create_submission_pdf()
