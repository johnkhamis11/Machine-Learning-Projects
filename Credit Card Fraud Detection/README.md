# 💳 Credit Card Fraud Detection (High-Security Analytics)

This project implements a robust anomaly detection system designed to identify fraudulent credit card transactions. By leveraging statistical analysis and machine learning, the model distinguishes between legitimate usage and potential security threats with high precision and sensitivity.

## 🏆 Model Excellence & Performance
The highlight of this project is its ability to maintain high reliability in a highly imbalanced environment:
- **Accuracy:** Achieved a balanced accuracy of **~94%** on both fraud and legitimate classes.
- **Precision & Recall:** Optimized for a high **Precision (~0.98)** on fraud detection to minimize false alarms while maintaining a strong **Recall (~0.91)** to catch actual fraud.
- **Model Choice:** Utilized **Logistic Regression** for its transparency and efficiency in binary classification for financial transactions.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Analysis:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`
- **Evaluation:** Confusion Matrix, Precision-Recall Curve, Classification Report.

## 📑 Workflow

### 1. Data Exploration & Sampling
- **Class Imbalance:** Analyzed the massive skewness between legitimate (Class 0) and fraudulent (Class 1) transactions.
- **Under-sampling Technique:** Created a balanced dataset by sampling from the majority class to ensure the model learns the distinct patterns of fraud effectively.



### 2. Feature Analysis
- **Anonymized Data:** Handled V1-V28 features (resulting from PCA transformation) to maintain user privacy while extracting predictive signals.
- **Transaction Timing:** Analyzed how time and transaction amount correlate with the likelihood of fraud.

### 3. Model Training
- **Algorithm:** Logistic Regression was selected for its proven effectiveness in high-dimensional financial datasets.
- **Standardization:** Ensured the "Amount" and "Time" features were scaled to prevent bias in the model's coefficients.

### 4. Evaluation & Results
The model was evaluated using metrics that go beyond simple accuracy:
- **Confusion Matrix:** Successfully identified 86 out of 95 fraud cases in the test sample with only 2 false positives.
- **F1-Score:** Achieved a balanced score of **0.94**, demonstrating the model's robustness.



## 📊 Key Findings
- Financial fraud patterns can be accurately detected even with simplified linear models if the data is balanced and pre-processed correctly.
- The model shows extreme reliability, making it a viable baseline for real-time transaction monitoring systems.

---
*Developed with a focus on Financial Security and Anomaly Detection.*
