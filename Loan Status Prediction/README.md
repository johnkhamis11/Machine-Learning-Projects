# 🏦 Loan Status Prediction (Credit Risk Analysis)

This project develops a machine learning-based classification system to predict whether a loan application should be **Approved (Y)** or **Rejected (N)**. By analyzing applicant profiles, financial history, and credit worthiness, the model helps financial institutions automate and optimize their lending decisions.

## 📌 Project Overview
The core objective is to minimize credit risk by identifying high-risk applicants. The project places a heavy emphasis on **Recall**, ensuring that potential defaulters are identified as accurately as possible to safeguard the institution's capital.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Analysis:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`
- **Class Balancing:** `SMOTE` (Imbalanced-learn)
- **Model Tuning:** `GridSearchCV`

## 📑 Workflow

### 1. Exploratory Data Analysis (EDA)
- **Categorical Insights:** Analyzed how factors like `Education`, `Marital Status`, and `Property Area` influence loan approval rates.
- **Financial Ratios:** Visualized the distribution of `ApplicantIncome` and `LoanAmount` to identify potential outliers and trends.
- **Credit History Impact:** Confirmed that `Credit_History` is the most significant feature in determining loan eligibility.



### 2. Data Preprocessing & Balancing
- **Missing Value Treatment:** Imputed missing values for critical features like `Credit_History`, `Self_Employed`, and `LoanAmount`.
- **Feature Encoding:** Applied `LabelEncoder` for binary categorical variables and handled multi-class features.
- **Scaling:** Used `StandardScaler` to bring income and loan values to a uniform scale.
- **Handling Imbalance:** Implemented **SMOTE** to balance the classes, significantly boosting the model's ability to detect minority class (Rejected) cases.

### 3. Comprehensive Model Benchmarking
An extensive comparison was conducted across various classification algorithms:
- Logistic Regression & KNN
- **Support Vector Machine (SVM)** (Identified as a top performer for Recall)
- Decision Tree & Random Forest
- **Ensemble & Boosting:** AdaBoost, Gradient Boosting, XGBoost, LightGBM, and CatBoost.

### 4. Evaluation & Results
The project prioritized metrics that matter in financial risk:
- **Recall (Primary Metric):** To ensure maximum detection of high-risk loans.
- **Confusion Matrix:** Detailed analysis of misclassifications to understand the model's conservative vs. aggressive nature.
- **ROC-AUC Score:** To evaluate the model's capability to distinguish between classes.



## 📊 Key Findings
- **SVM** emerged as one of the best models for achieving high **Recall**, making it a strong candidate for conservative lending environments.
- Credit history remains the most dominant factor, but the model successfully captures secondary influences from income levels and employment types.

---
*Developed with a focus on Financial Risk Management and Predictive Modeling.*
