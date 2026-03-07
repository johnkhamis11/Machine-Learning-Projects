# 📉 Customer Churn Prediction using Machine Learning

This project develops a predictive system to identify customers who are likely to leave a service (churn). By analyzing customer demographics, account information, and usage patterns, the model provides actionable insights to help businesses improve retention strategies and reduce revenue loss.

## 📌 Project Overview
The dataset contains information about telecom customers, including their services, account details, and whether they stayed or left. The objective is to build a binary classification model that accurately predicts the `Churn` status using a variety of machine learning algorithms.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `XGBoost`
- **Ensemble Techniques:** `VotingClassifier`, `Bagging`, `Boosting`

## 📑 Workflow

### 1. Data Cleaning & Preprocessing
- **Handling Categorical Data:** Applied `LabelEncoder` and `OneHotEncoder` to transform non-numeric data like `Gender`, `Contract`, and `PaymentMethod`.
- **Feature Scaling:** Used `StandardScaler` to normalize numerical features (e.g., `tenure`, `MonthlyCharges`), ensuring consistent performance across distance-based algorithms like KNN and SVM.
- **Handling Imbalance:** Analyzed the class distribution of the target variable to ensure the model doesn't lean towards the majority class.

### 2. Exploratory Data Analysis (EDA)
- **Churn Drivers:** Visualized how `Contract` type (Month-to-month vs. One year) and `tenure` significantly impact churn rates.
- **Financial Impact:** Analyzed the relationship between `MonthlyCharges`, `TotalCharges`, and customer loyalty.
- **Correlation Analysis:** Generated a heatmap to identify which features have the strongest influence on the churn decision.



### 3. Model Benchmarking & Selection
I implemented and compared a comprehensive list of models to find the best performer:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree & Random Forest
- **Advanced Ensemble Models:** AdaBoost, Gradient Boosting, and **XGBoost**.
- **Voting Classifier:** Combined multiple models to improve overall stability and accuracy.

### 4. Evaluation & Results
The models were evaluated using a wide range of metrics to ensure a balanced view of performance:
- **Accuracy & F1-Score:** To measure overall correctness and the balance between precision and recall.
- **ROC-AUC Score:** Used to evaluate the model's ability to distinguish between churn and non-churn customers.
- **Confusion Matrix:** Detailed analysis of False Positives vs. False Negatives, which is critical for business decision-making.

## 💾 Final Deliverables
- Comprehensive Jupyter Notebook with end-to-end analysis.
- Optimized classification pipeline capable of identifying high-risk customers.
- Detailed visualization of feature importance showing the top factors leading to customer churn.

---
*Developed with a focus on Business Intelligence and Predictive Analytics.*
