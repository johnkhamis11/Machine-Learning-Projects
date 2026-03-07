# 🩺 Diabetes Prediction System using Machine Learning

This project implements a high-precision classification system to predict the likelihood of diabetes in patients based on medical diagnostic measurements. The pipeline covers everything from deep data exploration to model persistence, ensuring a production-ready solution for healthcare analytics.

## 📌 Project Overview
The objective is to diagnostically predict whether or not a patient has diabetes based on certain diagnostic measurements included in the dataset (e.g., Glucose levels, BMI, Blood Pressure). The project addresses common real-world data issues such as class imbalance and feature scaling.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Analysis:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `XGBoost`, `CatBoost`, `LightGBM`
- **Preprocessing:** `SMOTE` (for class imbalance), `StandardScaler`, `LabelEncoder`
- **Model Deployment:** `Joblib`

## 📑 Workflow

### 1. Exploratory Data Analysis (EDA)
- **Statistical Profiling:** Analyzed the distribution of variables like Glucose and BMI.
- **Correlation Heatmap:** Identified key medical indicators that have the strongest correlation with diabetes.
- **Outlier Detection:** Managed extreme values to ensure they don't skew the model's learning process.

### 2. Data Preprocessing & Balancing
- **Feature Scaling:** Applied `StandardScaler` to ensure all medical metrics are on the same scale.
- **Handling Class Imbalance:** Used **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class, significantly improving the model's ability to detect diabetic cases (Recall).



### 3. Model Training & Hyperparameter Tuning
Benchmarks were performed across multiple algorithms:
- Logistic Regression & SVC
- **Random Forest Classifier** (Final Optimized Model)
- **Gradient Boosting Models:** XGBoost, LightGBM, CatBoost.
- **Fine-Tuning:** Used `GridSearchCV` to find the optimal parameters for the Random Forest model.

### 4. Evaluation Metrics
The model was evaluated using:
- **Accuracy Score:** High overall correctness.
- **Confusion Matrix:** To minimize False Negatives (critical in medical diagnosis).
- **ROC-AUC Curve:** To measure the separation power of the classifier.



## 💾 Model Persistence
The final trained components are exported for easy integration into web apps (like Streamlit):
- `random_forest_model.pkl`: The optimized classification model.
- `scaler.pkl`: The fitted scaler for consistent data normalization.

---
*Developed with a focus on Healthcare Informatics and High-Precision Diagnostics.*
