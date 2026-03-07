# Autism Spectrum Disorder (ASD) Prediction using Machine Learning

This project aims to develop a robust machine learning model to predict and classify Autism Spectrum Disorder (ASD) traits based on behavioral features and demographic data. The project involves comprehensive data preprocessing, handling class imbalance, and benchmarking several state-of-the-art classifiers.

## 📌 Project Overview
The dataset contains features related to screening methods (A1-A10 scores) and individual characteristics. The goal is to accurately identify potential ASD cases to assist in early intervention and screening processes.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `XGBoost`, `CatBoost`, `LightGBM`
- **Class Imbalance Handling:** `Imbalanced-learn` (SMOTE)

## 📑 Workflow

### 1. Data Exploration (EDA)
- Analyzed the distribution of target classes.
- Visualized correlations between the A1-A10 scores and the final diagnosis.
- Identified and handled missing values and outliers.

### 2. Data Preprocessing
- **Feature Engineering:** Dropped irrelevant columns (e.g., `age_desc`, `id`).
- **Encoding:** Applied `LabelEncoder` and `OrdinalEncoder` for categorical features like ethnicity and country of residence.
- **Scaling:** Utilized `StandardScaler` to normalize feature distributions.
- **Resampling:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to address class imbalance in the training set.

### 3. Model Training & Optimization
Evaluated multiple classification algorithms:
- Logistic Regression & KNN
- Support Vector Machines (SVC)
- Random Forest & Extra Trees
- Boosting Algorithms: **XGBoost, CatBoost, and LightGBM**
- Hyperparameter tuning using **GridSearchCV** for optimal performance.

### 4. Evaluation Metrics
Models were evaluated using:
- Accuracy Score
- Precision & Recall
- F1-Score
- ROC-AUC Curve
- Confusion Matrix

## 📊 Key Findings
The analysis revealed that:
- Behavioral scores (A1-A10) are the strongest predictors of ASD.
- Tree-based ensemble models (like CatBoost and XGBoost) achieved superior performance compared to traditional linear models.
- Features such as `ethnicity` and `jaundice` history also showed significant correlation with the target variable.

## ⚙️ Setup & Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost lightgbm imbalanced-learn
