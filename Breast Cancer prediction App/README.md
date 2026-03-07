# Breast Cancer Prediction using Machine Learning 🎗️

This project focuses on building a highly accurate classification model to predict whether a breast tumor is **Malignant (M)** or **Benign (B)** based on diagnostic features. The project utilizes various machine learning algorithms and advanced preprocessing techniques to ensure reliable predictions.

## 📌 Project Overview
The dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. This model serves as a decision-support tool for medical professionals in early cancer detection.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Analysis:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `XGBoost`, `CatBoost`, `LightGBM`
- **Balancing Technique:** `SMOTE` (Imbalanced-learn)
- **Model Deployment:** `Joblib` (for saving models)

## 📑 Workflow

### 1. Exploratory Data Analysis (EDA)
- Analyzed the distribution of diagnoses (Malignant vs. Benign).
- Heatmap correlation to identify highly correlated features (e.g., radius, perimeter, and area).
- Distribution plots for various features to understand their impact on the target class.

### 2. Data Preprocessing
- **Cleaning:** Dropped unnecessary columns such as `id` and `Unnamed: 32`.
- **Encoding:** Converted target labels ('M' and 'B') into numerical values using `LabelEncoder`.
- **Feature Scaling:** Applied `StandardScaler` to ensure all features contribute equally to the model.
- **Handling Imbalance:** Used **SMOTE** to balance the classes within the training data for better generalization.

### 3. Model Training & Selection
The project involved training and comparing multiple classifiers:
- Logistic Regression & KNN
- Support Vector Classifier (SVC)
- Decision Trees & Random Forest
- **Ensemble Methods:** AdaBoost, Gradient Boosting, Extra Trees.
- **Advanced Boosting:** XGBoost, LightGBM, and CatBoost.

### 4. Hyperparameter Tuning
Used **GridSearchCV** to optimize the performance of the best-performing models (like AdaBoost and CatBoost), focusing on parameters such as `n_estimators` and `learning_rate`.

### 5. Evaluation Metrics
Models were rigorously tested using:
- **Accuracy Score**
- **Confusion Matrix**
- **Precision, Recall, and F1-Score** (Crucial for medical diagnosis)
- **Feature Importance Analysis**

## 💾 Model Persistence
The final trained model (`breast_cancer_adaboost_final.pkl`) and the scaler (`scaler.pkl`) have been exported using `joblib` for easy integration into web applications or production environments.

## ⚙️ Setup & Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost lightgbm imbalanced-learn joblib
