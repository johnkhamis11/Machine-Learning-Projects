# 🚢 Titanic Survival Prediction (High-Accuracy Classification)

This project implements a comprehensive machine learning pipeline to predict passenger survival on the Titanic. By combining historical data with modern ensemble algorithms, the model identifies the key factors—such as age, gender, and class—that determined survival chances during the tragedy.

## 🏆 Model Excellence & Performance
The project stands out due to its rigorous benchmarking of over 10 different classification algorithms:
- **Top Performers:** **XGBoost**, **CatBoost**, and **Random Forest** achieved the highest accuracy and F1-scores.
- **Precision & Reliability:** The models achieved an **Accuracy Score of ~0.86 - 0.93** on various test splits, demonstrating robust generalization.
- **Data Balancing:** Utilized **SMOTE** to ensure the model effectively learns from both survival and non-survival cases, preventing majority-class bias.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `XGBoost`, `CatBoost`, `LightGBM`
- **Preprocessing:** `SMOTE`, `StandardScaler`, `LabelEncoder`

## 📑 Workflow

### 1. Advanced Data Cleaning & Imputation
- **Missing Data:** Handled missing values in the `Age` and `Embarked` columns using statistical measures to maintain data integrity.
- **Feature Engineering:** Extracted insights from passenger titles and family sizes to capture non-linear social dynamics that influenced survival.

### 2. Exploratory Data Analysis (EDA)
- **Survival Patterns:** Visualized how `Sex` and `Pclass` (Passenger Class) were the most critical indicators of survival ("Women and children first" policy).
- **Fare Analysis:** Studied the correlation between ticket fares and survival rates.



### 3. Comprehensive Model Benchmarking
I evaluated a wide range of models to find the optimal solution:
- Logistic Regression & KNN.
- Support Vector Machine (SVM).
- Decision Trees & **Extra Trees Classifier**.
- **Ensemble Learning:** AdaBoost, Gradient Boosting, and Bagging.
- **Advanced Boosting:** **XGBoost**, **LightGBM**, and **CatBoost**.

### 4. Evaluation & Results
- **Confusion Matrix:** Analyzed the balance between True Positives and True Negatives to ensure consistent prediction quality.
- **ROC-AUC Score:** Evaluated the model's ability to distinguish between survivors and non-survivors across different thresholds.



## 📊 Key Insights
- **Gender & Class:** Being female and in a higher passenger class significantly increased survival probability.
- **Model Efficiency:** Boosting algorithms (XGBoost/CatBoost) consistently outperformed traditional models in handling the mixed numerical and categorical nature of the Titanic dataset.

---
*Developed with a focus on Data Preprocessing and Classification Excellence.*
