# 🏥 Heart Disease Prediction System (High-Precision & Recall Classifier)

This project features a robust Machine Learning pipeline designed to predict the presence of heart disease with exceptional accuracy. By leveraging clinical data and advanced ensemble techniques, the model serves as a highly reliable diagnostic support tool.

## 🏆 Model Excellence & Performance
The highlight of this project is the meticulous approach to model selection and data balancing, leading to superior results:
- **Accuracy Highlights:** Achieved outstanding performance by benchmarking over 10 different classifiers.
- **Top Models:** **CatBoost**, **Random Forest**, and **Extra Trees** showed the highest reliability in identifying risk factors.
- **Data Balancing:** Successfully addressed class imbalance using **SMOTE**, ensuring the model is highly sensitive to positive heart disease cases (High Recall).

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Analysis:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `CatBoost`, `XGBoost`, `LightGBM`
- **Balancing:** `SMOTE` (Imbalanced-learn)
- **Model Export:** `Joblib`

## 📑 Workflow

### 1. Exploratory Data Analysis (EDA)
- **Feature Distribution:** Analyzed age, sex, and chest pain types to understand their impact on heart health.
- **Correlation Matrix:** Identified that `cp` (chest pain), `thalach` (max heart rate), and `ca` (number of major vessels) are the most significant predictors.



### 2. Advanced Data Preprocessing
- **Feature Engineering:** Created an `age_group` feature to capture non-linear age-related risks.
- **Normalization:** Applied `StandardScaler` to ensure features like `chol` (cholesterol) and `trestbps` (resting blood pressure) are on a uniform scale.
- **Class Imbalance:** Implemented **SMOTE** to balance the dataset, preventing the model from being biased toward the majority class.

### 3. Comprehensive Model Benchmarking
The project involved an extensive comparison of:
- Logistic Regression & KNN
- Support Vector Classifier (SVC)
- Decision Trees & Random Forest
- **Boosting Algorithms:** AdaBoost, Gradient Boosting, XGBoost, LightGBM, and **CatBoost**.
- **Ensemble Techniques:** Bagging and Extra Trees.

### 4. Evaluation & Results
The final models were evaluated based on:
- **Accuracy & F1-Score:** To ensure a balance between precision and sensitivity.
- **Confusion Matrix:** Specifically focused on minimizing False Negatives to ensure no high-risk patient is missed.
- **ROC-AUC Curve:** Demonstrated excellent separation between the classes.



## 💾 Model Persistence
The final trained model and pre-processing pipeline are exported for seamless deployment:
- `final_model.pkl`: The optimized high-performance classifier.
- `scaler.pkl`: The fitted scaler for input normalization.

---
*Developed with a focus on Medical Data Science and Predictive Healthcare.*
