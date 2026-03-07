# Calories Burnt Prediction using Machine Learning 🔥

This project aims to develop a regression model that predicts the number of calories burnt during physical activities. By analyzing factors such as duration, heart rate, body temperature, and user demographics, the model provides accurate estimations to help individuals monitor their fitness progress.

## 📌 Project Overview
The dataset consists of exercise data paired with personal information (age, gender, BMI). The core objective is to utilize regression techniques to find the relationship between physical intensity and energy expenditure.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `XGBoost`
- **Model Export:** `Joblib`

## 📑 Workflow

### 1. Data Integration & Cleaning
- Combined two separate datasets: one containing personal attributes and another containing calorie data.
- Handled categorical data (Gender) by converting it into numerical format (Mapping/Encoding).
- Checked for and handled any missing values to ensure data integrity.

### 2. Exploratory Data Analysis (EDA)
- **Correlation Analysis:** Identified a strong positive correlation between `Heart_Rate`, `Duration`, and `Calories`.
- **Distribution Analysis:** Visualized the age distribution and gender representation in the dataset.
- **Visual Insights:** Used scatter plots and regression plots to understand how body temperature and heart rate impact calorie burn.

### 3. Model Training & Selection
The project involved benchmarking various regression algorithms:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- **XGBoost Regressor** (Final model choice due to high accuracy)

### 4. Model Optimization
- Fine-tuned the **XGBoost** model using optimal parameters:
  - `learning_rate`: 0.1
  - `max_depth`: 5
  - `n_estimators`: 200
- Evaluated performance using **Mean Absolute Error (MAE)** and **R-squared (R²)** score.

### 5. Evaluation & Results
The model achieved high precision in predicting calories, making it a reliable tool for fitness tracking applications. Feature importance analysis showed that `Duration` and `Heart_Rate` are the most significant factors in the prediction.

## 💾 Model Persistence
The final trained model and the feature scaler have been saved for future deployment:
- `model.joblib`: The optimized XGBoost model.
- `scaler.joblib`: The pre-fitted scaler for consistent input normalization.

## ⚙️ Setup & Installation
1. Clone this repository.
2. Install the necessary libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
