# Big Mart Sales Prediction 🛒

This project aims to build a predictive model to estimate the sales of various products across different stores of a retail chain (Big Mart). By analyzing product and outlet attributes, the model helps in understanding the key factors that drive sales and improving inventory management.

## 📌 Project Overview
The dataset contains sales data for 1559 products across 10 outlets in different cities. The goal is to perform regression analysis to predict the `Item_Outlet_Sales` variable, enabling the business to optimize their sales strategy.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `XGBoost`
- **Model Optimization:** `GridSearchCV`

## 📑 Workflow

### 1. Data Cleaning & Preprocessing
- **Handling Missing Values:** Imputed missing values for `Item_Weight` (using mean) and `Outlet_Size` (using mode).
- **Data Standardization:** Corrected inconsistent labels in `Item_Fat_Content` (e.g., merging 'LF', 'low fat', and 'Low Fat').
- **Feature Engineering:** - Calculated the number of years a store has been established.
    - Simplified item categories.

### 2. Exploratory Data Analysis (EDA)
- Analyzed the distribution of sales across different types of outlets.
- Visualized the impact of `Item_Visibility` and `Item_Type` on sales.
- Correlation analysis to identify the most influential features.

### 3. Feature Transformation
- **Encoding:** Applied `LabelEncoder` for ordinal features and handled categorical variables.
- **Scaling:** Used `StandardScaler` to bring numerical features to a common scale for better model performance.

### 4. Model Training & Evaluation
Multiple regression models were implemented and compared:
- Linear Regression
- Lasso & Ridge Regression
- K-Nearest Neighbors (KNN)
- Support Vector Regressor (SVR)
- **Decision Tree & Random Forest Regressor**
- **Gradient Boosting & AdaBoost Regressor**
- **XGBoost Regressor**

### 5. Results & Performance
Models were evaluated using the **R-squared (R²)** score and **Mean Squared Error (MSE)**. Hyperparameter tuning was performed using `GridSearchCV` to find the best performing parameters for models like Random Forest and XGBoost.

## ⚙️ Setup & Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
