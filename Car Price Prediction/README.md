# Car Price Prediction using Machine Learning 🚗

This project aims to develop a predictive model that estimates the selling price of used cars based on various features such as the car's age, mileage, fuel type, transmission, and owner history. This tool can help both buyers and sellers determine a fair market value for a vehicle.

## 📌 Project Overview
The dataset contains information about used cars, including technical specifications and historical data. The goal is to perform regression analysis to predict the continuous variable `Selling_Price` with high precision.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`
- **Models Used:** Linear Regression, Random Forest, Gradient Boosting

## 📑 Workflow

### 1. Exploratory Data Analysis (EDA)
- **Feature Analysis:** Visualized the relationship between `Year` and `Selling_Price` (identifying that newer cars generally command higher prices).
- **Categorical Insights:** Analyzed how `Fuel_Type` (Petrol vs. Diesel) and `Transmission` (Manual vs. Automatic) affect the car's value.
- **Correlation Heatmap:** Identified strong correlations between the `Present_Price` and the `Selling_Price`.

### 2. Data Preprocessing
- **Feature Engineering:** Created a new feature to calculate the age of the car from the current year.
- **Categorical Encoding:** Converted categorical variables like `Fuel_Type`, `Seller_Type`, and `Transmission` into numerical values using `LabelEncoder`.
- **Data Splitting:** Divided the dataset into training and testing sets to evaluate model generalization.
- **Scaling:** Applied `StandardScaler` to normalize numerical features for better model convergence.

### 3. Model Training & Benchmarking
Implemented and compared multiple regression models:
- **Linear Regression:** Used as a baseline model.
- **Random Forest Regressor:** Captured non-linear relationships and provided high accuracy.
- **Gradient Boosting Regressor:** Optimized for the best performance through ensemble learning.

### 4. Hyperparameter Tuning
- Utilized **GridSearchCV** to fine-tune the `Gradient Boosting` and `Random Forest` models.
- Optimized parameters such as `n_estimators`, `max_depth`, and `learning_rate` to minimize the Root Mean Squared Error (RMSE).

### 5. Evaluation & Results
The models were evaluated using:
- **R-squared (R²) Score:** To measure how well the model explains the variance in the data.
- **Root Mean Squared Error (RMSE):** To measure the average magnitude of the prediction error.
- The **Gradient Boosting** model emerged as the top performer with the highest Test R² score.

## ⚙️ Setup & Installation
1. Clone the repository.
2. Install the necessary libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
