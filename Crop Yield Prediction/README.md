# Crop Yield Prediction using Machine Learning 🌾

This project aims to predict agricultural crop yields by analyzing environmental factors and historical data. By leveraging machine learning, we can provide valuable insights for farmers and policymakers to optimize food production and manage resources more effectively.

## 📌 Project Overview
The dataset integrates various factors that influence crop productivity, including weather conditions, soil quality, and pesticide usage. The goal is to build a regression model that accurately estimates the `hg/ha_yield` (hectogram per hectare) for different crops and regions.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`
- **Primary Model:** `RandomForestRegressor`

## 📑 Workflow

### 1. Exploratory Data Analysis (EDA)
- **Trend Analysis:** Visualized how crop yields have changed over the years across different countries.
- **Correlation Study:** Examined the relationship between average rainfall, pesticides, and temperature on the final yield.
- **Outlier Detection:** Identified and handled anomalies in the dataset to ensure model stability.

### 2. Data Preprocessing
- **Categorical Encoding:** Used `LabelEncoder` to transform categorical variables such as `Area`, `Item`, and `Year` into numerical formats.
- **Data Splitting:** Divided the data into training (80%) and testing (20%) sets to validate model performance.
- **Feature Selection:** Focused on key drivers: Average Rainfall, Pesticides, and Temperature.

### 3. Model Training
The project primarily utilizes the **Random Forest Regressor** due to its ability to handle non-linear relationships and its robustness against overfitting in agricultural datasets.

### 4. Evaluation & Performance
- **R-squared (R²) Score:** Used to measure the proportion of variance explained by the model.
- **Mean Squared Error (MSE):** Used to quantify the prediction accuracy.
- **Feature Importance:** Generated a visualization to show which factors (e.g., Rainfall vs. Temperature) contribute most to the yield prediction.



## 📊 Key Results
- The model successfully identifies high-yield patterns based on specific climate conditions.
- Visualization of the results shows a strong alignment between predicted and actual yield values.

## ⚙️ Setup & Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
