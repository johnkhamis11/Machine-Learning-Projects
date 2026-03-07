# 🏠 House Price Prediction using Advanced Regression

This project builds a high-performance machine learning pipeline to predict real estate prices. By analyzing geographical, demographic, and structural features (such as median income, house age, and proximity to the ocean), the model provides accurate valuation estimates essential for real estate analytics.

## 📌 Project Overview
The objective is to predict the `median_house_value` using various independent variables. The project handles complex data challenges, including non-linear relationships, spatial features, and multi-collinearity.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Analysis:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `XGBoost`
- **Model Persistence:** `Joblib`

## 📑 Workflow

### 1. Exploratory Data Analysis (EDA)
- **Feature Correlation:** Identified that `median_income` is the strongest predictor of house value.
- **Spatial Analysis:** Visualized the impact of geographical location (Latitude/Longitude) and ocean proximity on pricing.
- **Distribution Study:** Analyzed the skewness of features like `total_rooms` and `population`.



### 2. Data Preprocessing
- **Missing Value Imputation:** Handled null values in features like `total_bedrooms` using statistical measures.
- **Categorical Encoding:** Converted `ocean_proximity` and other text features into numerical formats using `OneHotEncoder` and `LabelEncoder`.
- **Feature Scaling:** Applied `StandardScaler` to normalize the data, which significantly improved the performance of distance-based models.

### 3. Model Training & Benchmarking
I performed an extensive comparison across several regression algorithms to ensure the highest $R^2$ score:
- Linear Regression, Lasso, and Ridge.
- K-Nearest Neighbors (KNN) & Support Vector Regressor (SVR).
- **Ensemble Learning:** Random Forest, AdaBoost, and Bagging Regressors.
- **Advanced Boosting:** **XGBoost** and Gradient Boosting (GBM).

### 4. Evaluation & Results
The models were evaluated using:
- **R-Squared ($R^2$):** To measure the goodness of fit.
- **Mean Absolute Error (MAE):** To understand the average prediction error in dollar terms.
- **Visual Validation:** Created Scatter plots of Predicted vs. Actual values to verify model consistency.



## 💾 Model Persistence
The final optimized model and the preprocessing scaler are exported for production use:
- `house_price_prediction.pkl`: The final trained ensemble model.
- `scaler.pkl`: The fitted scaler for consistent data transformation.

---
*Developed with a focus on Real Estate Analytics and Predictive Modeling.*
