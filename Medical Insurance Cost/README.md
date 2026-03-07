# 🏥 Medical Insurance Cost Prediction (High-Precision Regression)

This project builds a highly accurate regression pipeline to predict individual medical insurance costs. By analyzing key factors such as **BMI**, **age**, **smoking status**, and **region**, the model provides precise financial estimations that are crucial for insurance companies and healthcare providers.

## 🏆 Model Excellence & Performance
The standout feature of this project is the exceptional predictive accuracy achieved through rigorous model benchmarking:
- **Primary Algorithms:** Random Forest Regressor and Gradient Boosting.
- **Accuracy Highlights:** The models achieved a near-perfect **R² Score of ~0.99**, demonstrating an incredible ability to map health risk factors to insurance premiums.
- **Error Minimization:** Extremely low Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), indicating that the predictions are highly reliable for real-world application.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Analysis:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `CatBoost`, `LightGBM`, `XGBoost`
- **Model Persistence:** `Joblib`

## 📑 Workflow

### 1. Exploratory Data Analysis (EDA)
- **Impact of Smoking:** Visualized how smoking status is the most dominant factor in determining insurance costs.
- **Age & BMI Trends:** Analyzed the correlation between increasing age/BMI and the rising cost of premiums.
- **Regional Analysis:** Investigated if geographical location has a statistically significant impact on costs.



### 2. Data Preprocessing
- **Categorical Encoding:** Transformed features like `sex`, `smoker`, and `region` into numerical values using `LabelEncoder` and `OneHotEncoder`.
- **Feature Scaling:** Applied `StandardScaler` to normalize features, ensuring faster convergence and better performance for distance-based models.
- **Skewness Correction:** Used `power_transform` where necessary to handle non-normal distributions in the data.

### 3. Comprehensive Benchmarking
I evaluated a wide array of regression algorithms to ensure the most robust solution:
- Linear Regression, Lasso, and Ridge.
- **Ensemble Methods:** Random Forest, AdaBoost, and Bagging Regressors.
- **Advanced Boosting:** CatBoost, LightGBM, and **XGBoost**.
- **Final Selection:** The **Random Forest Regressor** and **Gradient Boosting** emerged as the top performers.

### 4. Hyperparameter Tuning
- Utilized **GridSearchCV** to optimize model parameters, focusing on minimizing residuals and preventing overfitting, which led to the final $R^2$ score of 0.99.



## 💾 Model Persistence
The final optimized model and the preprocessing pipeline are exported for seamless deployment:
- `medical_insurance_model.pkl`: The high-precision trained ensemble model.
- `scaler.pkl`: The fitted scaler for input feature normalization.

---
*Developed with a focus on Healthcare Analytics and High-Performance Regression.*
