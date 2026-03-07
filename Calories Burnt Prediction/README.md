# 🚀 Ultra-Precision Calories Burnt Prediction

This project implements a state-of-the-art regression pipeline to predict calories burnt during physical activities with **near-perfect accuracy**. By integrating physiological data with exercise metrics, the model achieves an extraordinary level of precision, making it suitable for high-end fitness analytics.

## 🏆 Model Excellence & Performance
The model performance stands out as the core highlight of this project:
- **Primary Algorithm:** XGBoost Regressor (Fine-tuned).
- **R² Score:** **~0.999** (99.9% variance explained), indicating an almost perfect fit to the data.
- **Mean Absolute Error (MAE):** Minimal error rates, showcasing the model's reliability in predicting exact calorie expenditure.
- **Optimization:** Achieving this level of precision involved meticulous feature engineering and hyperparameter optimization ($learning\_rate=0.1, max\_depth=5, n\_estimators=200$).

## 🛠️ Tech Stack
- **Languages & Frameworks:** Python, Scikit-learn, XGBoost.
- **Data Handling:** Pandas, NumPy.
- **Visualization:** Seaborn, Matplotlib.
- **Deployment Ready:** Models and scalers are exported using `joblib` for production use.

## 📑 Detailed Workflow

### 1. Advanced Data Preprocessing
- **Seamless Integration:** Merged exercise logs with user physical profiles (Gender, Age, BMI).
- **Feature Transformation:** Encoded categorical features and normalized numerical values using `StandardScaler` to ensure optimal model convergence.

### 2. Exploratory Data Analysis (EDA)
- **Correlation Insights:** Discovered a massive linear correlation between `Heart_Rate`, `Duration`, and `Calories`.
- **Distribution Study:** Analyzed the impact of body temperature and age on metabolic rate during exercise.



### 3. Model Benchmarking
While several models were tested (Linear Regression, Decision Trees, Random Forest), the **XGBoost Regressor** was selected as the final model for its superior ability to minimize residuals and handle complex feature interactions.

## 💾 Model Persistence
The project includes ready-to-use exported files for seamless integration:
- `model.joblib`: The high-precision trained XGBoost model.
- `scaler.joblib`: The pre-fitted scaler used for input normalization.

---
*Developed with a focus on High-Performance Predictive Modeling.*
