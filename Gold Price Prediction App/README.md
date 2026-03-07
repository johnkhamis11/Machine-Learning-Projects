# 💰 Gold Price Prediction (High-Precision Regression)

This project implements a sophisticated machine learning pipeline to predict gold prices (`GLD`) by analyzing its relationship with key economic indicators such as the S&P 500 index (`SPX`), Silver prices (`SLV`), US Oil prices (`USO`), and currency exchange rates (`EUR/USD`).

## 🏆 Model Excellence & Performance
The standout feature of this project is the exceptional predictive accuracy achieved through rigorous model selection and tuning:
- **Primary Algorithms:** Random Forest Regressor and KNeighbors Regressor.
- **Accuracy Highlights:** The models achieved a near-perfect **R² Score of ~0.98 - 0.99**, demonstrating an incredible ability to capture the market's volatility.
- **Low Residuals:** Minimal Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) values, indicating that predicted prices are highly aligned with actual market data.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Analysis:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `CatBoost`, `LightGBM`, `XGBoost`
- **Model Persistence:** `Joblib`

## 📑 Workflow

### 1. Feature Analysis & Correlation
- **Economic Indicators:** Analyzed how assets like Silver (`SLV`) and the Stock Market (`SPX`) influence Gold prices.
- **Heatmap Insights:** Generated a correlation matrix identifying that Silver (`SLV`) has the strongest positive correlation with Gold prices.



### 2. Data Preprocessing
- **Feature Scaling:** Applied `StandardScaler` to ensure the different price ranges of the S&P 500 and Oil don't bias the model.
- **Splitting:** Used an 80/20 train-test split to rigorously validate the model's performance on unseen data.

### 3. Comprehensive Benchmarking
I evaluated a wide array of regressors to find the most robust solution:
- Linear Regression, Lasso, and Ridge.
- **Ensemble Methods:** Random Forest, AdaBoost, Gradient Boosting, and Bagging.
- **Advanced Boosting:** CatBoost and LightGBM.
- **Final Selection:** The **Random Forest Regressor** and **KNN** provided the most stable and accurate results.

### 4. Performance Visualization
- **Actual vs. Predicted:** Created plots that show the predicted values almost perfectly overlapping with the actual gold prices over time.



## 💾 Model Persistence
The final optimized model is exported and ready for deployment into financial applications:
- `best_model.pkl`: The high-accuracy trained regressor.
- `scaler.pkl`: The fitted scaler for input feature normalization.

---
*Developed with a focus on Financial Analytics and High-Performance Regression.*
