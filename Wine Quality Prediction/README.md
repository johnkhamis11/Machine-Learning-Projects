# 🍷 Wine Quality Prediction (High-Performance Classification)

This project develops an advanced classification model to predict the quality of wine based on its chemical properties. By leveraging high-performance boosting algorithms and detailed feature analysis, the model provides a reliable way to categorize wine quality for production and quality control.

## 🏆 Model Excellence & Performance
The highlight of this project is the achievement of superior classification accuracy through state-of-the-art algorithms:
- **Primary Algorithm:** **CatBoost Classifier**.
- **Performance Highlights:** The model achieved exceptional precision in distinguishing between different quality levels.
- **Top Predictors:** Identified that `alcohol`, `sulphates`, and `volatile acidity` are the most critical factors influencing wine quality.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Analysis:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `XGBoost`, `LightGBM`, **`CatBoost`**
- **Preprocessing:** `StandardScaler`, `LabelEncoder`

## 📑 Workflow

### 1. Exploratory Data Analysis (EDA)
- **Chemical Insights:** Investigated how acidity and alcohol content correlate with the quality rating.
- **Correlation Matrix:** Generated a heatmap to detect multi-collinearity among chemical components.
- **Outlier Management:** Analyzed the distribution of features like `residual sugar` and `chlorides` to ensure data robustness.



### 2. Data Preprocessing
- **Categorical Transformation:** Used `LabelEncoder` to convert quality ratings into a format suitable for machine learning.
- **Feature Scaling:** Applied `StandardScaler` to ensure that features with different magnitudes (e.g., `total sulfur dioxide` vs. `pH`) are processed fairly by the model.
- **Data Splitting:** Implemented a stratified split to maintain the distribution of quality classes in both training and testing sets.

### 3. Comprehensive Model Benchmarking
I evaluated a wide variety of classification techniques:
- Logistic Regression & KNN.
- **Support Vector Machine (SVM)** and Random Forest.
- **Advanced Ensemble Methods:** Extra Trees, AdaBoost, and Bagging.
- **Boosting Excellence:** Compared XGBoost and LightGBM against **CatBoost**, with the latter providing the best balance of speed and accuracy.

### 4. Feature Importance Analysis
Using the CatBoost model's internal metrics, I visualized the importance of each feature. This analysis revealed that `alcohol` content is the single most important factor in determining perceived wine quality.



## 📊 Key Findings
- High-quality wines consistently show higher **alcohol** and **sulphates** levels.
- **Volatile acidity** showed a negative correlation with quality, meaning higher acidity often leads to lower quality ratings.
- Ensemble models, specifically **CatBoost**, proved to be far more effective than traditional linear models for this dataset.

---
*Developed with a focus on Chemical Analytics and Advanced Boosting Techniques.*
