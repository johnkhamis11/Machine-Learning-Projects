# Autism Disease Prediction

## 📌 Project Overview
This project aims to predict the likelihood of Autism Spectrum Disorder (ASD) based on behavioral and demographic features. Using machine learning classification, the model provides an early screening tool to assist in clinical diagnosis.

## 📊 Dataset Description
The dataset consists of clinical and behavioral data points:
* **Features**: Includes screening questions (A1-A10), age, gender, ethnicity, and jaundice history.
* **Target**: `Class/ASD` (Binary: Yes/No).
* **Preprocessing**: Handled missing values, encoded categorical variables, and performed feature scaling.

## 🛠️ Tech Stack
* **Language**: Python
* **Libraries**: 
    * **Data Analysis**: Pandas, NumPy
    * **Visualization**: Matplotlib, Seaborn
    * **Machine Learning**: Scikit-learn

## 🤖 Model & Performance
The project implements a classification pipeline:
* **Algorithm**: Logistic Regression (LogReg)
* **Evaluation Metrics**:
    * **Accuracy**: High precision and recall scores achieved on the test set.
    * **Visualization**: Includes a Confusion Matrix and Classification Report to verify model reliability.

## 📂 Repository Structure
* `Autism_Prediction.ipynb`: Complete data analysis and model training pipeline.
* `train.csv`: The dataset used for training and testing.
* `README.md`: Project documentation.
