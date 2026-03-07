# 👥 Customer Segmentation using K-Means Clustering

This project focuses on identifying distinct groups within a customer base using Unsupervised Machine Learning. By segmenting customers based on demographics and spending patterns, businesses can tailor their marketing strategies and improve customer relationship management (CRM).

## 📌 Project Overview
The objective is to analyze a mall customer dataset to discover hidden patterns. The model segments customers based on features such as **Age**, **Annual Income**, and **Spending Score**, helping to identify "Target Customers" and "High-Value Segments."

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Analysis:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn` (K-Means Clustering)
- **Evaluation:** `Silhouette Score`, `Elbow Method`

## 📑 Workflow

### 1. Exploratory Data Analysis (EDA)
- **Distribution Study:** Visualized the age, income, and spending score distributions.
- **Gender Analysis:** Analyzed how spending patterns differ between male and female customers.
- **Correlation Mapping:** Identified relationships between income levels and spending habits.

### 2. Data Preprocessing
- **Feature Selection:** Focused on relevant metrics for clustering (Annual Income vs. Spending Score).
- **Scaling:** Applied `StandardScaler` to ensure that features with different scales (like age vs. income) are treated equally by the clustering algorithm.

### 3. Determining Optimal Clusters
- **Elbow Method:** Implemented the Within-Cluster Sum of Squares (WCSS) to identify the "elbow" point, ensuring the most efficient number of clusters.
- **Silhouette Analysis:** Validated the consistency within clusters to ensure distinct and well-separated groups.



### 4. K-Means Implementation
Applied the K-Means algorithm to segment the data into optimal clusters. Each customer was assigned a label representing their specific segment.

### 5. Cluster Profiling & Insights
Analyzed the mean values of each cluster to define customer personas:
- **High Earners, High Spenders:** The primary target for luxury promotions.
- **High Earners, Low Spenders:** Potential for targeted marketing to increase engagement.
- **Low Earners, High Spenders:** Impulsive buyers.
- **Average Spenders:** The stable middle-class segment.



## 📊 Key Results
- **Optimized Clustering:** Successfully identified the ideal number of segments (e.g., 5-6 clusters) that maximize business utility.
- **Data-Driven Personas:** Provided a clear statistical profile for each customer group, allowing for precise marketing targeting.

---
*Developed with a focus on Unsupervised Learning and Marketing Analytics.*
