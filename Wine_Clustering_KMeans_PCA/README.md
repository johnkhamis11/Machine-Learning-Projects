# Wine Clustering with KMeans and PCA

Unsupervised grouping of 178 Italian wines by their chemical properties using KMeans, with PCA for visualization. The real wine types are held back and used only at the end to check how well the clusters match.

## Overview

Clustering has no labels to learn from, so the main questions are: how many groups are there, and how good are they? This project answers the first with the Elbow method and Silhouette score, and the second with three internal metrics plus an external comparison against the known wine cultivars.

## Dataset

| Item | Details |
|---|---|
| Name | Wine dataset (UCI) |
| Source | `sklearn.datasets.load_wine` (built in, no internet needed) |
| Samples | 178 wines |
| Features | 13 chemical measurements (alcohol, malic acid, flavanoids, color intensity, proline, and others) |
| Real classes | 3 cultivars (59 / 71 / 48 samples), **not used for training** |
| Missing values | None |

## Project Workflow

1. **Importing libraries** and fixing the random seed.
2. **Exploratory Data Analysis**: shape, data types, summary statistics, duplicates, missing values.
3. **Data visualization**: feature histograms, correlation heatmap, pair plot of four key features.
4. **Data preprocessing**: `StandardScaler` (features have very different scales) and a 2-component PCA for plotting.
5. **Finding the best K**: KMeans for K = 2 to 10, recording Inertia, Silhouette, Davies-Bouldin, and Calinski-Harabasz. Elbow and Silhouette plots.
6. **KMeans model** with K = 3.
7. **Evaluating the clusters**: internal metrics, PCA scatter with centroids, cluster profiles (mean of each feature per cluster), and a centroid heatmap.
8. **Comparison with the real wine types**: Adjusted Rand Index, a cross-tabulation, and side-by-side PCA plots.

## Evaluation Metrics

| Metric | Type | Better when |
|---|---|---|
| Inertia | Internal (Elbow method) | Lower (look for the bend) |
| Silhouette | Internal | Higher (max 1) |
| Davies-Bouldin | Internal | Lower |
| Calinski-Harabasz | Internal | Higher |
| Adjusted Rand Index | External (uses real labels) | Higher (max 1) |

## Results (K = 3)

| Metric | Value |
|---|---|
| Cluster sizes | 65 / 51 / 62 |
| Silhouette | 0.285 |
| Davies-Bouldin | 1.389 |
| Calinski-Harabasz | 70.94 |
| Adjusted Rand Index vs. real wine types | 0.897 |
| Variance explained by the 2 PCA components | 36.2% + 19.2% |

The clusters, found without using any labels, closely match the three real cultivars.

## Tech Stack

Python · scikit-learn · NumPy · Pandas · Matplotlib · Seaborn · joblib

## Getting Started

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

Open `Wine_Clustering_KMeans_PCA.ipynb` in Jupyter or [Google Colab](https://colab.research.google.com/) and run all cells. No GPU or internet connection is needed.

## Project Structure

```
.
├── Wine_Clustering_KMeans_PCA.ipynb
└── README.md
```

## What This Project Demonstrates

- Why feature scaling is essential for distance-based algorithms like KMeans.
- Choosing K with the Elbow method and several validation metrics instead of guessing.
- Using PCA to visualize high-dimensional clusters in 2D.
- Interpreting clusters through their feature profiles, and validating them externally.

## Possible Improvements

- Try Gaussian Mixture Models or Hierarchical Clustering.
- Compare other scalers or remove highly correlated features.
- Add a silhouette plot per sample to inspect individual cluster quality.
