# Handwritten Digits Clustering: KMeans vs Gaussian Mixture vs Hierarchical

Unsupervised grouping of 1,797 handwritten digit images into 10 clusters, without ever using the labels for training. PCA reduces the dimensions before clustering, and t-SNE is used to visualize the result. The real digits are used only at the end to check how well each algorithm recovered them.

## Overview

Can an algorithm discover digits just by looking at pixels? This project clusters small 8×8 images with three different algorithms and checks the groups in three ways: internal metrics, visual inspection (t-SNE plots and cluster centroid images), and agreement with the true digits.

## Dataset

| Item | Details |
|---|---|
| Name | Optical Recognition of Handwritten Digits (UCI) |
| Source | `sklearn.datasets.load_digits` (built in, no internet needed) |
| Samples | 1,797 images, 8×8 grayscale |
| Features | 64 pixel values (0 to 16) |
| Real classes | 10 digits (about 180 images each), **not used for training** |
| Missing values | None |

## Project Workflow

1. **Importing libraries** and fixing the random seed.
2. **Exploratory Data Analysis**: shapes, pixel range, summary statistics, duplicates, missing values, digit counts.
3. **Data visualization**: sample digits, real digit distribution, the average image of each digit, and the average pixel intensity map.
4. **Data preprocessing**:
   - Scale pixels to [0, 1].
   - Cumulative explained variance plot, then PCA keeping 95% of the variance (64 features reduced to 29).
   - t-SNE projection to 2D for plotting.
5. **Finding the best K**: KMeans for K = 2 to 15 with Inertia and Silhouette (Elbow and Silhouette plots).
6. **KMeans** with 10 clusters.
7. **Gaussian Mixture Model** with 10 components.
8. **Hierarchical clustering** (Ward linkage) with 10 clusters.
9. **Evaluating the clusters**: t-SNE plots of the real digits and the three clusterings, KMeans centroid images (what each cluster "looks like"), and a heatmap of real digit vs. cluster.
10. **Model comparison**: one table and a bar plot.

## Algorithm Settings

| Algorithm | Settings |
|---|---|
| KMeans | `n_clusters=10`, `n_init=10`, `random_state=42` |
| Gaussian Mixture | `n_components=10`, `covariance_type='full'`, `random_state=42` |
| Hierarchical | `AgglomerativeClustering`, `n_clusters=10`, Ward linkage |

All three run on the 29-dimensional PCA representation.

## Evaluation Metrics

| Metric | Type | Better when |
|---|---|---|
| Silhouette | Internal | Higher (max 1) |
| Davies-Bouldin | Internal | Lower |
| Calinski-Harabasz | Internal | Higher |
| Adjusted Rand Index (ARI) | External (uses real digits) | Higher (max 1) |
| Normalized Mutual Information (NMI) | External (uses real digits) | Higher (max 1) |

## Results

| Model | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI | NMI |
|---|---|---|---|---|---|
| Hierarchical | 0.192 | 1.850 | 172.8 | 0.743 | 0.828 |
| Gaussian Mixture | 0.189 | 1.901 | 178.8 | 0.696 | 0.782 |
| KMeans | 0.195 | 1.840 | 184.4 | 0.662 | 0.738 |

Results from one run with `random_state=42`; small differences are possible across library versions.

An interesting observation: the internal metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) slightly favor KMeans, while the external metrics (ARI, NMI) favor Hierarchical clustering. Internal metrics measure how compact and separated the clusters are, not whether they match the real digits, so the two views can disagree. In the K sweep, the Silhouette score peaks at K = 9 (about 0.20) and then stays roughly flat between 0.19 and 0.20 up to K = 15, so the Elbow and Silhouette plots do not point clearly to 10 by themselves. K = 10 is used because the data contains 10 digits.

## Tech Stack

Python · scikit-learn · NumPy · Pandas · Matplotlib · Seaborn · joblib

## Getting Started

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

Open `Digits_Clustering_KMeans_GMM_Hierarchical.ipynb` in Jupyter or [Google Colab](https://colab.research.google.com/) and run all cells. No GPU or internet connection is needed.

## Project Structure

```
.
├── Digits_Clustering_KMeans_GMM_Hierarchical.ipynb
└── README.md
```

## What This Project Demonstrates

- Reducing dimensions with PCA before clustering, and choosing the number of components from explained variance.
- Visualizing high-dimensional clusters with t-SNE.
- Interpreting a cluster by looking at its centroid image.
- Comparing algorithms with both internal and external metrics, and understanding why they can disagree.

## Possible Improvements

- Choose the number of Gaussian components with BIC or AIC.
- Try DBSCAN or Spectral Clustering.
- Repeat the experiment on the larger MNIST dataset (28×28 images).
- Use a CNN autoencoder to learn features before clustering.
