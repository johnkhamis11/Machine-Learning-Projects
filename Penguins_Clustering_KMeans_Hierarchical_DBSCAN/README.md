# Penguin Clustering: KMeans vs Hierarchical vs DBSCAN

Unsupervised grouping of penguins by their body measurements, comparing three clustering algorithms: KMeans, Agglomerative (Hierarchical) Clustering, and DBSCAN. The real species are held back and used only to check the results.

## Overview

Different clustering algorithms make different assumptions: KMeans and Ward hierarchical clustering look for compact round groups, while DBSCAN finds dense regions and marks sparse points as noise. This project runs all three on the same scaled data and compares them with internal metrics and against the known species.

## Dataset

| Item | Details |
|---|---|
| Name | Palmer Penguins |
| Source | `seaborn.load_dataset('penguins')` (internet required) |
| Samples | 344 penguins (rows with missing values are dropped) |
| Features used | `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g` |
| Real classes | 3 species: Adelie, Chinstrap, Gentoo, **not used for training** |

## Project Workflow

1. **Importing libraries** and fixing the random seed.
2. **Exploratory Data Analysis**: shape, data types, summary statistics, duplicates, missing values, species counts.
3. **Data visualization**: feature histograms, correlation heatmap, pair plot.
4. **Data preprocessing**: drop missing rows, `StandardScaler`, and a 2-component PCA for plotting.
5. **KMeans**: Elbow and Silhouette plots for K = 2 to 8, then K = 3.
6. **Hierarchical clustering**: dendrogram (Ward linkage), then 3 clusters.
7. **DBSCAN**: K-distance graph to guide `eps`, then `eps = 0.6` and `min_samples = 5`.
8. **Evaluating the clusters**: PCA plots of the real species and the three clusterings, KMeans cluster profiles, and cross-tabulations against species.
9. **Model comparison**: one table and a bar plot.

## Algorithm Settings

| Algorithm | Settings |
|---|---|
| KMeans | `n_clusters=3`, `n_init=10`, `random_state=42` |
| Hierarchical | `AgglomerativeClustering`, `n_clusters=3`, Ward linkage |
| DBSCAN | `eps=0.6`, `min_samples=5` (tune `eps` with the K-distance graph) |

## Evaluation Metrics

| Metric | Type | Better when |
|---|---|---|
| Silhouette | Internal | Higher (max 1) |
| Davies-Bouldin | Internal | Lower |
| Calinski-Harabasz | Internal | Higher |
| Adjusted Rand Index (vs. species) | External | Higher (max 1) |

For DBSCAN, the internal metrics are computed on non-noise points only, and the number of noise points is reported.

## Results

<!-- Fill in after running the notebook -->

| Model | Clusters | Noise Points | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI (vs. species) |
|---|---|---|---|---|---|---|
| KMeans | — | — | — | — | — | — |
| Hierarchical | — | — | — | — | — | — |
| DBSCAN | — | — | — | — | — | — |

## Tech Stack

Python · scikit-learn · SciPy · NumPy · Pandas · Matplotlib · Seaborn · joblib

## Getting Started

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib
```

Open `Penguins_Clustering_KMeans_Hierarchical_DBSCAN.ipynb` in Jupyter or [Google Colab](https://colab.research.google.com/) and run all cells. No GPU is needed; an internet connection is required to load the dataset.

## Project Structure

```
.
├── Penguins_Clustering_KMeans_Hierarchical_DBSCAN.ipynb
└── README.md
```

## What This Project Demonstrates

- How different clustering algorithms behave on the same data.
- Reading a dendrogram and a K-distance graph to choose parameters.
- Evaluating clusters both internally and against known labels, and why the two can disagree.
- Handling noise points explicitly with DBSCAN.

## Possible Improvements

- Search `eps` and `min_samples` systematically for DBSCAN.
- Add Gaussian Mixture Models and compare.
- Include categorical features (island, sex) with a suitable encoding.
- Visualize clusters with t-SNE or UMAP in addition to PCA.
