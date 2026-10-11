# Penguin Clustering: KMeans vs Hierarchical vs DBSCAN

Unsupervised grouping of penguins by their body measurements, comparing three clustering algorithms: KMeans, Agglomerative (Hierarchical) Clustering, and DBSCAN. The real species are held back during training and used only at the end to check the results.

## Overview

Different clustering algorithms make different assumptions. KMeans and Ward hierarchical clustering look for compact, round groups, while DBSCAN finds dense regions and labels sparse points as noise. This project runs all three on the same scaled data and compares them with internal metrics and against the known species.

## Dataset

| Item | Details |
|---|---|
| Name | Palmer Penguins |
| Source | `seaborn.load_dataset('penguins')` (internet required) |
| Samples | 344 penguins, 333 after dropping rows with missing values |
| Features used | `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g` |
| Real classes | 3 species: Adelie (152), Gentoo (124), Chinstrap (68), **not used for training** |
| Missing values | 2 rows have no measurements; 11 rows have no `sex` value |

After cleaning, 333 penguins remain: 146 Adelie, 119 Gentoo, and 68 Chinstrap.

## Project Workflow

1. **Importing libraries** and fixing the random seed.
2. **Exploratory Data Analysis**: shape, data types, summary statistics, duplicates, missing values, species counts.
3. **Data visualization**: feature histograms, correlation heatmap, pair plot.
4. **Data preprocessing**: drop missing rows, `StandardScaler`, and a 2-component PCA for plotting (the two components keep 88.1% of the variance: 68.6% + 19.5%).
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
| DBSCAN | `eps=0.6`, `min_samples=5` |

**Choosing K for KMeans.** The Elbow curve bends clearly at K = 3. The Silhouette score is highest at K = 2 (about 0.53) and decreases as K grows. K = 3 is used because it matches the elbow and the three real species.

**Choosing `eps` for DBSCAN.** The K-distance graph rises slowly until about 0.6 and then climbs sharply, so 0.6 is used as the neighborhood radius.

## Evaluation Metrics

| Metric | Type | Better when |
|---|---|---|
| Silhouette | Internal | Higher (max 1) |
| Davies-Bouldin | Internal | Lower |
| Calinski-Harabasz | Internal | Higher |
| Adjusted Rand Index (ARI, vs. species) | External | Higher (max 1) |

For DBSCAN, the internal metrics are computed on non-noise points only. The ARI is computed on all points, with noise treated as its own group.

## Results

| Model | Clusters | Noise Points | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI (vs. species) |
|---|---|---|---|---|---|---|
| DBSCAN | 2 | 32 | 0.557 | 0.672 | 498.5 | 0.579 |
| Hierarchical | 3 | 0 | 0.452 | 0.850 | 408.2 | **0.913** |
| KMeans | 3 | 0 | 0.446 | 0.942 | 427.8 | 0.799 |

### How each cluster matches the real species

| Species | KMeans (C0 / C1 / C2) | Hierarchical (C0 / C1 / C2) | DBSCAN (noise / C0 / C1) |
|---|---|---|---|
| Adelie | 124 / 0 / 22 | 0 / 146 / 0 | 15 / 131 / 0 |
| Chinstrap | 5 / 0 / 63 | 0 / 11 / 57 | 10 / 58 / 0 |
| Gentoo | 0 / 119 / 0 | 119 / 0 / 0 | 7 / 0 / 112 |

### KMeans cluster profiles (mean of each feature)

| Cluster | Bill length (mm) | Bill depth (mm) | Flipper length (mm) | Body mass (g) | Mostly |
|---|---|---|---|---|---|
| 0 | 38.3 | 18.1 | 188.6 | 3,594 | Adelie |
| 1 | 47.6 | 15.0 | 217.2 | 5,092 | Gentoo |
| 2 | 47.7 | 18.7 | 196.9 | 3,898 | Chinstrap |

## Key Findings

- **Gentoo penguins are easy to separate.** Every algorithm keeps them apart from the other species. They have the longest flippers and the heaviest bodies.
- **Hierarchical clustering matches the species best** (ARI 0.913): only 11 Chinstrap penguins fall into the Adelie cluster.
- **KMeans is less accurate on the Adelie/Chinstrap boundary** (ARI 0.799): 22 Adelie and 5 Chinstrap penguins land in the wrong cluster, because these two species overlap in their measurements.
- **DBSCAN has the best internal scores but the lowest ARI.** It finds only 2 clusters: Adelie and Chinstrap merge into one dense region, and Gentoo forms the other. It also leaves 32 points as noise, which are excluded from its internal metrics, so its scores are not directly comparable with the other two models.
- **Internal and external metrics can disagree.** A high Silhouette score means compact, well-separated groups, not groups that match the real species.

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
