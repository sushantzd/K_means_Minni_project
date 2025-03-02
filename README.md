# K-Means Clustering - Mini Project

## 📌 Project Overview

This project demonstrates the implementation of **K-Means Clustering** on a synthetic dataset generated using `make_blobs` from Scikit-learn. It includes:

- Cluster visualization
- Finding the optimal number of clusters using the **Elbow Method** and **Silhouette Score**
- Performance evaluation of K-Means clustering

## 📂 Dataset

The dataset is created using:

```python
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=1000, centers=2, n_features=10, random_state=42)
```

- `n_samples=1000`: 1000 data points
- `centers=2`: Two actual clusters
- `n_features=10`: 10-dimensional feature space

## 🚀 Implementation

### 1️⃣ Finding Optimal Clusters

#### **Elbow Method**

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia_values = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

plt.plot(K, inertia_values, 'bo-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()
```

#### **Silhouette Score**

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, labels))
```

### 2️⃣ Final Clustering

Using the best **k** from Silhouette Score:

```python
best_k = 2  # From Silhouette Score
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
```

## 📊 Visualization

Since data is **high-dimensional (10D)**, we use **PCA (Principal Component Analysis)** for 2D projection:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering with PCA")
plt.show()
```

## 🎯 Results

- **Elbow Method suggests 6 clusters**, but high-dimensional data may cause overestimation.
- **Silhouette Score correctly suggests 2 clusters**, matching ground truth.
- **Visualization confirms correct clustering.**

## 🛠️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Libraries Used

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

## 📌 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/sushantzd/KMeans-Clustering-Project.git
   ```
2. Navigate to the project folder:
   ```bash
   cd KMeans-Clustering-Project
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook "K Means Clustering.ipynb"
   ```

## 📜 Author

[Sushant Choudhary](https://github.com/sushantzd)

