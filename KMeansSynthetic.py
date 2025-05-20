import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loadDataset():
    """
    Process/clean the dataset, split the feature string into individual
    float values, clean dataset. 
    Input: path to dataset file
    Output: data array
    """
    df = pd.read_csv("dataset")
    data_split = df.iloc[:, 0].str.split(" ", n=1, expand=True)
    labels = data_split[0]
    features = data_split[1].str.strip().str.split(" ", expand=True).apply(pd.to_numeric)
    features.columns = [f"f{i}" for i in range(features.shape[1])]
    features.insert(0, "label", labels)
    X = features.drop(columns=['label']).values

    return X


def initialSelection(X, k):
    """Randomly initialize k centroids from the dataset.
        input:dataset & k, output: centroids."""
    np.random.seed(10)
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]


def generateSyntheticData(X):
    # Generate synthetic data
    n_samples, n_features = X.shape
    np.random.seed(10)
    X_synthetic = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

    return X_synthetic


def computeDistance(X, centroids):
    # Compute the distance of each point to the centroids.
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return distances


def assignClusterIds(X, centroids):
    """ Assign each point to the nearest centroid.
        Input: data & centroids, output: clusters """
    cluster = np.argmin(computeDistance(X, centroids), axis=1)
    return cluster


def computeClusterRepresentatives(X, cluster, k):
    """Recompute updated cluster representatives.
        Input: data & cluster, output: cluster_representative """
    cluster_representative = []
    for i in range(k):
        cluster_mean = np.mean(X[cluster == i], axis=0)
        cluster_representative.append(cluster_mean)

    cluster_representative = np.array(cluster_representative)
    return cluster_representative


def clustername(X, k, max_iters=100):
    """ Compute the Kmeans. input: data, k, max_iter
        Output: clusters, centroids """
    centroids = initialSelection(X, k)
    for _ in range(max_iters):
        clusters = assignClusterIds(X, centroids)
        new_centroids = computeClusterRepresentatives(X, clusters, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return clusters, centroids


def computeSilhouette(X, labels):
    # Compute average silhouette score for all samples.
    n = X.shape[0]
    silhouette_score = np.zeros(n)
    for i in range(n):
        same_cluster = X[labels == labels[i]]
        other_clusters = []
        for label in np.unique(labels):
            if label != labels[i]:
                other_clusters.append(X[labels == label])

        if len(same_cluster) > 1:
            a = np.mean(np.linalg.norm(same_cluster - X[i], axis=1))
        else:
            a = 0

        if other_clusters:
            b = min(np.mean(np.linalg.norm(cluster - X[i], axis=1)) for cluster in other_clusters)
        else:
            b = 0
        silhouette_score[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0
        mean_sihouette = np.mean(silhouette_score)

    return mean_sihouette


def computeSilhouetteKm(X, k):
    # Run KMeans for k = 1 to 9 and compute silhouette scores
    silhouette_scores_synthetic = []
    silhouette_scores_kmeans = []

    for k in range(1, 10):
        labels, _ = clustername(generateSyntheticData(X), k)
        score = computeSilhouette(generateSyntheticData(X), labels) if k > 1 else 0
        silhouette_scores_synthetic.append(score)

    for k in range(1, 10):
        labels, _ = clustername(X, k)
        score = computeSilhouette(X, labels) if k > 1 else 0
        silhouette_scores_kmeans.append(score)

    return silhouette_scores_kmeans, silhouette_scores_synthetic


def plotSilhouttee(silhouette_scores_kmeans, silhouette_scores_synthetic):
    """
    The function plot number of clusters vs
    silhouttee coefficient synthetic values and kmeans values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 10), silhouette_scores_synthetic, marker='o', label='Synthetic Data', color="crimson")
    plt.plot(range(1, 10), silhouette_scores_kmeans, marker='s', linestyle='--', label='Original Data', color="darkblue")
    plt.title("Original vs Synthetic Data")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.legend()
    plt.xticks(range(1, 10))
    plt.show()


silhouette_scores_ = computeSilhouetteKm(loadDataset(), 9)
plotSilhouttee(silhouette_scores_[0], silhouette_scores_[1])
