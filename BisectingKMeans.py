import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loadDataset():
    """
    Process the dataset, split the feature string into individual float values,
    clean dataset.
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


def computeSumfSquare(X, clusters, centroids):
    """ Sum of Squared Errors (SSE) for given clustering.
        Input: data, clusters & centroids, output: sum squares."""
    sum_square = 0
    for i, c in enumerate(centroids):
        cluster_points = X[clusters == i]
        sum_square += ((cluster_points - c)**2).sum()

    return sum_square


def bisectingKmeans(X, k=9, max_iters=100):
    """
    Implements the Bisecting KMeans, choose largest cluster to split,
    Bisect with standard kmeans (k=2), choose best of multiple runs,
    split into two clusters. Input: data, clusters, output: the lables.
    """
    clusters = [X]
    cluster_labels = [np.zeros(X.shape[0], dtype=int)]

    cluster_indices = [np.arange(X.shape[0])]
    while len(clusters) < k:
        sizes = [len(c) for c in clusters]
        split_idx = np.argmax(sizes)
        cluster_to_split = clusters.pop(split_idx)
        indices_to_split = cluster_indices.pop(split_idx)
        best_labels, best_centroids, best_sumsqe = None, None, np.inf

        for _ in range(max_iters):
            labels_temp, centroids_temp = clustername(cluster_to_split, 2)
            sumsq_temp = computeSumfSquare(cluster_to_split, labels_temp, centroids_temp)
            if sumsq_temp < best_sumsqe:
                best_labels, best_centroids, best_sumsqe = labels_temp, centroids_temp, sumsq_temp

        for cluster_label in [0, 1]:
            new_cluster = cluster_to_split[best_labels == cluster_label]
            new_indices = indices_to_split[best_labels == cluster_label]
            clusters.append(new_cluster)
            cluster_indices.append(new_indices)

    full_labels = np.zeros(X.shape[0], dtype=int)
    for i, indices in enumerate(cluster_indices):
        full_labels[indices] = i

    return full_labels


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
    # Compute silhouette scores for Bisecting K-Means
    silhouette_scores_bisecting = []
    for k in range(1, 10):
        if k == 1:
            silhouette_scores_bisecting.append(0)
        else:
            labels = bisectingKmeans(X, k=k)
            score = computeSilhouette(X, labels)
            silhouette_scores_bisecting.append(score)

    silhouette_scores_kmeans = []

    for k in range(1, 10):
        labels, _ = clustername(X, k)
        score = computeSilhouette(X, labels) if k > 1 else 0
        silhouette_scores_kmeans.append(score)

    return silhouette_scores_bisecting, silhouette_scores_kmeans


def plotSilhouttee(silhouette_scores_bisecting, silhouette_scores_kmeans):
    """
    The function plot number of clusters vs
    silhouttee coefficient synthetic values and kmeans values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 10), silhouette_scores_bisecting, marker='o', linestyle='-',label="Bisecting K-Means", color="orangered")
    plt.plot(range(1, 10), silhouette_scores_kmeans, marker='s', label='KMeans++', linestyle='-.', color="teal")
    plt.title("KMeans vs Bisecting K-Means")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.xticks(range(1, 10))
    plt.legend()
    plt.show()


silhouette_scores_ = computeSilhouetteKm(loadDataset(), 9)
plotSilhouttee(silhouette_scores_[0], silhouette_scores_[1])
