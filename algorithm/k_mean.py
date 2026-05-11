import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.random_state = random_state

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        random_indices = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[labels == j].mean(axis=0) if len(X[labels == j]) > 0 else self.centroids[j]
                for j in range(self.k)
            ])

            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                print(f"Converged at iteration {i}")
                break

            self.centroids = new_centroids

        # ===== FITNESS (WCSS) =====
        fitness = 0
        for j in range(self.k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                fitness += np.sum((cluster_points - self.centroids[j]) ** 2)

        return labels, fitness

    def plot_clusters(self, X, labels):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        centroids_pca = pca.transform(self.centroids)
        
        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap("viridis", len(unique_labels))

        for i, label in enumerate(unique_labels):
            cluster_data = X_pca[labels == label]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                        color=colors(i), label=f'Cluster {label}', alpha=0.6)

        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                    s=200, c='red', marker='X', edgecolors='black', label='Centroids')

        plt.title(f"K-Means Results (PCA-reduced to 2D)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()