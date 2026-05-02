import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        # 1. Initialize centroids randomly from the dataset
        random_indices = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # 2. Assignment Step: Find the closest centroid for each point
            # We use the Euclidean distance formula: sqrt(sum((x - y)^2))
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # 3. Update Step: Move centroids to the mean of assigned points
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.k)])

            # 4. Check for Convergence: If centroids don't move much, stop early
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                print(f"Converged at iteration {i}")
                break
                
            self.centroids = new_centroids
        
        return labels

    def plot_clusters(self, X, labels):
        # 1. Fit PCA on the data
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # 2. Transform centroids using the SAME PCA fit
        centroids_pca = pca.transform(self.centroids)
        
        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap("viridis", len(unique_labels))

        for i, label in enumerate(unique_labels):
            cluster_data = X_pca[labels == label]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                        color=colors(i), label=f'Cluster {label}', alpha=0.6)

        # Plot the transformed centroids
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                    s=200, c='red', marker='X', edgecolors='black', label='Centroids')

        plt.title(f"K-Means Results (PCA-reduced to 2D)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()