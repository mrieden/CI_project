import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class KMedoids:
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.medoids = None
        self.random_state = random_state

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # Initialize medoids by randomly selecting actual data points
        random_indices = np.random.choice(len(X), self.k, replace=False)
        self.medoids = X[random_indices]

        for i in range(self.max_iters):
            # Calculate Manhattan distances from all points to all medoids using broadcasting
            distances = np.sum(np.abs(X[:, np.newaxis] - self.medoids), axis=2)
            labels = np.argmin(distances, axis=1)

            new_medoids = np.zeros_like(self.medoids)

            # Update medoids by finding the point in each cluster with the minimum total distance
            for j in range(self.k):
                cluster_points = X[labels == j]
                
                # Handle empty clusters
                if len(cluster_points) == 0:
                    new_medoids[j] = self.medoids[j]
                    continue
                    
                # Vectorized pairwise Manhattan distances within the cluster
                pairwise_distances = np.sum(np.abs(cluster_points[:, np.newaxis] - cluster_points), axis=2)
                
                # The new medoid is the point with the lowest cost (sum of distances to all others in cluster)
                cost = np.sum(pairwise_distances, axis=1)
                best_medoid_idx = np.argmin(cost)
                new_medoids[j] = cluster_points[best_medoid_idx]

            # Check for convergence
            if np.all(np.abs(new_medoids - self.medoids) < self.tol):
                print(f"Converged at iteration {i}")
                break
                
            self.medoids = new_medoids
        
        return labels

    def plot_clusters(self, X, labels):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        medoids_pca = pca.transform(self.medoids)
        
        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap("viridis", len(unique_labels))

        for i, label in enumerate(unique_labels):
            cluster_data = X_pca[labels == label]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                        color=colors(i), label=f'Cluster {label}', alpha=0.6)

        plt.scatter(medoids_pca[:, 0], medoids_pca[:, 1], 
                    s=200, c='red', marker='X', edgecolors='black', label='Medoids')

        plt.title("K-Medoids Results (PCA-reduced to 2D)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()