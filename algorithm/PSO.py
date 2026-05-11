import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class PSOClustering:
    def __init__(self, k=4, n_particles=20, max_iters=100,
                w=0.7, c1=1.5, c2=1.5, random_state=None):
        self.k = k
        self.n_particles = n_particles
        self.max_iters = max_iters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.random_state = random_state
        self.centroids = None
        self.history = []

    def assign_clusters(self, X, centroids):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fitness(self, X, centroids):
        """Returns the Sum of Squared Distances (SSD) across all clusters."""
        labels = self.assign_clusters(X, centroids)
        ssd = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                ssd += np.sum((cluster_points - centroids[i]) ** 2)  # SSD
        return ssd

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Initialize particles
        particles = np.random.rand(self.n_particles, self.k, n_features)
        velocities = np.zeros_like(particles)

        # pBest
        pbest = particles.copy()
        pbest_scores = np.array([self.fitness(X, p) for p in particles])

        # gBest
        gbest = pbest[np.argmin(pbest_scores)]

        # Main loop
        for iteration in range(self.max_iters):
            for i in range(self.n_particles):
                score = self.fitness(X, particles[i])
                if score < pbest_scores[i]:
                    pbest[i] = particles[i].copy()
                    pbest_scores[i] = score

            gbest = pbest[np.argmin(pbest_scores)]
            self.history.append(np.min(pbest_scores))  # track best SSD per iteration

            for i in range(self.n_particles):
                r1 = np.random.rand()
                r2 = np.random.rand()
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (pbest[i] - particles[i])
                    + self.c2 * r2 * (gbest - particles[i])
                )
                particles[i] += velocities[i]

        self.centroids = gbest
        self.ssd = self.fitness(X, self.centroids)  # store final SSD
        return self

    def predict(self, X):
        return self.assign_clusters(X, self.centroids)

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

        plt.title("PSO Clustering Results (PCA-reduced to 2D)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def plot_convergence(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.history, linewidth=2)
        plt.title("PSO Convergence Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Sum of Squared Distances (SSD)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()