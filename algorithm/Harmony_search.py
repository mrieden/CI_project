import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class HarmonySearchClustering:
    def __init__(self, k, hms=20, hmcr=0.9, par=0.3, bw=0.01, random_state=None, max_iter=200):
        self.k = k
        self.hms = hms
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.max_iter = max_iter
        self.random_state = random_state
        self.history = []

    def fitness_func(self, centroids, X):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        ssd = 0
        for i in range(self.k):
            points = X[labels == i]
            if len(points) > 0:
                ssd += np.sum((points - centroids[i])**2)
        return ssd

    def fit(self, X):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        np.random.seed(self.random_state)
        n_feat = X.shape[1]
        lb, ub = X.min(), X.max()
        hm = np.random.uniform(lb, ub, (self.hms, self.k, n_feat))
        hm_fit = np.array([self.fitness_func(h, X) for h in hm])
        for _ in range(self.max_iter):
            new_h = np.zeros((self.k, n_feat))
            for i in range(self.k):
                for j in range(n_feat):
                    if np.random.rand() < self.hmcr:
                        new_h[i, j] = hm[np.random.randint(self.hms), i, j]
                        if np.random.rand() < self.par:
                            new_h[i, j] += self.bw * (np.random.rand() - 0.5)
                    else:
                        new_h[i, j] = np.random.uniform(lb, ub)
            f_new = self.fitness_func(new_h, X)
            worst = np.argmax(hm_fit)
            if f_new < hm_fit[worst]:
                hm[worst], hm_fit[worst] = new_h, f_new
            self.history.append(np.min(hm_fit))
        self.best_centroids = hm[np.argmin(hm_fit)]
        return self
    

    def plot_history(self):
        plt.plot(self.history)
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness")
        plt.title("Harmony Search Convergence")
        plt.show()
    

    def plot_clusters(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # Compute distances and assign labels
        distances = np.linalg.norm(
            X[:, np.newaxis] - self.best_centroids,
            axis=2
        )
        labels = np.argmin(distances, axis=1)

        # Reduce dimensions if features > 2
        if X.shape[1] > 2:
            pca = PCA(n_components=2)

            X_plot = pca.fit_transform(X)
            centroids_plot = pca.transform(self.best_centroids)
        else:
            X_plot = X
            centroids_plot = self.best_centroids

        # Plot clusters
        plt.figure(figsize=(8, 6))

        for i in range(self.k):
            cluster_points = X_plot[labels == i]

            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                label=f"Cluster {i}"
            )

        # Plot centroids
        plt.scatter(
            centroids_plot[:, 0],
            centroids_plot[:, 1],
            s=300,
            c='red',
            marker='X',
            label='Centroids'
        )

        plt.title("Harmony Search Clustering")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True)
        plt.show()