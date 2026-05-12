from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt


class FuzzyCMeans:
    def __init__(self, k=3, m=2.0, max_iters=100, tol=1e-4, random_state=None):
        self.k = k
        self.m = m
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.centers = None
        self.U = None
        self.fitness_score = None

    def initialize_membership(self, n_samples):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        U = np.random.rand(n_samples, self.k)
        return U / np.sum(U, axis=1, keepdims=True)

    def update_centers(self, X):
        um = self.U ** self.m
        return (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)

    def update_membership(self, X):
        dist = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        dist = np.fmax(dist, 1e-10)

        power = 2 / (self.m - 1)
        inv_dist = dist ** (-power)

        return inv_dist / np.sum(inv_dist, axis=1, keepdims=True)

    def fit(self, X):

        X = np.asarray(X)
        n_samples = X.shape[0]

        self.U = self.initialize_membership(n_samples)

        for i in range(self.max_iters):
            old_U = self.U.copy()

            self.centers = self.update_centers(X)
            self.U = self.update_membership(X)

            if np.linalg.norm(self.U - old_U) < self.tol:
                print(f"Converged at iteration {i}")
                break

        # ===== HARD LABELS FOR COMPARISON =====
        labels = np.argmax(self.U, axis=1)

        # ===== SHARED FITNESS (SILHOUETTE) =====
        unique_labels = np.unique(labels)

        if len(unique_labels) > 1:
            fitness = silhouette_score(X, labels)
        else:
            fitness = -1

        self.fitness_score = fitness

        return labels, fitness

    def plot_membership_intensity(self, X):

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        centers_pca = pca.transform(self.centers)

        labels = np.argmax(self.U, axis=1)

        plt.figure(figsize=(8, 6))

        colors = plt.cm.get_cmap("tab10", self.k)

        for j in range(self.k):

            cluster_points = X_pca

            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                color=colors(j),
                alpha=self.U[:, j],
                s=30,
                label=f"Cluster {j}"
            )

        plt.scatter(
            centers_pca[:, 0],
            centers_pca[:, 1],
            s=250,
            c='red',
            marker='X',
            edgecolors='black',
            label='Cluster Centers'
        )

        plt.title(
            f"Fuzzy C-Means "
            f"(Silhouette = {self.fitness_score:.4f})"
        )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.text(
            0.02, 0.02,
            "Color = Cluster Identity\nTransparency = Membership Strength",
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7)
        )

        plt.show()