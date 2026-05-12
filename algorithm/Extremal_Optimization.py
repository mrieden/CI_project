import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class ExtremalOptimization:

    def __init__(
        self,
        k: int = 3,
        max_iters: int = 300,
        tau: float = 1.5,
        perturbation_scale: float = 0.3,
        random_state=None,
    ):
        self.k = k
        self.max_iters = max_iters
        self.tau = tau
        self.perturbation_scale = perturbation_scale
        self.random_state = random_state

        self.centroids = None
        self.fitness = None

    # -------------------------
    # Assignment
    # -------------------------
    def _assign(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    # -------------------------
    # Internal EO cost (KEPT as WCSS for search pressure)
    # -------------------------
    def _centroid_costs(self, X, labels, centroids):
        costs = np.zeros(self.k)

        for j in range(self.k):
            pts = X[labels == j]
            if len(pts) > 0:
                costs[j] = np.sum((pts - centroids[j]) ** 2)

        return costs

    # -------------------------
    # EO selection
    # -------------------------
    def _eo_select_worst(self, costs):
        order = np.argsort(costs)[::-1]  # worst first

        ranks = np.empty(self.k, dtype=float)
        for r, idx in enumerate(order, start=1):
            ranks[idx] = r

        weights = ranks ** (-self.tau)
        weights /= weights.sum()

        return int(np.random.choice(self.k, p=weights))

    # -------------------------
    # perturbation
    # -------------------------
    def _perturb(self, centroid, X):
        base = X[np.random.randint(len(X))]
        noise = np.random.normal(
            scale=self.perturbation_scale * X.std(axis=0),
            size=centroid.shape
        )
        return base + noise

    # -------------------------
    # MAIN FIT
    # -------------------------
    def fit(self, X):

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # init centroids
        init_idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[init_idx].copy()

        labels = self._assign(X, self.centroids)

        best_centroids = self.centroids.copy()

        # dummy initial silhouette
        unique_labels = np.unique(labels)
        best_fitness = -1

        if len(unique_labels) > 1:
            best_fitness = silhouette_score(X, labels)

        for i in range(self.max_iters):

            costs = self._centroid_costs(X, labels, self.centroids)

            victim = self._eo_select_worst(costs)

            candidate_centroids = self.centroids.copy()
            candidate_centroids[victim] = self._perturb(
                self.centroids[victim], X
            )

            candidate_labels = self._assign(X, candidate_centroids)

            # ===== SILHOUETTE FITNESS =====
            unique = np.unique(candidate_labels)

            if len(unique) > 1:
                candidate_fitness = silhouette_score(X, candidate_labels)
            else:
                candidate_fitness = -1

            # keep best
            if candidate_fitness > best_fitness:
                best_fitness = candidate_fitness
                best_centroids = candidate_centroids.copy()

            # exploration step (EO behavior preserved)
            self.centroids = candidate_centroids
            labels = candidate_labels

        # final solution
        self.centroids = best_centroids
        labels = self._assign(X, self.centroids)

        unique = np.unique(labels)

        if len(unique) > 1:
            fitness = silhouette_score(X, labels)
        else:
            fitness = -1

        self.fitness = fitness

        print(f"EO finished — silhouette: {fitness:.4f}")

        return labels, fitness

    # -------------------------
    # PLOT
    # -------------------------
    def plot_clusters(self, X, labels):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        centroids_pca = pca.transform(self.centroids)

        plt.figure(figsize=(8, 6))

        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap("plasma", len(unique_labels))

        for i, label in enumerate(unique_labels):
            cluster_data = X_pca[labels == label]

            plt.scatter(
                cluster_data[:, 0],
                cluster_data[:, 1],
                color=colors(i),
                label=f"Cluster {label}",
                alpha=0.6,
            )

        plt.scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            s=200,
            c="red",
            marker="X",
            edgecolors="black",
            label="Centroids",
        )

        plt.title(
            f"Extremal Optimization (Silhouette = {self.fitness:.4f})"
        )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.show()


    def grid_search(self, X):

        param_grid = {
            'k': [3, 4],
            'max_iters': [100],
            'tau': [1.2, 1.5, 2.0],
            'perturbation_scale': [0.1, 0.3, 0.5]
        }

        best_score = -np.inf
        best_params = None

        for k in param_grid['k']:

            for max_iters in param_grid['max_iters']:

                for tau in param_grid['tau']:

                    for perturbation_scale in param_grid['perturbation_scale']:

                        # update hyperparameters
                        self.k = k
                        self.max_iters = max_iters
                        self.tau = tau
                        self.perturbation_scale = perturbation_scale

                        # run EO
                        labels, score = self.fit(X)

                        print(
                            f"k={k}, "
                            f"iters={max_iters}, "
                            f"tau={tau}, "
                            f"perturb={perturbation_scale} "
                            f"-> silhouette={score:.4f}"
                        )

                        # track best
                        if score > best_score:
                            best_score = score
                            best_params = {
                                "k": k,
                                "max_iters": max_iters,
                                "tau": tau,
                                "perturbation_scale": perturbation_scale
                            }

        print("\nBest Parameters:")
        print(best_params)

        print(f"Best Silhouette Score: {best_score:.4f}")

        return best_params, best_score

