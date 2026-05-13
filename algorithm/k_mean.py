import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.random_state = random_state
        self.fitness = None

    def fit(self, X):

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # ===== Initialize Centroids =====
        random_indices = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_indices]

        # ===== Main Loop =====
        for i in range(self.max_iters):

            # Compute distances
            distances = np.linalg.norm(
                X[:, np.newaxis] - self.centroids,
                axis=2
            )

            # Assign clusters
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([
                X[labels == j].mean(axis=0)
                if len(X[labels == j]) > 0
                else self.centroids[j]
                for j in range(self.k)
            ])

            # Check convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                print(f"Converged at iteration {i}")
                break

            self.centroids = new_centroids

        # ===== Shared Fitness Function =====
        unique_labels = np.unique(labels)

        # Silhouette requires at least 2 clusters
        if len(unique_labels) > 1:
            fitness = silhouette_score(X, labels)
        else:
            fitness = -1

        self.fitness = fitness

        return labels, fitness

    def plot_clusters(self, X, labels):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        centroids_pca = pca.transform(self.centroids)

        plt.figure(figsize=(8, 6))

        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap("viridis", len(unique_labels))

        for i, label in enumerate(unique_labels):

            cluster_data = X_pca[labels == label]

            plt.scatter(
                cluster_data[:, 0],
                cluster_data[:, 1],
                color=colors(i),
                label=f'Cluster {label}',
                alpha=0.6
            )

        plt.scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            s=200,
            c='red',
            marker='X',
            edgecolors='black',
            label='Centroids'
        )

        plt.title(
            f"K-Means Results "
            f"(Silhouette Score = {self.fitness:.4f})"
        )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        plt.legend()

        plt.grid(True, linestyle='--', alpha=0.5)

        plt.show()


def grid_search(
    X,
    k_values=range(2, 9),
    max_iters_values=(100,),
    tol_values=(1e-4,),
    n_runs=3,
    random_state=None,
    verbose=True,
    plot=True,
):
    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy()
    else:
        X_arr = np.asarray(X)

    records = []
    best_score = -np.inf
    best_model = None
    best_labels = None

    for k in k_values:
        for max_iters in max_iters_values:
            for tol in tol_values:
                for run in range(n_runs):
                    seed = (
                        None
                        if random_state is None
                        else random_state + run
                    )
                    model = KMeans(
                        k=k,
                        max_iters=max_iters,
                        tol=tol,
                        random_state=seed,
                    )
                    labels, score = model.fit(X_arr)
                    records.append(
                        dict(k=k, max_iters=max_iters, tol=tol,
                            run=run, silhouette_score=score)
                    )

                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_labels = labels

    results = pd.DataFrame(records)

    if verbose:
        summary = (
            results
            .groupby(["k", "max_iters", "tol"])["silhouette_score"]
            .max()
            .reset_index()
            .sort_values("silhouette_score", ascending=False)
        )
        print("\n=== Grid Search Results (best run per combination) ===")
        print(summary.to_string(index=False))
        print(
            f"\nBest: k={best_model.k}, max_iters={best_model.max_iters}, "
            f"tol={best_model.tol}  →  silhouette={best_score:.4f}"
        )

    if plot:
        best_per_k = (
            results
            .groupby("k")["silhouette_score"]
            .max()
            .reset_index()
        )
        plt.figure(figsize=(8, 4))
        plt.plot(
            best_per_k["k"],
            best_per_k["silhouette_score"],
            marker="o",
            linewidth=2,
            color="steelblue",
        )
        plt.axvline(
            best_model.k,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Best k={best_model.k}",
        )
        plt.title("Grid Search: Silhouette Score vs k")
        plt.xlabel("k (number of clusters)")
        plt.ylabel("Best Silhouette Score")
        plt.xticks(list(k_values))
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return best_model, best_labels, results