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


def grid_search(
    X,
    k_values=range(2, 9),
    tau_values=(1.2, 1.5, 2.0),
    perturbation_scale_values=(0.1, 0.3, 0.5),
    max_iters_values=(300,),
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
        for tau in tau_values:
            for perturbation_scale in perturbation_scale_values:
                for max_iters in max_iters_values:
                    for run in range(n_runs):
                        seed = (
                            None
                            if random_state is None
                            else random_state + run
                        )
                        model = ExtremalOptimization(
                            k=k,
                            max_iters=max_iters,
                            tau=tau,
                            perturbation_scale=perturbation_scale,
                            random_state=seed,
                        )
                        labels, score = model.fit(X_arr)
                        records.append(
                            dict(
                                k=k,
                                tau=tau,
                                perturbation_scale=perturbation_scale,
                                max_iters=max_iters,
                                run=run,
                                silhouette_score=score,
                            )
                        )

                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_labels = labels

    results = pd.DataFrame(records)

    if verbose:
        summary = (
            results
            .groupby(["k", "tau", "perturbation_scale", "max_iters"])["silhouette_score"]
            .max()
            .reset_index()
            .sort_values("silhouette_score", ascending=False)
        )
        print("\n=== Grid Search Results (best run per combination) ===")
        print(summary.to_string(index=False))
        print(
            f"\nBest: k={best_model.k}, tau={best_model.tau}, "
            f"perturbation_scale={best_model.perturbation_scale}, "
            f"max_iters={best_model.max_iters}"
            f"  →  silhouette={best_score:.4f}"
        )

    if plot:
        plt.figure(figsize=(8, 4))
        colors = plt.cm.get_cmap("tab10", len(list(tau_values)))

        for idx, tau_val in enumerate(tau_values):
            best_per_k = (
                results[results["tau"] == tau_val]
                .groupby("k")["silhouette_score"]
                .max()
                .reset_index()
            )
            plt.plot(
                best_per_k["k"],
                best_per_k["silhouette_score"],
                marker="o",
                linewidth=2,
                color=colors(idx),
                label=f"tau={tau_val}",
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