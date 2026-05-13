import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class HarmonySearchClustering:
    def __init__(self, k, hms=20, hmcr=0.9, par=0.3, bw=0.01, random_state=None, max_iter=200, sample_size=2000):
        self.k = k
        self.hms = hms
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.max_iter = max_iter
        self.random_state = random_state
        self.sample_size = sample_size
        self.history = []

    def assign_clusters(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fitness_fn(self, X, centroids):
        X = np.asarray(X, dtype=np.float64)

        if len(X) > self.sample_size:
            idx = np.random.choice(len(X), self.sample_size, replace=False)
            X_eval, labels_eval = X[idx], self.assign_clusters(X[idx], centroids)
        else:
            X_eval, labels_eval = X, self.assign_clusters(X, centroids)

        # Check AFTER subsampling, on the actual data passed to silhouette_score
        if len(np.unique(labels_eval)) < 2:
            return -0.5

        return silhouette_score(X_eval, labels_eval)

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        self.history = []
        np.random.seed(self.random_state)
        n_feat = X.shape[1]

        lb = X.min(axis=0)          # per-feature bounds
        ub = X.max(axis=0)
        bw = (ub - lb) * 0.01       # adaptive bandwidth

        hm = np.array([
            np.random.uniform(lb, ub, (self.k, n_feat))
            for _ in range(self.hms)
        ])
        hm_fit = np.array([self.fitness_fn(X, h) for h in hm])

        for _ in range(self.max_iter):
            new_h = np.zeros((self.k, n_feat))
            for i in range(self.k):
                for j in range(n_feat):
                    if np.random.rand() < self.hmcr:
                        new_h[i, j] = hm[np.random.randint(self.hms), i, j]
                        if np.random.rand() < self.par:
                            new_h[i, j] += bw[j] * (np.random.rand() - 0.5)  # per-feature bw
                    else:
                        new_h[i, j] = np.random.uniform(lb[j], ub[j])         # per-feature bounds

            f_new = self.fitness_fn(X, new_h)
            worst = np.argmin(hm_fit)
            if f_new > hm_fit[worst]:
                hm[worst], hm_fit[worst] = new_h, f_new

            self.history.append(np.max(hm_fit))

        self.best_centroids = hm[np.argmax(hm_fit)]
        return self

    def plot_history(self):
        plt.plot(self.history)
        plt.xlabel("Iteration")
        plt.ylabel("Best Silhouette Score")
        plt.title("Harmony Search Convergence")
        plt.show()

    def plot_clusters(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        labels = self.assign_clusters(X, self.best_centroids)

        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(X)
            centroids_plot = pca.transform(self.best_centroids)
        else:
            X_plot = X
            centroids_plot = self.best_centroids

        plt.figure(figsize=(8, 6))
        for i in range(self.k):
            cluster_points = X_plot[labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")

        plt.scatter(
            centroids_plot[:, 0], centroids_plot[:, 1],
            s=300, c='red', marker='X', label='Centroids'
        )

        plt.title("Harmony Search Clustering")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True)
        plt.show()


def grid_search(
    X,
    k_values=range(2, 9),
    hmcr_values=(0.8, 0.9),
    par_values=(0.1, 0.3),
    bw_values=(0.05, 0.1),
    hms_values=(20, 50),
    max_iter=200,
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
        for hmcr in hmcr_values:
            for par in par_values:
                for bw in bw_values:
                    for hms in hms_values:
                        for run in range(n_runs):
                            seed = None if random_state is None else random_state + run
                            model = HarmonySearchClustering(
                                k=k,
                                hmcr=hmcr,
                                par=par,
                                bw=bw,
                                hms=hms,
                                max_iter=max_iter,
                                random_state=seed,
                            )
                            model.fit(X_arr)
                            labels = model.assign_clusters(X_arr, model.best_centroids)
                            score = model.history[-1]

                            records.append(dict(
                                k=k, hmcr=hmcr, par=par, bw=bw,
                                hms=hms, run=run, silhouette_score=score,
                            ))

                            if score > best_score:
                                best_score = score
                                best_model = model
                                best_labels = labels

    results = pd.DataFrame(records)

    if verbose:
        summary = (
            results
            .groupby(["k", "hmcr", "par", "bw", "hms"])["silhouette_score"]
            .max()
            .reset_index()
            .sort_values("silhouette_score", ascending=False)
        )
        print("\n=== Grid Search Results (best run per combination) ===")
        print(summary.to_string(index=False))
        print(
            f"\nBest: k={best_model.k}, hmcr={best_model.hmcr}, "
            f"par={best_model.par}, bw={best_model.bw}, hms={best_model.hms}"
            f"  →  silhouette={best_score:.4f}"
        )

    if plot:
        plt.figure(figsize=(8, 4))
        colors = plt.cm.get_cmap("tab10", len(list(par_values)))

        for idx, par in enumerate(par_values):
            best_per_k = (
                results[results["par"] == par]
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
                label=f"par={par}",
            )

        plt.axvline(best_model.k, color="red", linestyle="--", alpha=0.7, label=f"Best k={best_model.k}")
        plt.title("Grid Search: Silhouette Score vs k")
        plt.xlabel("k (number of clusters)")
        plt.ylabel("Best Silhouette Score")
        plt.xticks(list(k_values))
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return best_model, best_labels, results