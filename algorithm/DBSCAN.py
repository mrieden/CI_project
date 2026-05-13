import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from itertools import product


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.n_clusters_ = None
        self.fitness = None

    def fit(self, X):

        X = np.array(X, dtype=float)
        n = len(X)

        labels = np.full(n, -1)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        def region_query(idx):
            dists = np.linalg.norm(X - X[idx], axis=1)
            return np.where(dists <= self.eps)[0]

        def expand_cluster(idx, neighbors, cid):
            labels[idx] = cid
            i = 0

            while i < len(neighbors):
                pt = neighbors[i]

                if not visited[pt]:
                    visited[pt] = True
                    new_neighbors = region_query(pt)

                    if len(new_neighbors) >= self.min_samples:
                        neighbors = np.union1d(neighbors, new_neighbors)

                if labels[pt] == -1:
                    labels[pt] = cid

                i += 1

        for idx in range(n):

            if visited[idx]:
                continue

            visited[idx] = True
            neighbors = region_query(idx)

            if len(neighbors) < self.min_samples:
                labels[idx] = -1
            else:
                expand_cluster(idx, neighbors, cluster_id)
                cluster_id += 1

        self.labels_ = labels
        self.n_clusters_ = cluster_id

        # ===== SILHOUETTE FITNESS =====
        mask = labels != -1
        X_filtered = X[mask]
        labels_filtered = labels[mask]

        unique_labels = np.unique(labels_filtered)

        if len(unique_labels) > 1:
            fitness = silhouette_score(X_filtered, labels_filtered)
        else:
            fitness = -1

        self.fitness = fitness

        print(
            f"Found {self.n_clusters_} cluster(s), "
            f"{np.sum(labels == -1)} noise point(s), "
            f"silhouette={fitness:.4f}"
        )

        return labels

    def plot_clusters(self, X, labels=None, title_suffix=""):

        if labels is None:
            labels = self.labels_

        X = np.array(X, dtype=float)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap("viridis", len(unique_labels))

        plt.figure(figsize=(8, 6))

        for i, label in enumerate(unique_labels):

            mask = labels == label

            style = dict(
                alpha=0.6,
                color=colors(i),
                label=f"Cluster {label}"
            )

            if label == -1:
                style.update(
                    color="grey",
                    marker="x",
                    label="Noise",
                    alpha=0.4
                )

            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                **style
            )

        plt.title(
            f"DBSCAN Results (Silhouette = {self.fitness:.4f})"
            + (f" — {title_suffix}" if title_suffix else "")
        )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.show()


def grid_search(
    X,
    eps_values=(0.3, 0.5, 0.7, 1.0),
    min_samples_values=(3, 5, 10),
    verbose=True,
    plot=True,
):
    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy()
    else:
        X_arr = np.array(X, dtype=float)

    eps_list = list(eps_values)
    min_samples_list = list(min_samples_values)

    records = []
    best_score = -np.inf
    best_model = None
    best_labels = None

    for eps, min_s in product(eps_list, min_samples_list):
        model = DBSCAN(eps=eps, min_samples=min_s)
        labels = model.fit(X_arr)
        score = model.fitness

        records.append(
            dict(
                eps=eps,
                min_samples=min_s,
                n_clusters=model.n_clusters_,
                noise_points=int(np.sum(labels == -1)),
                silhouette_score=score,
            )
        )

        if verbose:
            print(
                f"eps={eps:.3f}  min_samples={min_s:>3d}  "
                f"clusters={model.n_clusters_:>2d}  "
                f"noise={np.sum(labels == -1):>4d}  "
                f"silhouette={score:.4f}"
            )

        if score > best_score:
            best_score = score
            best_model = model
            best_labels = labels

    results = pd.DataFrame(records)

    if verbose:
        print(
            f"\nBest: eps={best_model.eps}, "
            f"min_samples={best_model.min_samples}"
            f"  →  silhouette={best_score:.4f}"
        )

    if plot:
        # ── heatmap of silhouette scores ──────────────────────────────
        score_grid = (
            results
            .pivot(index="min_samples", columns="eps", values="silhouette_score")
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(score_grid.values, aspect="auto", cmap="YlGnBu")

        ax.set_xticks(range(len(eps_list)))
        ax.set_xticklabels([f"{e:.3g}" for e in score_grid.columns])
        ax.set_yticks(range(len(min_samples_list)))
        ax.set_yticklabels(score_grid.index)
        ax.set_xlabel("eps")
        ax.set_ylabel("min_samples")
        ax.set_title("Grid Search: Silhouette Score Heatmap")

        for r in range(score_grid.shape[0]):
            for c in range(score_grid.shape[1]):
                val = score_grid.values[r, c]
                ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="black")

        plt.colorbar(im, ax=ax, label="Silhouette Score")
        plt.tight_layout()
        plt.show()

        # ── best cluster plot ─────────────────────────────────────────
        best_model.plot_clusters(
            X_arr,
            best_labels,
            title_suffix=f"eps={best_model.eps}, min_samples={best_model.min_samples}",
        )

    return best_model, best_labels, results