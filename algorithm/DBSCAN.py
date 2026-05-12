import numpy as np
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

    @staticmethod
    def grid_search(X, eps_values, min_samples_values,
                    score_fn=None, verbose=True, plot_best=True):

        X = np.array(X, dtype=float)

        # ===== unified silhouette scoring =====
        if score_fn is None:
            def score_fn(X, labels):

                mask = labels != -1
                X_f = X[mask]
                labels_f = labels[mask]

                if len(np.unique(labels_f)) < 2:
                    return -1.0

                return silhouette_score(X_f, labels_f)

        results = []

        for eps, min_s in product(eps_values, min_samples_values):

            model = DBSCAN(eps=eps, min_samples=min_s)
            labels = model.fit(X)

            score = score_fn(X, labels)

            results.append({
                "eps": eps,
                "min_samples": min_s,
                "n_clusters": model.n_clusters_,
                "noise": int(np.sum(labels == -1)),
                "score": score,
                "model": model,
                "labels": labels
            })

            if verbose:
                print(
                    f"eps={eps:.3f}  min_samples={min_s:>3d}  "
                    f"clusters={model.n_clusters_:>2d}  "
                    f"noise={np.sum(labels==-1):>4d}  "
                    f"score={score:.4f}"
                )

        results.sort(key=lambda r: r["score"], reverse=True)
        best = results[0]

        print(
            f"\n★ Best → eps={best['eps']} "
            f"min_samples={best['min_samples']} "
            f"score={best['score']:.4f}"
        )

        if plot_best:
            best["model"].plot_clusters(
                X,
                best["labels"],
                title_suffix=f"eps={best['eps']}, min_samples={best['min_samples']}"
            )

        return results, best["model"]