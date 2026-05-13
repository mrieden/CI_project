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
        self.bw = bw  # used as scale factor in grid_search; fit() derives bw_vec internally
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
            X_eval = X[idx]
            labels_eval = self.assign_clusters(X_eval, centroids)
        else:
            X_eval = X
            labels_eval = self.assign_clusters(X, centroids)

        active_clusters = len(np.unique(labels_eval))

        if active_clusters < 2:
            return -5

        score = silhouette_score(X_eval, labels_eval)

        # Penalize for every missing cluster
        missing = self.k - active_clusters
        if missing > 0:
            penalty = missing * (1 / self.k)  # e.g. k=4, 1 missing → -0.25
            score -= penalty

        return score

    # ------------------------------------------------------------------ #
    #  NEW: refine centroids with a few K-Means steps after HS search     #
    # ------------------------------------------------------------------ #
    def _kmeans_refine(self, X, centroids, n_iter=10):
        c = centroids.copy()
        for _ in range(n_iter):
            labels = self.assign_clusters(X, c)
            new_c = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i) else c[i]
                for i in range(self.k)
            ])
            if np.allclose(c, new_c):
                break
            c = new_c
        return c

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        self.history = []
        np.random.seed(self.random_state)
        n_feat = X.shape[1]

        lb = X.min(axis=0)
        ub = X.max(axis=0)
        # FIX 2: use self.bw as the scale factor (default 0.01)
        bw_vec = (ub - lb) * self.bw

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
                            new_h[i, j] += bw_vec[j] * (np.random.rand() - 0.5)
                        # FIX 1: clamp after perturbation to keep centroids in bounds
                        new_h[i, j] = np.clip(new_h[i, j], lb[j], ub[j])
                    else:
                        new_h[i, j] = np.random.uniform(lb[j], ub[j])

            f_new = self.fitness_fn(X, new_h)
            worst = np.argmin(hm_fit)
            if f_new > hm_fit[worst]:
                hm[worst] = new_h
                hm_fit[worst] = f_new

            self.history.append(float(np.max(hm_fit)))

        # FIX 3: pick best harmony and refine with K-Means polish
        best_raw = hm[np.argmax(hm_fit)]
        self.best_centroids = self._kmeans_refine(X, best_raw)
        self.best_score_ = self.fitness_fn(X, self.best_centroids)
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

    def grid_search(self, X, param_grid=None):
        best_score = -np.inf
        best_params = None

        for hmcr in param_grid['hmcr']:
            for par in param_grid['par']:
                for bw in param_grid['bw']:
                    for hms in param_grid['hms']:
                        for k in param_grid['k']:
                            self.hmcr = hmcr
                            self.par = par
                            self.bw = bw   # fit() will use this as the scale factor
                            self.hms = hms
                            self.k = k

                            self.fit(X)
                            # FIX 4: use best_score_ (post-refinement) instead of last history value
                            score = self.best_score_
                            print(f"Tested params: HMCR={hmcr}, PAR={par}, BW={bw}, HMS={hms}, K={k} => Silhouette Score: {score:.4f}")

                            if score > best_score:
                                best_score = score
                                best_params = {'hmcr': hmcr, 'par': par, 'bw': bw, 'hms': hms, 'k': k}

        print(f"\nBest params: {best_params} with Silhouette Score: {best_score:.4f}")
        return best_params, best_score