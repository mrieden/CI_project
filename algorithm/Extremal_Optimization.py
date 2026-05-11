import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class ExtremalOptimization:
    """
    Extremal Optimization (EO) for clustering.

    EO is a nature-inspired metaheuristic that iteratively improves the
    worst-performing element in the current solution. For clustering, each
    centroid is scored by its contribution to WCSS; the worst centroid is
    perturbed, and the move is accepted if it improves global fitness.

    Parameters
    ----------
    k : int
        Number of clusters.
    max_iters : int
        Maximum number of improvement iterations.
    tau : float
        EO "temperature" exponent (τ). Higher values focus more on the
        absolute worst centroid; lower values allow broader exploration.
        Typical range: 1.0–2.5.
    perturbation_scale : float
        Standard deviation of the Gaussian noise applied to perturbed
        centroids, expressed as a fraction of the dataset's std deviation.
    random_state : int or None
        Seed for reproducibility.
    """

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Return the cluster label for every point in X."""
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _wcss(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Compute the total Within-Cluster Sum of Squares (fitness)."""
        total = 0.0
        for j in range(self.k):
            pts = X[labels == j]
            if len(pts) > 0:
                total += np.sum((pts - centroids[j]) ** 2)
        return total

    def _centroid_costs(
        self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """
        Return the individual WCSS contribution of each centroid.
        This is the 'fitness' used to rank elements for EO selection.
        """
        costs = np.zeros(self.k)
        for j in range(self.k):
            pts = X[labels == j]
            if len(pts) > 0:
                costs[j] = np.sum((pts - centroids[j]) ** 2)
        return costs

    def _eo_select_worst(self, costs: np.ndarray) -> int:
        """
        EO selection: pick the centroid to perturb using a power-law
        probability proportional to rank^(-tau).  The worst centroid
        (rank 1 = highest cost) has the highest selection probability.
        """
        # Rank from worst (1) to best (k)
        order = np.argsort(costs)[::-1]          # indices, worst first
        ranks = np.empty(self.k, dtype=float)
        for rank, idx in enumerate(order, start=1):
            ranks[idx] = rank

        weights = ranks ** (-self.tau)
        weights /= weights.sum()
        return int(np.random.choice(self.k, p=weights))

    def _perturb(self, centroid: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Replace the selected centroid with a randomly chosen data point,
        then add a small Gaussian jitter.  Using a real data point ensures
        the new candidate stays inside the data manifold.
        """
        base = X[np.random.randint(len(X))]
        noise = np.random.normal(
            scale=self.perturbation_scale * X.std(axis=0), size=centroid.shape
        )
        return base + noise

    # ------------------------------------------------------------------
    # Public API  (mirrors KMeans)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray):
        """
        Run Extremal Optimization to find cluster centroids.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
        fitness : float  — final WCSS (lower is better)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # ---- Initialisation: random distinct data points ----
        init_idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[init_idx].copy().astype(float)

        labels = self._assign(X, self.centroids)
        best_centroids = self.centroids.copy()
        best_fitness = self._wcss(X, labels, self.centroids)

        for i in range(self.max_iters):
            costs = self._centroid_costs(X, labels, self.centroids)

            # ---- Select worst centroid (probabilistic via power-law) ----
            victim = self._eo_select_worst(costs)

            # ---- Propose a perturbation ----
            candidate_centroids = self.centroids.copy()
            candidate_centroids[victim] = self._perturb(
                self.centroids[victim], X
            )

            # ---- Re-assign and evaluate ----
            candidate_labels = self._assign(X, candidate_centroids)
            candidate_fitness = self._wcss(X, candidate_labels, candidate_centroids)

            # ---- Accept only improvements (greedy EO) ----
            if candidate_fitness < best_fitness:
                best_fitness = candidate_fitness
                best_centroids = candidate_centroids.copy()

            # Always move to the candidate for the next iteration's ranking
            # (allows temporary worsening → broader exploration)
            self.centroids = candidate_centroids
            labels = candidate_labels

        # ---- Restore best solution found ----
        self.centroids = best_centroids
        labels = self._assign(X, self.centroids)
        fitness = self._wcss(X, labels, self.centroids)

        print(f"EO finished — best WCSS: {fitness:.4f}")
        return labels, fitness

    def plot_clusters(self, X: np.ndarray, labels: np.ndarray):
        """PCA-reduced 2-D scatter plot (identical interface to KMeans)."""
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        centroids_pca = pca.transform(self.centroids)

        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap("plasma", len(unique_labels))

        for i, label in enumerate(unique_labels):
            cluster_data = X_pca[labels == label]
            plt.scatter(
                cluster_data[:, 0], cluster_data[:, 1],
                color=colors(i), label=f"Cluster {label}", alpha=0.6,
            )

        plt.scatter(
            centroids_pca[:, 0], centroids_pca[:, 1],
            s=200, c="red", marker="X", edgecolors="black", label="Centroids",
        )

        plt.title("Extremal Optimization Results (PCA-reduced to 2D)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()