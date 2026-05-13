import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class PSOClustering:

    def __init__(self, k=4, n_particles=20, max_iters=100,
                w=0.7, c1=1.5, c2=1.5, random_state=None, sample_size=1000):

        self.k = k
        self.n_particles = n_particles
        self.max_iters = max_iters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.random_state = random_state
        self.sample_size = sample_size

        self.centroids = None
        self.history = []
        self.final_fitness = None

    def assign_clusters(self, X, centroids):

        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)
def fitness_fn(self, X, centroids):
    X = np.asarray(X, dtype=np.float64)

    if len(X) > self.sample_size:
        idx = np.random.choice(len(X), self.sample_size, replace=False)
        X_eval = X[idx]
        labels = self.assign_clusters(X_eval, centroids)
    else:
        X_eval = X
        labels = self.assign_clusters(X, centroids)

    if len(np.unique(labels)) < 2:
        return -0.5

    return silhouette_score(X_eval, labels)

    def fit(self, X):

        X = np.asarray(X, dtype=np.float64)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        data_min = np.min(X, axis=0)
        data_max = np.max(X, axis=0)

        particles = np.random.uniform(
            data_min,
            data_max,
            size=(self.n_particles, self.k, n_features)
        )

        velocities = np.zeros_like(particles)

        pbest = particles.copy()

        pbest_scores = np.array([
            self.fitness_fn(X, p) for p in particles
        ])

        gbest = pbest[np.argmax(pbest_scores)].copy()

        self.history = []

        v_max = (data_max - data_min) * 0.1

        for t in range(self.max_iters):

            w = self.w * (1 - t / self.max_iters)

            for i in range(self.n_particles):

                score = self.fitness_fn(X, particles[i])

                if score > pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = particles[i].copy()

            gbest = pbest[np.argmax(pbest_scores)].copy()
            best_fitness = np.max(pbest_scores)

            self.history.append(best_fitness)

            for i in range(self.n_particles):

                r1 = np.random.rand(self.k, n_features)
                r2 = np.random.rand(self.k, n_features)

                velocities[i] = (
                    w * velocities[i]
                    + self.c1 * r1 * (pbest[i] - particles[i])
                    + self.c2 * r2 * (gbest - particles[i])
                )

                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], data_min, data_max)

        self.centroids = gbest
        self.final_fitness = self.fitness_fn(X, self.centroids)

        print(f"PSO finished — silhouette: {self.final_fitness:.4f}")

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.assign_clusters(X, self.centroids)

    def plot_clusters(self, X, labels):

        X = np.asarray(X, dtype=np.float64)

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
            edgecolors='black'
        )

        plt.title(f"PSO Clustering (Silhouette = {self.final_fitness:.4f})")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.show()

    def plot_convergence(self):

        plt.figure(figsize=(8, 5))
        plt.plot(self.history, linewidth=2)
        plt.title("PSO Convergence")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()


def grid_search(
    X,
    k_values=range(2, 9),
    w_values=(0.5, 0.7, 0.9),
    c1_values=(1.5, 2.0),
    c2_values=(1.5, 2.0),
    n_particles_values=(10, 20, 30),
    max_iters=100,
    n_runs=3,
    random_state=None,
    verbose=True,
    plot=True,
):

    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy()
    else:
        X_arr = np.asarray(X, dtype=np.float64)

    records = []
    best_score = -np.inf
    best_model = None
    best_labels = None

    for k in k_values:
        for w in w_values:
            for c1 in c1_values:
                for c2 in c2_values:
                    for n_particles in n_particles_values:
                        for run in range(n_runs):
                            seed = (
                                None
                                if random_state is None
                                else random_state + run
                            )
                            model = PSOClustering(
                                k=k,
                                n_particles=n_particles,
                                max_iters=max_iters,
                                w=w,
                                c1=c1,
                                c2=c2,
                                random_state=seed,
                            )
                            model.fit(X_arr)
                            score = model.final_fitness
                            labels = model.predict(X_arr)

                            records.append(
                                dict(
                                    k=k,
                                    w=w,
                                    c1=c1,
                                    c2=c2,
                                    n_particles=n_particles,
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
            .groupby(["k", "w", "c1", "c2", "n_particles"])["silhouette_score"]
            .max()
            .reset_index()
            .sort_values("silhouette_score", ascending=False)
        )
        print("\n=== Grid Search Results (best run per combination) ===")
        print(summary.to_string(index=False))
        print(
            f"\nBest: k={best_model.k}, w={best_model.w}, "
            f"c1={best_model.c1}, c2={best_model.c2}, "
            f"n_particles={best_model.n_particles}"
            f"  →  silhouette={best_score:.4f}"
        )

    if plot:
        plt.figure(figsize=(8, 4))
        colors = plt.cm.get_cmap("tab10", len(list(w_values)))

        for idx, w_val in enumerate(w_values):
            best_per_k = (
                results[results["w"] == w_val]
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
                label=f"w={w_val}",
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