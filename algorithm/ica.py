import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class ImperialistCompetitiveClustering:
    def __init__(
        self,
        k,
        n_countries=50,
        n_imperialists=5,
        max_iter=200,
        assimilation_coef=1.5,
        revolution_rate=0.1,
        beta=2.0,
        random_state=None,
        sample_size=500,
    ):
        self.k = k
        self.n_countries = n_countries
        self.n_imperialists = n_imperialists
        self.max_iter = max_iter
        self.assimilation_coef = assimilation_coef
        self.revolution_rate = revolution_rate
        self.beta = beta
        self.random_state = random_state
        self.sample_size = sample_size
        self.history = []

        self.best_centroids = None
        self.best_fitness = -np.inf

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

    def initialize_countries(self, X):
        n_feat = X.shape[1]
        lb, ub = X.min(), X.max()
        countries = np.random.uniform(lb, ub, (self.n_countries, self.k, n_feat))
        fitness = np.array([self.fitness_fn(X, c) for c in countries])
        return countries, fitness

    def form_empires(self, countries, fitness):
        sorted_idx = np.argsort(fitness)[::-1]
        imperialists = countries[sorted_idx[: self.n_imperialists]]
        imperialist_fit = fitness[sorted_idx[: self.n_imperialists]]

        colonies = countries[sorted_idx[self.n_imperialists :]]
        colonies_fit = fitness[sorted_idx[self.n_imperialists :]]

        power = imperialist_fit - np.min(imperialist_fit) + 1e-9
        n_colonies = np.round((power / power.sum()) * len(colonies)).astype(int)

        empires = []
        start = 0
        for i, n in enumerate(n_colonies):
            assigned = colonies[start : start + n] if n > 0 else np.empty((0,) + colonies.shape[1:])
            assigned_fit = colonies_fit[start : start + n] if n > 0 else np.array([])
            empires.append(
                {
                    "imperialist": imperialists[i],
                    "imperialist_fit": imperialist_fit[i],
                    "colonies": assigned,
                    "colonies_fit": assigned_fit,
                }
            )
            start += n

        return empires

    def assimilation(self, empire):
        imperialist = empire["imperialist"]
        colonies = empire["colonies"]

        if len(colonies) == 0:
            return empire

        direction = imperialist - colonies
        rand_coeff = np.random.rand(*colonies.shape) * self.assimilation_coef
        colonies = colonies + rand_coeff * direction

        empire["colonies"] = colonies
        return empire

    def revolution(self, empire, X):
        colonies = empire["colonies"]
        if len(colonies) == 0:
            return empire

        n_revolt = int(self.revolution_rate * len(colonies))
        if n_revolt <= 0:
            return empire

        n_feat = X.shape[1]
        lb, ub = X.min(), X.max()
        idx = np.random.choice(len(colonies), n_revolt, replace=False)
        colonies[idx] = np.random.uniform(lb, ub, (n_revolt, self.k, n_feat))

        empire["colonies"] = colonies
        return empire

    def competition(self, empires):
        if len(empires) <= 1:
            return empires

        total_powers = []
        for e in empires:
            if len(e["colonies_fit"]) > 0:
                total_power = e["imperialist_fit"] + self.beta * np.mean(e["colonies_fit"])
            else:
                total_power = e["imperialist_fit"]
            total_powers.append(total_power)

        weakest_idx = np.argmin(total_powers)
        strongest_idx = np.argmax(total_powers)

        if len(empires[weakest_idx]["colonies"]) > 0:
            w_col_idx = np.argmin(empires[weakest_idx]["colonies_fit"])
            colony = empires[weakest_idx]["colonies"][w_col_idx]
            colony_fit = empires[weakest_idx]["colonies_fit"][w_col_idx]

            empires[weakest_idx]["colonies"] = np.delete(
                empires[weakest_idx]["colonies"], w_col_idx, axis=0
            )
            empires[weakest_idx]["colonies_fit"] = np.delete(
                empires[weakest_idx]["colonies_fit"], w_col_idx
            )

            empires[strongest_idx]["colonies"] = np.append(
                empires[strongest_idx]["colonies"], [colony], axis=0
            )
            empires[strongest_idx]["colonies_fit"] = np.append(
                empires[strongest_idx]["colonies_fit"], colony_fit
            )

        empires = [e for e in empires if len(e["colonies"]) > 0 or len(empires) == 1]
        return empires

    def fit(self, X):
        np.random.seed(self.random_state)
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        countries, fitness = self.initialize_countries(X)
        empires = self.form_empires(countries, fitness)

        global_best_centroids = None
        global_best_fitness = -np.inf

        for _ in range(self.max_iter):
            for empire in empires:
                empire = self.assimilation(empire)
                empire = self.revolution(empire, X)

                if len(empire["colonies"]) > 0:
                    empire["colonies_fit"] = np.array(
                        [self.fitness_fn(X, c) for c in empire["colonies"]]
                    )

                if len(empire["colonies"]) > 0:
                    best_col_idx = np.argmax(empire["colonies_fit"])
                    if empire["colonies_fit"][best_col_idx] > empire["imperialist_fit"]:
                        tmp = empire["imperialist"].copy()
                        tmp_fit = empire["imperialist_fit"]
                        empire["imperialist"] = empire["colonies"][best_col_idx].copy()
                        empire["imperialist_fit"] = empire["colonies_fit"][best_col_idx]
                        empire["colonies"][best_col_idx] = tmp
                        empire["colonies_fit"][best_col_idx] = tmp_fit

            empires = self.competition(empires)

            best_emp = max(empires, key=lambda e: e["imperialist_fit"])

            if best_emp["imperialist_fit"] > global_best_fitness:
                global_best_fitness = best_emp["imperialist_fit"]
                global_best_centroids = best_emp["imperialist"].copy()

            self.history.append(global_best_fitness)

        self.best_centroids = global_best_centroids
        self.best_fitness = global_best_fitness
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        return self.assign_clusters(X, self.best_centroids)

    def plot_history(self):
        plt.plot(self.history)
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness (Silhouette Score)")
        plt.title("Fitness History")
        plt.grid()
        plt.show()

    def plot_clusters(self, X, labels):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        centroids_pca = pca.transform(self.best_centroids)

        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap("viridis", len(unique_labels))

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

        plt.title("Imperialist Competitive Algorithm Clustering")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()


def grid_search(
    X,
    k_values=range(2, 9),
    n_countries_values=(30, 50),
    n_imperialists_values=(3, 5),
    assimilation_coef_values=(1.0, 1.5),
    revolution_rate_values=(0.05, 0.1),
    beta_values=(1.0, 2.0),
    max_iter=50,
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
        for n_countries in n_countries_values:
            for n_imperialists in n_imperialists_values:
                for assimilation_coef in assimilation_coef_values:
                    for revolution_rate in revolution_rate_values:
                        for beta in beta_values:
                            for run in range(n_runs):
                                seed = (
                                    None
                                    if random_state is None
                                    else random_state + run
                                )
                                model = ImperialistCompetitiveClustering(
                                    k=k,
                                    n_countries=n_countries,
                                    n_imperialists=n_imperialists,
                                    assimilation_coef=assimilation_coef,
                                    revolution_rate=revolution_rate,
                                    beta=beta,
                                    max_iter=max_iter,
                                    random_state=seed,
                                )
                                model.fit(X_arr)
                                score = model.best_fitness
                                labels = model.predict(X_arr)

                                records.append(dict(
                                    k=k,
                                    n_countries=n_countries,
                                    n_imperialists=n_imperialists,
                                    assimilation_coef=assimilation_coef,
                                    revolution_rate=revolution_rate,
                                    beta=beta,
                                    run=run,
                                    silhouette_score=score,
                                ))

                                if score > best_score:
                                    best_score = score
                                    best_model = model
                                    best_labels = labels

    results = pd.DataFrame(records)

    if verbose:
        summary = (
            results
            .groupby([
                "k", "n_countries", "n_imperialists",
                "assimilation_coef", "revolution_rate", "beta"
            ])["silhouette_score"]
            .max()
            .reset_index()
            .sort_values("silhouette_score", ascending=False)
        )
        print("\n=== Grid Search Results (best run per combination) ===")
        print(summary.to_string(index=False))
        print(
            f"\nBest: k={best_model.k}, "
            f"n_countries={best_model.n_countries}, "
            f"n_imperialists={best_model.n_imperialists}, "
            f"assimilation_coef={best_model.assimilation_coef}, "
            f"revolution_rate={best_model.revolution_rate}, "
            f"beta={best_model.beta}"
            f"  →  silhouette={best_score:.4f}"
        )

    if plot:
        plt.figure(figsize=(8, 4))
        colors = plt.cm.get_cmap("tab10", len(list(revolution_rate_values)))

        for idx, rr in enumerate(revolution_rate_values):
            best_per_k = (
                results[results["revolution_rate"] == rr]
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
                label=f"revolution_rate={rr}",
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