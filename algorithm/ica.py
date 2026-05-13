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
        self.best_fitness = -np.inf  # silhouette is maximised

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def assign_clusters(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fitness_fn(self, X, centroids):
        labels = self.assign_clusters(X, centroids)
        X = np.asarray(X, dtype=np.float64)

        # FIX 1: harsher penalty for degenerate solutions
        if len(np.unique(labels)) < 2:
            return -1.0

        if len(X) > self.sample_size:
            idx = np.random.choice(len(X), self.sample_size, replace=False)
            sampled_labels = labels[idx]
            if len(np.unique(sampled_labels)) < 2:
                return -1.0
            return silhouette_score(X[idx], sampled_labels)

        return silhouette_score(X, labels)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_countries(self, X):
        n_feat = X.shape[1]
        lb, ub = X.min(axis=0), X.max(axis=0)
        # Sample random centroids within per-feature bounds
        countries = np.stack(
            [
                np.random.uniform(lb, ub, (self.k, n_feat))
                for _ in range(self.n_countries)
            ]
        )
        fitness = np.array([self.fitness_fn(X, c) for c in countries])
        return countries, fitness

    # ------------------------------------------------------------------
    # Empire formation
    # ------------------------------------------------------------------

    def form_empires(self, countries, fitness):
        sorted_idx = np.argsort(fitness)[::-1]
        imperialists = countries[sorted_idx[: self.n_imperialists]]
        imperialist_fit = fitness[sorted_idx[: self.n_imperialists]]

        colonies = countries[sorted_idx[self.n_imperialists :]]
        colonies_fit = fitness[sorted_idx[self.n_imperialists :]]

        power = imperialist_fit - np.min(imperialist_fit) + 1e-9
        n_colonies = np.round((power / power.sum()) * len(colonies)).astype(int)

        # FIX 2: ensure all colonies are assigned (fix rounding drop)
        diff = len(colonies) - n_colonies.sum()
        n_colonies[-1] += diff

        empires = []
        start = 0
        for i, n in enumerate(n_colonies):
            n = max(int(n), 0)
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

    # ------------------------------------------------------------------
    # Assimilation
    # ------------------------------------------------------------------

    def assimilation(self, empire, X):
        imperialist = empire["imperialist"]
        colonies = empire["colonies"]

        if len(colonies) == 0:
            return empire

        lb, ub = X.min(axis=0), X.max(axis=0)

        direction = imperialist - colonies
        rand_coeff = np.random.rand(*colonies.shape) * self.assimilation_coef
        colonies = colonies + rand_coeff * direction

        # FIX 3: clip colonies back into data bounds after assimilation
        colonies = np.clip(colonies, lb, ub)

        empire["colonies"] = colonies
        return empire

    # ------------------------------------------------------------------
    # Revolution
    # ------------------------------------------------------------------

    def revolution(self, empire, X):
        colonies = empire["colonies"]
        if len(colonies) == 0:
            return empire

        n_revolt = int(self.revolution_rate * len(colonies))
        if n_revolt <= 0:
            return empire

        n_feat = X.shape[1]
        lb, ub = X.min(axis=0), X.max(axis=0)
        idx = np.random.choice(len(colonies), n_revolt, replace=False)
        colonies[idx] = np.stack(
            [np.random.uniform(lb, ub, (self.k, n_feat)) for _ in range(n_revolt)]
        )

        empire["colonies"] = colonies
        return empire

    # ------------------------------------------------------------------
    # Inter-empire competition
    # ------------------------------------------------------------------

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

        # Remove empires with no colonies (except if only one empire remains)
        empires = [e for e in empires if len(e["colonies"]) > 0 or len(empires) == 1]
        return empires

    # ------------------------------------------------------------------
    # Main fit loop
    # ------------------------------------------------------------------

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
                # FIX 4: re-evaluate imperialist fitness each iteration
                empire["imperialist_fit"] = self.fitness_fn(X, empire["imperialist"])

                empire = self.assimilation(empire, X)   # FIX 3 applied inside
                empire = self.revolution(empire, X)

                if len(empire["colonies"]) > 0:
                    empire["colonies_fit"] = np.array(
                        [self.fitness_fn(X, c) for c in empire["colonies"]]
                    )

                # Swap colony → imperialist if colony is fitter
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

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return self.assign_clusters(X, self.best_centroids)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_history(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.history)
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness (Silhouette Score)")
        plt.title("Fitness History")
        plt.grid()
        plt.tight_layout()
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
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------

    def grid_search(self, X, param_grid=None):
        """
        param_grid example:
            {
                'k': [3, 4],
                'n_countries': [30, 50],
                'n_imperialists': [3, 5],
                'assimilation_coef': [1.0, 1.5],
                'revolution_rate': [0.05, 0.1],
                'beta': [1.0, 2.0]
            }
        """
        original_max_iter = self.max_iter
        self.max_iter = 50  # faster sweeps during search

        best_score = -np.inf
        best_params = None

        for k in param_grid["k"]:
            for n_countries in param_grid["n_countries"]:
                for n_imperialists in param_grid["n_imperialists"]:
                    for assimilation_coef in param_grid["assimilation_coef"]:
                        for revolution_rate in param_grid["revolution_rate"]:
                            for beta in param_grid["beta"]:
                                self.k = k
                                self.n_countries = n_countries
                                self.n_imperialists = n_imperialists
                                self.assimilation_coef = assimilation_coef
                                self.revolution_rate = revolution_rate
                                self.beta = beta
                                self.history = []  # reset history each run

                                self.fit(X)
                                score = self.best_fitness

                                print(
                                    f"k={k}, countries={n_countries}, "
                                    f"imp={n_imperialists}, assim={assimilation_coef}, "
                                    f"rev={revolution_rate}, beta={beta} "
                                    f"-> silhouette={score:.4f}"
                                )

                                if score > best_score:
                                    best_score = score
                                    best_params = {
                                        "k": k,
                                        "n_countries": n_countries,
                                        "n_imperialists": n_imperialists,
                                        "assimilation_coef": assimilation_coef,
                                        "revolution_rate": revolution_rate,
                                        "beta": beta,
                                    }

        # Restore original max_iter
        self.max_iter = original_max_iter

        print("\nBest Parameters:")
        print(best_params)
        print(f"Best Silhouette Score: {best_score:.4f}")

        return best_params, best_score