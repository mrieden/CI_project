import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch import seed  

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
    ):
        self.k = k
        self.n_countries = n_countries
        self.n_imperialists = n_imperialists
        self.max_iter = max_iter
        self.assimilation_coef = assimilation_coef
        self.revolution_rate = revolution_rate
        self.beta = beta
        self.random_state = random_state
        self.history = []

        self.best_centroids = None
        self.best_fitness = np.inf

    def fitness_func(self, centroids, X):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        ssd = 0
        for i in range(self.k):
            points = X[labels == i]
            if len(points) > 0:
                ssd += np.sum((points - centroids[i]) ** 2)
        return ssd

    def initialize_countries(self, X):
        n_feat = X.shape[1]
        lb, ub = X.min(), X.max()
        countries = np.random.uniform(lb, ub, (self.n_countries, self.k, n_feat))
        fitness = np.array([self.fitness_func(c, X) for c in countries])
        return countries, fitness

    def form_empires(self, countries, fitness):
        sorted_idx = np.argsort(fitness)
        imperialists = countries[sorted_idx[: self.n_imperialists]]
        imperialist_fit = fitness[sorted_idx[: self.n_imperialists]]

        colonies = countries[sorted_idx[self.n_imperialists :]]
        colonies_fit = fitness[sorted_idx[self.n_imperialists :]]

        power = np.max(imperialist_fit) - imperialist_fit + 1e-9
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

        weakest_idx = np.argmax(total_powers)
        strongest_idx = np.argmin(total_powers)

        if len(empires[weakest_idx]["colonies"]) > 0:
            w_col_idx = np.argmax(empires[weakest_idx]["colonies_fit"])
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

        for _ in range(self.max_iter):
            for empire in empires:
                empire = self.assimilation(empire)
                empire = self.revolution(empire, X)

                if len(empire["colonies"]) > 0:
                    empire["colonies_fit"] = np.array(
                        [self.fitness_func(c, X) for c in empire["colonies"]]
                    )

                if len(empire["colonies"]) > 0:
                    best_col_idx = np.argmin(empire["colonies_fit"])
                    if empire["colonies_fit"][best_col_idx] < empire["imperialist_fit"]:
                        tmp = empire["imperialist"].copy()
                        tmp_fit = empire["imperialist_fit"]

                        empire["imperialist"] = empire["colonies"][best_col_idx].copy()
                        empire["imperialist_fit"] = empire["colonies_fit"][best_col_idx]

                        empire["colonies"][best_col_idx] = tmp
                        empire["colonies_fit"][best_col_idx] = tmp_fit

            empires = self.competition(empires)

            best_emp = min(empires, key=lambda e: e["imperialist_fit"])
            self.history.append(best_emp["imperialist_fit"])

        self.best_centroids = best_emp["imperialist"]
        self.best_fitness = best_emp["imperialist_fit"]
        return self 
    

    def predict(self, X):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        distances = np.linalg.norm(X[:, np.newaxis] - self.best_centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels
    

    def plot_history(self):
        plt.plot(self.history)
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness (WCSS)")
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

        colors = plt.cm.get_cmap(
            "viridis",
            len(unique_labels)
        )

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

        plt.title("Imperialist Competitive Algorithm Clustering")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        plt.legend()

        plt.grid(True, linestyle='--', alpha=0.5)

        plt.show()