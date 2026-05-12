import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class GeneticKMeans:
    def __init__(
        self,
        k=3,
        population_size=20,
        generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8,
        random_state=None
    ):
        self.k = k
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.random_state = random_state
        self.centroids = None

    def _kmeans_plus_plus(self, X):
        n_samples = X.shape[0]

        centroids = []

        first_idx = np.random.randint(n_samples)
        centroids.append(X[first_idx])

        for _ in range(1, self.k):

            distances = np.array([
                min(np.linalg.norm(x - c) ** 2 for c in centroids)
                for x in X
            ])

            probabilities = distances / distances.sum()

            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def _fitness(self, X, centroids):

        distances = np.linalg.norm(
            X[:, np.newaxis] - centroids,
            axis=2
        )

        labels = np.argmin(distances, axis=1)

        wcss = 0

        for j in range(self.k):
            cluster_points = X[labels == j]

            if len(cluster_points) > 0:
                wcss += np.sum(
                    (cluster_points - centroids[j]) ** 2
                )

        return wcss, labels

    def _selection(self, population, fitness_scores):

        idx1, idx2 = np.random.choice(
            len(population),
            2,
            replace=False
        )

        if fitness_scores[idx1] < fitness_scores[idx2]:
            return population[idx1]
        else:
            return population[idx2]


    def _crossover(self, parent1, parent2):

        child = parent1.copy()

        for i in range(self.k):

            if np.random.rand() < 0.5:
                child[i] = parent2[i]

        return child

    def _mutation(self, child, X):

        for i in range(self.k):

            if np.random.rand() < self.mutation_rate:

                random_idx = np.random.randint(len(X))
                child[i] = X[random_idx]

        return child


    def fit(self, X):

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        population = np.array([
            self._kmeans_plus_plus(X)
            for _ in range(self.population_size)
        ])

        best_fitness = float("inf")
        best_centroids = None
        best_labels = None

        for generation in range(self.generations):

            fitness_scores = []
            labels_list = []

            for individual in population:

                fitness, labels = self._fitness(X, individual)

                fitness_scores.append(fitness)
                labels_list.append(labels)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_centroids = individual.copy()
                    best_labels = labels.copy()

            new_population = []

            while len(new_population) < self.population_size:

                parent1 = self._selection(population, fitness_scores)
                parent2 = self._selection(population, fitness_scores)

                if np.random.rand() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                # Mutation
                child = self._mutation(child, X)

                new_population.append(child)

            population = np.array(new_population)

            print(
                f"Generation {generation + 1} "
                f"| Best Fitness: {best_fitness:.4f}"
            )

        self.centroids = best_centroids

        return best_labels, best_fitness

    def plot_clusters(self, X, labels):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        pca = PCA(n_components=2)

        X_pca = pca.fit_transform(X)

        centroids_pca = pca.transform(self.centroids)

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

        plt.title("Genetic Algorithm Clustering")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        plt.legend()

        plt.grid(True, linestyle='--', alpha=0.5)

        plt.show()