import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


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
        self.fitness = None

    # -------------------------
    # kmeans++ initialization
    # -------------------------
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

    # -------------------------
    # ASSIGN + SILHOUETTE FITNESS
    # -------------------------
    def _fitness(self, X, centroids):

        distances = np.linalg.norm(
            X[:, np.newaxis] - centroids,
            axis=2
        )

        labels = np.argmin(distances, axis=1)

        unique = np.unique(labels)

        if len(unique) > 1:
            score = silhouette_score(X, labels)
        else:
            score = -1

        return score, labels

    # -------------------------
    # SELECTION (MAXIMIZE NOW)
    # -------------------------
    def _selection(self, population, fitness_scores):

        idx1, idx2 = np.random.choice(len(population), 2, replace=False)

        if fitness_scores[idx1] > fitness_scores[idx2]:
            return population[idx1]
        else:
            return population[idx2]

    # -------------------------
    # CROSSOVER
    # -------------------------
    def _crossover(self, parent1, parent2):

        child = parent1.copy()

        for i in range(self.k):
            if np.random.rand() < 0.5:
                child[i] = parent2[i]

        return child

    # -------------------------
    # MUTATION
    # -------------------------
    def _mutation(self, child, X):

        for i in range(self.k):

            if np.random.rand() < self.mutation_rate:
                idx = np.random.randint(len(X))
                child[i] = X[idx]

        return child

    # -------------------------
    # FIT
    # -------------------------
    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        population = np.array([
            self._kmeans_plus_plus(X)
            for _ in range(self.population_size)
        ])

        no_improvement_count = 0
        best_fitness = -1
        best_centroids = None
        best_labels = None

        for gen in range(self.generations):

            fitness_scores = []
            labels_list = []
            previous_best = best_fitness

            for individual in population:
                fitness, labels = self._fitness(X, individual)
                fitness_scores.append(fitness)
                labels_list.append(labels)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_centroids = individual.copy()
                    best_labels = labels.copy()

            # Early stopping
            if best_fitness > previous_best:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= 10:   # less aggressive
                print(f"Early stopping at generation {gen + 1}")
                break

            # ✅ Elitism: carry best individual into next generation
            elite_idx = np.argmax(fitness_scores)
            new_population = [population[elite_idx].copy()]

            while len(new_population) < self.population_size:
                parent1 = self._selection(population, fitness_scores)
                parent2 = self._selection(population, fitness_scores)

                if np.random.rand() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                child = self._mutation(child, X)
                new_population.append(child)

            population = np.array(new_population)
            print(f"Generation {gen + 1} | Best Silhouette: {best_fitness:.4f}")

        self.centroids = best_centroids
        self.fitness = best_fitness
        return best_labels, best_fitness

    # -------------------------
    # PLOT
    # -------------------------
    def plot_clusters(self, X, labels):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

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
            edgecolors='black',
            label='Centroids'
        )

        plt.title(
            f"Genetic Clustering "
            f"(Silhouette = {self.fitness:.4f})"
        )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.show()

    def grid_search(self, X, param_grid=None):

        # param_grid = {
        #     'k': [3,4],
        #     'population_size': [10, 20, 30],
        #     'generations': [50, 20],
        #     'mutation_rate': [0.05, 0.1, 0.2],
        #     'crossover_rate': [0.7, 0.8, 0.9]
        # }

        best_score = -np.inf
        best_params = None
        k =3
        

        

        for population_size in param_grid['population_size']:

            for generations in param_grid['generations']:

                for mutation_rate in param_grid['mutation_rate']:

                    for crossover_rate in param_grid['crossover_rate']:
                        for k in param_grid['k']:

                            # update parameters
                            self.k = k
                            self.population_size = population_size
                            self.generations = generations
                            self.mutation_rate = mutation_rate
                            self.crossover_rate = crossover_rate

                            # fit model
                            labels, score = self.fit(X)

                            print(
                                f"k={k}, "
                                f"pop={population_size}, "
                                f"gens={generations}, "
                                f"mutation={mutation_rate}, "
                                f"crossover={crossover_rate} "
                                f"-> silhouette={score:.4f}"
                            )

                            # save best
                            if score > best_score:

                                best_score = score

                                best_params = {
                                    'k': k,
                                    'population_size': population_size,
                                    'generations': generations,
                                    'mutation_rate': mutation_rate,
                                    'crossover_rate': crossover_rate
                                }

        print("\nBest Parameters:")
        print(best_params)

        print(
            f"Best Silhouette Score: "
            f"{best_score:.4f}"
        )

        return best_params, best_score