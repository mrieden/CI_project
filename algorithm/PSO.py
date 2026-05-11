import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ══════════════════════════════════════════
# تجهيز البيانات
# ══════════════════════════════════════════
def preprocess(df):
    df = df.copy()
    df = df.drop(columns=['ID'])

    for col in ['Ever_Married', 'Graduated', 'Profession', 'Var_1']:
        df[col] = df[col].fillna(df[col].mode()[0])

    df['Work_Experience'] = df['Work_Experience'].fillna(df['Work_Experience'].median())
    df['Family_Size']     = df['Family_Size'].fillna(df['Family_Size'].median())

    le = LabelEncoder()
    for col in ['Gender', 'Ever_Married', 'Graduated', 'Profession',
                'Spending_Score', 'Var_1']:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


train = pd.read_csv('Train.csv')
test  = pd.read_csv('Test.csv')

train_clean = preprocess(train)
test_clean  = preprocess(test)

X_train = train_clean.drop(columns=['Segmentation']).values.astype(float)
X_test  = test_clean.values.astype(float)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# ══════════════════════════════════════════
# PSO Clustering Class
# ══════════════════════════════════════════
class PSOClustering:
    def __init__(self, k=4, n_particles=20, max_iters=100,
                 w=0.7, c1=1.5, c2=1.5, random_state=None):

        self.k = k
        self.n_particles = n_particles
        self.max_iters = max_iters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.random_state = random_state
        self.centroids = None
        self.history = []  # عشان نرسم الـ convergence

    def assign_clusters(self, X, centroids):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fitness(self, X, centroids):
        labels = self.assign_clusters(X, centroids)
        total = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                total += np.sum((cluster_points - centroids[i])**2)
        return total

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # initialize particles
        particles = np.random.rand(self.n_particles, self.k, n_features)
        velocities = np.zeros_like(particles)

        # pBest
        pbest = particles.copy()
        pbest_scores = np.array([self.fitness(X, p) for p in particles])

        # gBest
        gbest = pbest[np.argmin(pbest_scores)]

        # main loop
        for iteration in range(self.max_iters):

            for i in range(self.n_particles):
                score = self.fitness(X, particles[i])

                if score < pbest_scores[i]:
                    pbest[i] = particles[i].copy()
                    pbest_scores[i] = score

            gbest = pbest[np.argmin(pbest_scores)]

            # سجل أفضل قيمة
            self.history.append(np.min(pbest_scores))

            # update
            for i in range(self.n_particles):
                r1 = np.random.rand()
                r2 = np.random.rand()

                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (pbest[i] - particles[i])
                    + self.c2 * r2 * (gbest - particles[i])
                )

                particles[i] += velocities[i]

        self.centroids = gbest
        return self

    def predict(self, X):
        return self.assign_clusters(X, self.centroids)


# ══════════════════════════════════════════
# تشغيل الموديل
# ══════════════════════════════════════════
pso = PSOClustering(k=4, n_particles=30, max_iters=100, random_state=42)
pso.fit(X_train)

labels = pso.predict(X_train)


# ══════════════════════════════════════════
# رسم الـ Clusters باستخدام PCA
# ══════════════════════════════════════════
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_train)

plt.figure()
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels)
plt.title("PSO Clustering")
plt.show()


# ══════════════════════════════════════════
# رسم الـ Convergence Curve
# ══════════════════════════════════════════
plt.figure()
plt.plot(pso.history)
plt.title("Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.show()