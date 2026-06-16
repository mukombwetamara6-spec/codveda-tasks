import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
# Make sure 'iris.csv' is in the same directory as this script!
df = pd.read_csv("1) iris.csv")

# Separate features (X) from the true labels (y)
# We drop 'species' because clustering is unsupervised learning
X = df.drop(columns=["species"])
# 2. Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Use the Elbow Method to find the optimal number of clusters
wcss = []  # Within-Cluster Sum of Squares
cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, wcss, marker="o", linestyle="--", color="b")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Inertia)")
plt.xticks(cluster_range)
plt.grid(True)
plt.show()

# 4. Apply K-Means with the optimal number of clusters
# Based on the Iris dataset, the "elbow" typically clarifies at k = 3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 5. Visualize the clusters using a 2D Scatter Plot
# We'll plot Sepal Length vs Sepal Width as an example
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df["sepal_length"],
    y=df["sepal_width"],
    hue=df["Cluster"],
    palette="viridis",
    s=100,
    style=df["Cluster"],
)

plt.title(f"K-Means Clustering (k={optimal_k}) - Sepal Features")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# Optional: Visualize using Petal Features for comparison
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df["petal_length"],
    y=df["petal_width"],
    hue=df["Cluster"],
    palette="Accent",
    s=100,
    style=df["Cluster"],
)

plt.title(f"K-Means Clustering (k={optimal_k}) - Petal Features")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()