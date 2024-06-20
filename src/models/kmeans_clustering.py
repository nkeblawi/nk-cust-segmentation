import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --------------------------------------------------------------
# Load dataset
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/DDU - Filtered Kajabi Data.pkl")

# --------------------------------------------------------------
# Create scaled features
# --------------------------------------------------------------

# Apply log transformation to handle skewness and zero values
df["Log_Product_Count"] = np.log1p(df["Product_Count"])
df["Log_Tags_Count"] = np.log1p(df["Tags_Count"])

# Include numerical features and standardize them (removed 'Log_Sign_In_Count' for now)
additional_features = df[["Is_Member", "Log_Product_Count", "Log_Tags_Count"]]
scaler = RobustScaler()
additional_features_scaled = scaler.fit_transform(additional_features)

# Convert the scaled features back to a DataFrame
df_scaled = pd.DataFrame(
    additional_features_scaled, columns=additional_features.columns, index=df.index
)

# Export scaled features to a dataset for visualizations
df_scaled.to_pickle("../../data/interim/DDU - Scaled Kajabi Data.pkl")


# --------------------------------------------------------------
# Cluster by Product -- Worked well with a 3D PCA score of 0.70
# Clustering by tags did not work well with a lower score (0.55)
# --------------------------------------------------------------

# Convert the 'Products' column into a binary matrix using CountVectorizer
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(", "), token_pattern=None)
X_products = vectorizer.fit_transform(df["Products"])

# Convert the binary matrix to a DataFrame
df_products = pd.DataFrame(
    X_products.toarray(), columns=vectorizer.get_feature_names_out()
)

# Combine the binary matrix with the additional features
X_combined_prod = pd.concat(
    [
        df_products,
        pd.DataFrame(additional_features_scaled, columns=additional_features.columns),
    ],
    axis=1,
)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=6, random_state=42)
df["Product_Cluster"] = kmeans.fit_predict(X_combined_prod)
score = silhouette_score(X_combined_prod, df["Product_Cluster"])
print("Silhouette Score for Product Clustering:", score)

# Use PCA to plot visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_combined_prod)

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_combined_prod)

# Explained variance can help understand the amount of information retained
print("Explained variance ratio (2D):", pca_2d.explained_variance_ratio_.sum())
print("Explained variance ratio (3D):", pca_3d.explained_variance_ratio_.sum())

# Clustering on 2D PCA results
clusters_2d = kmeans.fit_predict(X_pca_2d)
score_2d = silhouette_score(X_pca_2d, clusters_2d)
print("Silhouette Score for 2D PCA:", score_2d)

# Clustering on 3D PCA results
clusters_3d = kmeans.fit_predict(X_pca_3d)
score_3d = silhouette_score(X_pca_3d, clusters_3d)
print("Silhouette Score for 3D PCA:", score_3d)


# --------------------------------------------------------------
# Visualize PCA results
# ------------------------------------------------------------

# Plot 2D PCA
plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
scatter = ax.scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    c=clusters_2d,
    cmap="viridis",
    edgecolor="k",
    s=50,
    alpha=0.7,
)
ax.set_title("2D PCA Cluster Visualization - Segmentation by Product")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
legend = plt.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)
plt.savefig("../../reports/figures/kmeans-cluster-2d-pca.png")

# Plot 3D PCA
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    X_pca_3d[:, 0],
    X_pca_3d[:, 1],
    X_pca_3d[:, 2],
    c=clusters_3d,
    cmap="viridis",
    edgecolor="k",
    s=50,
    alpha=0.7,
)
ax.set_title("3D PCA Cluster Visualization - Segmentation by Product")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
legend = plt.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)
plt.savefig("../../reports/figures/kmeans-cluster-3d-pca.png")


# --------------------------------------------------------------
# Assign clusters to the original dataset (using Product results)
# --------------------------------------------------------------

# Count the number of customers in each cluster
df["Segment"] = clusters_3d
cluster_counts = df["Segment"].value_counts().sort_index()

# Plotting using matplotlib
plt.figure(figsize=(10, 6))
plt.bar(cluster_counts.index, cluster_counts.values, color="skyblue", edgecolor="k")
plt.xlabel("Cluster Number")
plt.ylabel("Number of Customers")
plt.title("Number of Customers in Each Cluster")
plt.xticks(cluster_counts.index)
plt.grid(axis="y")
plt.savefig("../../reports/figures/num-customers-by-segment.png")


# --------------------------------------------------------------
# Save the segmented dataset
# --------------------------------------------------------------

df.to_pickle("../../data/processed/DDU - Segmented Kajabi Data.pkl")
