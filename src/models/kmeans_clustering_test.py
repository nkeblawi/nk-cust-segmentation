import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from kmeans_clustering import (
    create_pipeline,
    save_pipeline,
    load_pipeline,
    BinaryMatrixTransformer,
    LogTransformer,
)
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join("../..")))
import src.visualization.plotter as plotter
from src.data.make_dataset import prune_dataset
from src.data.data_cleaning import clean_dataset
from src.features.build_features import create_additional_features

# --------------------------------------------------------------
# Load the raw dataset
# --------------------------------------------------------------

df = pd.read_csv("../../data/raw/DDU - Raw Kajabi Data.csv")

df_pruned = prune_dataset(df)
df_pruned.to_pickle("../../data/interim/DDU - Pruned Kajabi Data.pkl")

df_cleaned = clean_dataset(df_pruned)
df_cleaned.to_pickle("../../data/interim/DDU - Cleaned Kajabi Data.pkl")

df_filtered = create_additional_features(df_cleaned)
df_filtered.to_pickle("../../data/interim/DDU - Filtered Kajabi Data.pkl")


# --------------------------------------------------------------
# Run scaling function and apply Count Vectorization to products
# --------------------------------------------------------------

lf = LogTransformer()
lf.fit(df_filtered)
df_logtransformed = lf.transform(df_filtered)

bmf = BinaryMatrixTransformer()
bmf.fit(df_logtransformed)
X_combined_prod = bmf.transform(df_logtransformed)


# --------------------------------------------------------------
# Apply KMeans clustering
# --------------------------------------------------------------
kmeans = KMeans(n_clusters=6, random_state=42)
df_filtered["Product_Cluster"] = kmeans.fit_predict(X_combined_prod)
score = silhouette_score(X_combined_prod, df_filtered["Product_Cluster"])
print("Silhouette Score for Product Clustering:", score)


# --------------------------------------------------------------
# Use PCA to plot visualization
# --------------------------------------------------------------
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
plotter.ClusterPlotter(
    X_pca=X_pca_2d,
    clusters=clusters_2d,
    title="2D PCA Cluster Visualization - Segmentation by Product",
    xlabel="Principal Component 1",
    ylabel="Principal Component 2",
    filename="../../reports/figures/kmeans-cluster-2d-pca.png",
    plot_type="2d",
).plot()

# Plot 3D PCA
plotter.ClusterPlotter(
    X_pca=X_pca_3d,
    clusters=clusters_3d,
    title="3D PCA Cluster Visualization - Segmentation by Product",
    xlabel="Principal Component 1",
    ylabel="Principal Component 2",
    zlabel="Principal Component 3",
    filename="../../reports/figures/kmeans-cluster-3d-pca.png",
    plot_type="3d",
).plot()

# --------------------------------------------------------------
# Assign clusters to the original dataset (using Product results)
# --------------------------------------------------------------

# Count the number of customers in each cluster and plot the historgram
df_filtered["Segment"] = clusters_3d
cluster_counts = df_filtered["Segment"].value_counts().sort_index()

plotter.BarPlotter(
    data=cluster_counts,
    title="Number of Customers in Each Cluster",
    xlabel="Cluster Number",
    ylabel="Number of Customers",
    filename="../../reports/figures/num-customers-by-segment.png",
    xticks=cluster_counts.index,
).plot()

# --------------------------------------------------------------
# Save the segmented dataset
# --------------------------------------------------------------

new_df = df.merge(
    df_filtered[["ID", "Product_Cluster", "Segment"]], on="ID", how="left"
)
new_df.to_csv("../../data/processed/DDU - Segmented Kajabi Data.csv")


# --------------------------------------------------------------
# Save the KMeans models
# --------------------------------------------------------------

joblib.dump(kmeans, "../../models/kmeans_model.pkl")
joblib.dump(pca_2d, "../../models/pca_2d.pkl")
joblib.dump(pca_3d, "../../models/pca_3d.pkl")


# --------------------------------------------------------------
# Fit the pipeline on trained data and then save it
# --------------------------------------------------------------
trained_data = pd.read_csv("../../data/raw/DDU - Raw Kajabi Data.csv")
pipeline = create_pipeline()
pipeline.fit(trained_data)
save_pipeline(pipeline, "../../models/pca_pipeline.pkl")
