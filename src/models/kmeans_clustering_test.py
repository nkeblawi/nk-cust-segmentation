import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from kmeans_clustering import (
    create_pipeline,
    save_pipeline,
    BinaryMatrixTransformer,
    LogTransformer,
)
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join("../..")))
import src.visualization.plotter as plotter
from src.data.make_dataset import DataPruner
from src.data.data_cleaning import DataCleaner
from src.features.build_features import FeatureBuilder

# --------------------------------------------------------------
# Load the raw dataset
# --------------------------------------------------------------

df = pd.read_csv("../../data/raw/DDU - Raw Kajabi Data.csv")


# --------------------------------------------------------------
# Create a random train-test split
# --------------------------------------------------------------

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
pd.DataFrame.to_csv(df_test, "../../data/raw/DDU - Raw Kajabi Data - Test.csv")


# --------------------------------------------------------------
# Process the training data
# --------------------------------------------------------------

pruner = DataPruner()
df_pruned = pruner.fit_transform(df_train)
df_pruned.to_pickle("../../data/interim/DDU - Pruned Kajabi Data.pkl")

cleaner = DataCleaner()
df_cleaned = cleaner.fit_transform(df_pruned)
df_cleaned.to_pickle("../../data/interim/DDU - Cleaned Kajabi Data.pkl")

fbuilder = FeatureBuilder()
df_filtered = fbuilder.fit_transform(df_cleaned)
df_filtered.to_pickle("../../data/interim/DDU - Filtered Kajabi Data.pkl")


# --------------------------------------------------------------
# Run scaling function and apply Count Vectorization to products
# --------------------------------------------------------------

# Log Transform the 'Product_Count' and 'Tags_Count' columns
lf = LogTransformer()
lf.fit(df_filtered)
df_logtransformed = lf.transform(df_filtered)

# Apply Count Vectorization to products
bmf = BinaryMatrixTransformer()
bmf.fit(df_logtransformed)
X = bmf.transform(df_logtransformed)

# --------------------------------------------------------------
# Apply KMeans clustering
# --------------------------------------------------------------

kmeans = KMeans(n_clusters=6, random_state=42)

# Run the KMeans model
df_filtered = pd.read_pickle("../../data/interim/binary_df_full.pkl")
df_filtered["KMeans_Cluster"] = kmeans.fit_predict(X)

score = silhouette_score(X, df_filtered["KMeans_Cluster"])
print("Silhouette Score for KMeans Clustering:", score)


# --------------------------------------------------------------
# Use PCA to plot visualization
# --------------------------------------------------------------

pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X)

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X)

# Explained variance can help understand the amount of information retained
print("Explained variance ratio (2D):", pca_2d.explained_variance_ratio_.sum())
print("Explained variance ratio (3D):", pca_3d.explained_variance_ratio_.sum())

# KMeans Clustering on 2D PCA results
clusters_2d = kmeans.fit_predict(X_pca_2d)
score_2d = silhouette_score(X_pca_2d, clusters_2d)
print("Silhouette Score for KMeans 2D PCA:", score_2d)

# KMeans Clustering on 3D PCA results
clusters_3d = kmeans.fit_predict(X_pca_3d)
score_3d = silhouette_score(X_pca_3d, clusters_3d)
print("Silhouette Score for KMeans 3D PCA:", score_3d)


# --------------------------------------------------------------
# Repeat using the Gaussian Mixture Model (3D PCA only)
# ------------------------------------------------------------

gmm_3d = GaussianMixture(n_components=6, random_state=42)
gmm_labels_3d = gmm_3d.fit_predict(X_pca_3d)

gmm_silhouette_3d = silhouette_score(X_pca_3d, gmm_labels_3d)
print("Silhouette Score for GMM Clustering with 3D PCA:", gmm_silhouette_3d)


# --------------------------------------------------------------
# Visualize PCA results
# ------------------------------------------------------------

# Plot 2D PCA for KMeans
plotter.ClusterPlotter(
    X_pca=X_pca_2d,
    clusters=clusters_2d,
    title="KMeans 2D PCA Cluster Visualization - Segmentation by Product",
    xlabel="Principal Component 1",
    ylabel="Principal Component 2",
    filename="../../reports/figures/kmeans-cluster-2d-pca.png",
    plot_type="2d",
).plot()

# Plot 3D PCA for KMeans
plotter.ClusterPlotter(
    X_pca=X_pca_3d,
    clusters=clusters_3d,
    title="KMeans 3D PCA Cluster Visualization - Segmentation by Product",
    xlabel="Principal Component 1",
    ylabel="Principal Component 2",
    zlabel="Principal Component 3",
    filename="../../reports/figures/kmeans-cluster-3d-pca.png",
    plot_type="3d",
).plot()

# Plot 3D PCA for GMM
plotter.ClusterPlotter(
    X_pca=X_pca_3d,
    clusters=gmm_labels_3d,
    title="GMM 3D PCA Cluster Visualization - Segmentation by Product",
    xlabel="Principal Component 1",
    ylabel="Principal Component 2",
    zlabel="Principal Component 3",
    filename="../../reports/figures/gmm-cluster-3d-pca.png",
    plot_type="3d",
).plot()

# --------------------------------------------------------------
# Assign clusters to the original dataset (using Product results)
# --------------------------------------------------------------

# Count the number of customers in each cluster and plot the historgram
df_filtered["KMeans_3D_PCA_Cluster"] = clusters_3d
cluster_counts = df_filtered["KMeans_3D_PCA_Cluster"].value_counts().sort_index()

df_filtered["GMM_3D_PCA_Cluster"] = gmm_labels_3d
gmm_cluster_counts = df_filtered["GMM_3D_PCA_Cluster"].value_counts().sort_index()

plotter.BarPlotter(
    data=cluster_counts,
    title="Number of Customers in Each KMeans Cluster",
    xlabel="Cluster Number",
    ylabel="Number of Customers",
    filename="../../reports/figures/kmeans-num-customers-by-segment.png",
    xticks=cluster_counts.index,
).plot()

plotter.BarPlotter(
    data=gmm_cluster_counts,
    title="Number of Customers in Each GMM Cluster",
    xlabel="Cluster Number",
    ylabel="Number of Customers",
    filename="../../reports/figures/gmm-num-customers-by-segment.png",
    xticks=gmm_cluster_counts.index,
).plot()

# --------------------------------------------------------------
# Save the segmented dataset
# --------------------------------------------------------------

new_columns = pd.concat([df_filtered["ID"], df_filtered.iloc[:, 7:]], axis=1)
new_df = df_train.merge(new_columns, on="ID", how="left")
new_df.drop(columns=["Log_Product_Count", "Log_Tags_Count"], inplace=True)
new_df.to_csv("../../data/processed/DDU - Segmented Kajabi Data.csv")

# --------------------------------------------------------------
# Save the KMeans models
# --------------------------------------------------------------

joblib.dump(kmeans, "../../models/kmeans_model.pkl")
joblib.dump(pca_2d, "../../models/pca_2d.pkl")
joblib.dump(pca_3d, "../../models/pca_3d.pkl")
joblib.dump(gmm_3d, "../../models/gmm_3d.pkl")


# --------------------------------------------------------------
# Fit the pipeline on the data and then save it
# --------------------------------------------------------------
data = pd.read_csv("../../data/raw/DDU - Raw Kajabi Data.csv")
pipeline = create_pipeline()
pipeline.fit(data)
save_pipeline(pipeline, "../../models/pca_pipeline.pkl")
