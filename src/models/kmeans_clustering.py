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

### --------------------------------------------------------------
### Plot the scaled features to make sure distributions are normal
### StandardScaler() or RobustScaler() did not sufficiently normalize
### the distrubutions, so need to use log transformations to handle
### heavily skewed distrubutions cuased by many zero values

### Clustering by product is more visually informative than by tags
### But it is clearly biased towards number of products versus
### How many times a customer has signed (feature = "Sign In Count")
### Reason is that the vast majority (73.5%!) have not signed in once)
### Should I drop this as a feature and use other features instead?
### --------------------------------------------------------------

# Filter out rows with zero sign-in counts
# signin_filtered = df[df["Sign In Count"] > 0]

# Apply log transformation to handle skewness and zero values
# df["Log_Sign_In_Count"] = np.log1p(signin_filtered["Sign In Count"])
df["Log_Product_Count"] = np.log1p(df["Product_Count"])
df["Log_Tags_Count"] = np.log1p(df["Tags_Count"])

# Impute Log_sign_in_count with median to fill in missing values
# and still matches with the df['Sign In Count'].median() of 0
# df["Log_Sign_In_Count"] = df["Log_Sign_In_Count"].fillna( df["Log_Sign_In_Count"].median())

# Include numerical features and standardize them (removed 'Log_Sign_In_Count' for now)
additional_features = df[["Is_Member", "Log_Product_Count", "Log_Tags_Count"]]
scaler = RobustScaler()
additional_features_scaled = scaler.fit_transform(additional_features)

# Convert the scaled features back to a DataFrame
df_scaled = pd.DataFrame(
    additional_features_scaled, columns=additional_features.columns, index=df.index
)

# # Plot distributions of scaled features
# for column in df_scaled.columns:
#     fig, ax = plt.subplots()
#     ax.hist(df_scaled[column], bins=20, color="blue", alpha=0.7)
#     ax.set_title(f"{column} Histogram (Scaled)")
#     ax.set_xlabel(f"Scaled {column}")
#     ax.set_ylabel("Frequency")
#     plt.savefig(f"../../reports/figures/{column.lower()}_scaled.png")

# # Plot a histogram of sign-in counts for members vs non-members (scaled)
# members_scaled = df_scaled[df["Is_Member"] == 1]["Log_Sign_In_Count"]
# non_members_scaled = df_scaled[df["Is_Member"] == 0]["Log_Sign_In_Count"]

# # Separate plots for members vs non-members (scaled)
# fig, ax = plt.subplots(figsize=(30, 10))
# ax.hist(members_scaled, bins=20, color="green", alpha=0.5, label="Members")
# ax.set_title("Sign-in Count Histogram (Scaled) - Members")
# ax.set_xlabel("Scaled Number of Sign-ins")
# ax.set_ylabel("Frequency")
# ax.legend(loc="upper right")
# plt.savefig("../../reports/figures/signin-frequency_members_scaled.png")

# fig, ax = plt.subplots(figsize=(30, 10))
# ax.hist(non_members_scaled, bins=20, color="orange", alpha=0.5, label="Non-Members")
# ax.set_title("Sign-in Count Histogram (Scaled) - Non-Members")
# ax.set_xlabel("Scaled Number of Sign-ins")
# ax.set_ylabel("Frequency")
# ax.legend(loc="upper right")
# plt.savefig("../../reports/figures/signin-frequency_non-members_scaled.png")


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
kmeans = KMeans(n_clusters=4, random_state=42)
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

plt.figure(figsize=(10, 6))
plt.scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    c=clusters_2d,
    cmap="viridis",
    edgecolor="k",
    s=50,
    alpha=0.7,
)
plt.title("2D PCA Cluster Visualization - Segmentation by Product")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster Label")
plt.grid(True)
plt.savefig("../../reports/figures/kmeans-cluster-2d-pca.png")

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

df["Segment"] = clusters_3d

# Plot a visualization of the number of customers in each cluster
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

# Save the segmented dataset
df.to_pickle("../../data/processed/DDU - Segmented Kajabi Data.pkl")
