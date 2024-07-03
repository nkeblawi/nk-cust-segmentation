import os
import sys

# Add the project root and src directories to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

from src.models.kmeans_clustering import create_pipeline, save_pipeline, load_pipeline
from flask import Flask, render_template, request, redirect, send_file
import joblib
import pandas as pd
import mpld3
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
import base64
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# Set non-interactive backend for matplotlib
matplotlib.use("Agg")

app = Flask(__name__)
app.static_folder = "static"

# Load the model
kmeans_model = joblib.load("models/kmeans_model.pkl")

# Load the pipeline
pipeline = load_pipeline("models/pca_pipeline.pkl")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"
    if file:
        # Read the CSV file
        df = pd.read_csv(file)

        # Transform the data using the pipeline
        try:
            df_transformed = pipeline.transform(df)
            X_pca_3d = df_transformed
        except Exception as e:
            print(f"Error transforming the data: {e}")
            return f"Error processing file: {str(e)}"

        # Run the model on the transformed data
        clusters_3d = kmeans_model.predict(X_pca_3d)
        score_3d = silhouette_score(X_pca_3d, clusters_3d)

        # Add the cluster labels to the dataframe
        df_pca = pd.DataFrame(X_pca_3d, columns=["PC1", "PC2", "PC3"])
        df_pca["Cluster"] = clusters_3d

        # Perform the left join
        df_pca["ID"] = df["ID"].reset_index(drop=True)
        df_merged = df.merge(df_pca[["ID", "Cluster"]], on="ID", how="left")

        # Save the processed file
        output_file = "processed_output.csv"
        df_merged.to_csv(output_file, index=False)

        # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            X_pca_3d[:, 0],
            X_pca_3d[:, 1],
            X_pca_3d[:, 2],
            c=clusters_3d,
            cmap="viridis",
        )
        ax.set_title("3D Scatter Plot of Clusters")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_zlabel("PCA Component 3")

        # Save the plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode("utf8")

        return render_template(
            "results.html",
            plot_url=plot_url,
            silhouette_score=score_3d,
            download_link=output_file,
        )


@app.route("/download")
def download_file():
    return send_file("data/processed/processed_output.csv", as_attachment=True)


@app.route("/download_test_csv")
def download_test_csv():
    return send_file("data/raw/DDU - Raw Kajabi Data - Test.csv", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
