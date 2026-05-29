# Customer Segmentation — Kajabi Purchase History

A Flask web application that runs unsupervised clustering on a Kajabi customer export CSV and returns a labeled, downloadable segmentation file plus a 3D scatter plot of the results. Two algorithms are available: **KMeans** and a **Gaussian Mixture Model (GMM)**, both operating on a PCA-reduced feature space.

The project was applied to a real marketing campaign and helped boost sales and membership signups by identifying meaningfully different customer groups for targeted outreach.

---

## How It Works

### ML Pipeline

Raw Kajabi CSV data passes through a scikit-learn `Pipeline` defined in `src/models/kmeans_clustering.py`:

| Step | Class | What it does |
|---|---|---|
| 1 | `DataPruner` | Removes irrelevant/low-quality rows |
| 2 | `DataCleaner` | Standardizes column types and fills missing values |
| 3 | `FeatureBuilder` | Derives `Product_Count`, `Tags_Count`, `Is_Member` |
| 4 | `LogTransformer` | Log-transforms skewed counts; scales with `RobustScaler` |
| 5 | `BinaryMatrixTransformer` | One-hot encodes the `Products` column via `CountVectorizer` |
| 6 | `PCA(n_components=3)` | Reduces to 3 principal components for clustering |

The fitted pipeline is saved to `models/pca_pipeline.pkl` and loaded at runtime so the web app never re-trains.

### Model Selection

| Model | File | Notes |
|---|---|---|
| KMeans | `models/kmeans_model.pkl` | Default; fast, interpretable |
| GMM | `models/gmm_3d.pkl` | Slightly better silhouette score in testing; handles ellipsoidal clusters |

Both models use **6 clusters**. The GMM performed marginally better than KMeans on the original dataset.

### Scoring

After clustering, the app computes a **silhouette score** (−1 to 1) and returns context-sensitive feedback on the results page:

| Score range | Interpretation |
|---|---|
| < 0 | Poor fit — review model parameters |
| 0 – 0.25 | Weak structure — check data preprocessing |
| 0.25 – 0.50 | Moderate — try adding more data (300+ rows recommended) |
| 0.50 – 0.75 | Good — clusters are meaningfully distinct |
| 0.75 – 0.90 | Strong segmentation |
| > 0.90 | Suspiciously high — check for overfitting |

---

## Input Data Format

The app expects a Kajabi customer export CSV with at least these columns:

| Column | Type | Description |
|---|---|---|
| `ID` | integer | Unique customer identifier |
| `Products` | string | Comma-separated list of purchased products |
| `Tags` | string | Comma-separated customer tags |
| `Created At` | datetime | Account creation date |
| `Sign In Count` | integer | Total login count |
| `Last Activity` | datetime | Most recent activity timestamp |
| `Last Sign In At` | datetime | Most recent login timestamp |

At least **3 distinct products** and **3 distinct tags** are recommended for meaningful clustering. A scrubbed sample dataset is available to download from the app's index page.

---

## Project Structure

```
nk-cust-segmentation/
├── app.py                          # Flask app — routes: /, /upload, /download, /download_test_csv
├── Dockerfile                      # Production container (gunicorn on port 5000)
├── requirements.txt                # Runtime dependencies
├── templates/
│   ├── index.html                  # Upload form + model selector
│   └── results.html                # Silhouette score, 3D plot, download link, next steps
├── static/css/
│   └── styles.css
├── models/                         # Pre-trained, serialized model artifacts
│   ├── pca_pipeline.pkl            # Full preprocessing + PCA pipeline
│   ├── kmeans_model.pkl            # Trained KMeans model (6 clusters)
│   ├── gmm_3d.pkl                  # Trained GMM (3 PCA components, 6 clusters)
│   ├── pca_2d.pkl                  # 2-component PCA (exploratory use)
│   └── pca_3d.pkl                  # 3-component PCA (standalone, pre-pipeline)
├── src/
│   ├── data/
│   │   ├── make_dataset.py         # DataPruner transformer
│   │   └── data_cleaning.py        # DataCleaner transformer
│   ├── features/
│   │   └── build_features.py       # FeatureBuilder transformer
│   ├── models/
│   │   ├── kmeans_clustering.py    # Pipeline definition, LogTransformer, BinaryMatrixTransformer
│   │   └── kmeans_clustering_test.py
│   └── visualization/
│       ├── plotter.py              # Reusable plot classes
│       ├── visualize.py            # Plot functions
│       └── post_model_viz.py       # Post-clustering analysis helpers
├── notebooks/
│   └── nk-cust-segmentation-report.ipynb   # Full exploratory analysis and model evaluation
├── data/
│   ├── external/                   # Sample CSV for testing
│   ├── interim/                    # Intermediate pickles (binary_df_full.pkl written at runtime)
│   ├── processed/                  # Final labeled output CSV
│   └── raw/                        # Original immutable data
└── references/                     # Data dictionaries and reference materials
```

---

## Running the App

### Docker (recommended)

```bash
docker pull nkeblawi/nk-cust-segmentation:latest
docker run -dp 127.0.0.1:5000:5000 nkeblawi/nk-cust-segmentation:latest
```

Open `http://127.0.0.1:5000`.

### Local (development)

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

---

## App Workflow

1. Upload a Kajabi CSV and select KMeans or GMM.
2. The pipeline transforms the data and the chosen model assigns each customer to one of 6 segments.
3. The results page shows a silhouette score with interpretation, a 3D PCA scatter plot colored by segment, and a download button for the enriched CSV (original columns + `Segment` 1–6).
4. Use the Jupyter notebook (`notebooks/nk-cust-segmentation-report.ipynb`) for deeper post-model analysis: product popularity by segment, engagement patterns, churn likelihood, and subscriber conversion rates.

---

## Dependencies

| Package | Purpose |
|---|---|
| `Flask` | Web framework |
| `gunicorn` | Production WSGI server |
| `scikit-learn` | KMeans, GMM, PCA, Pipeline, silhouette score |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | 3D scatter plot rendering |
| `mpld3` | Matplotlib-to-D3 bridge (exploratory use) |
| `joblib` | Model serialization / deserialization |
| `ipython` | Notebook support |

---

## Future Improvements

- **Results page insights** — add post-model visualizations (product popularity per segment, engagement heatmaps) directly on `results.html` so actionable findings are available immediately after clustering.
- **Product grouping** — group individual products into broader categories before encoding to help the model find more meaningful segment boundaries.
- **Multi-platform support** — build a second pipeline for ConvertKit exports and expose a platform selector in the UI.
- **Classification follow-on** — once segments are manually labeled for a subset, train a classifier to predict segment membership for new customers without re-running clustering.
- **HDBSCAN / graph clustering** — evaluate density-based and graph-based alternatives, which may handle variable cluster density better than KMeans or GMM.
