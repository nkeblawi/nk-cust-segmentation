nk-cust-segmentation
==============================

Using cluster algorithm to segment customers based on purchase history

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
        ├── kmeans_model.pkl
        ├── pca_2d.pkl
        ├── pca_3d.pkl
        └── pca_pipeline.pkl
    │
    ├── notebooks          <- Jupyter notebooks for exploratory analysis and reporting.
    │   └── nk-cust-segmentation-report.ipynb
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │   └── data_cleaning.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── kmeans_clustering.py
    │   │   ├── kmeans_clustering_test.py   <- For testing, could be converted into a notebook
    │   │
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── plotter.py          <- Custom plotter classes for use across the project
    │       └── visualize.py        <- Custom plotter functions that use the plotter classes
    │       └── post_model_viz.py   <- For testing, could be used within a notebook
    |
    ├── templates  <- HTML files for frontend
    │   │   └── index.html
    │   │   └── results.html
    |
    └── app.p              <- Starts the frontend web app


--------

# PROJECT SUMMARY

KMeans with PCA did a decent job of grouping a specific customer list for a specific situation
within tight time constraints, and has helped boost sales and membership signups in a marketing
campaign. 

However, given the limitations, post-model analysis was necessary to label each 
segment, and further improvements to clustering results can be obtained by testing other models
such as HDBSCAN, graph clustering, and GMMs. 

The Gaussian Mixture Model performed slightly better than KMeans using Principal Component 
Analysis with 3 components and 6 clusters. Exploratory analysis was done in this notebook:

/notebooks/nk-cust-segmentation-report.ipynb

The model and pipeline files were exported to the /models/ folder. 

Once all clusters/segments have been labeled for a subset of data, a classification model can be 
used to predict segmentation for all remaining data within the same context of the original use 
case. This may provide an additional boost to subsequent marketing campaigns.

To start the project, follow these instructions:

- Pull latest Docker image from my Azure Image Registry:
`docker pull nkcustseg.azurecr.io/nk-cust-segmentation:latest`

- Run the container from image:
`docker run -dp 127.0.0.1:5000:5000 nkeblawi/nk-cust-segmentation:latest`