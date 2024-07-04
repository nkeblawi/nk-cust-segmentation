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
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
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
    │   │   ├── kmeans_clustering_test.py
    │   │
    │   └── templates  <- HTML files for frontend
    │   │   └── index.html
    │   │   └── results.html
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── plotter.py
    │       └── visualize.py
    │       └── post_model_viz.py
    │
    └── app.p              <- Starts the frontend web app


--------

<p><small>Project based on the <a target='_blank' href='https://drivendata.github.io/cookiecutter-data-science/'>cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

--------

### PROJECT NOTES

# Feature engineering
I plot the scaled features to make sure distributions are normal, and they were not. 
I tried StandardScaler() or RobustScaler(), but either did not adequately normalize
the distrubutions, so I needed to use log transformations to handle heavily skewed 
distrubutions cuased by many zero values.

Clustering by product is more visually informative than by tags, but it is clearly 
biased towards number of products versus how many times a customer has signed in
(feature = "Sign In Count"). 

The reason is that the vast majority (73.5%!) have not signed in once, so I decided 
to drop this as a feature and instead apply PCA on the vectorized products feature. 
Before applying 2D and 3D PCA, I used scikit-learn's CountVectorizer() class to apply 
both one-hot encoding and tokenization on product data in one step. 

# Dimensionality reduction
3D PCA seems to score slightly higher than 2D, but both score well. Increasing the 
number of clusters from 4 to 6 improved the KMeans Silhouette score to 0.76 from 0.70.

Other dimensionality reduction techniques such as UMAP have been used, but these did
not score as well as PCA has. For this reason, PCA has been selected as the primary
dimensionality reduction technique for KMeans (and any new clustering model I test 
going forward).

# Model discussion and limitations
KMeans was used with 2D and 3D PCA-processed data to understand how the customer list
is clustered or grouped, and whether useful information can be gleaned from the results.

The silhouette score is used to evaluate the model. It measures how similar a point is to 
its own cluster commpared to other clusters, with a score ranging between -1 and 1.
- Close to 1 means well-defined clusters
- Close to 0 means overlapping or poorly-defined clusters (or lack thereof)
- Less than 0: Incorrect clustering or serious issues with cluster separation

However, KMeans has some limitations that prevent use of this model for broader use case
scenarios, such as segmentation on various datasets other than Kajabi or ConvertKit.

For instance, it cannot be used to recommend products to a specific group of customers. 
For that, a recommender system would be used. Being an unsupervised ML algorithm, it 
cannot identify each cluster/segment with meaningful information. Post-model analysis is
needed to identify and label these clusters accordingly. 

Also, KMeans requires the number of clusters to be specified prior to running the model, 
given that n_clusters is one of the most important hyperparameters. This is limiting 
since we may not know how many segments/groups actually exist within our user list.

Other limitations of KMeans include:
- assumption that clusters are spherical with equal variance (not the case for my results)
- sensitive to outliers that affect the centroid positions, leading to bias
- lacks flexibility with hyperparameters (users shouldn't have to set n_clusters themselves)

# Next steps
Based on the above limitations of KMeans clustering (which did a good job with exploratory
analysis and a "first draft" of a clustering algorithm that got the job done in a pinch), 
the next steps include testing other clustering models such as:

1) Graph clustering using the Louvain Method. This works well with large datasets and can 
capture non-linear relationships between customers based on their interactions with products.

2) Density-based spatial clustering using HDBSCAN, which does not require specifiying the
number of clusters in advance. Use this if the number of segments is not known in advance.

3) Gaussian Mixture Models (GMM), which provides soft clustering where each customer can 
belong to multiple segments with different probabilities (very likely when multiple products
or multiple account tiers are offered by a business).
