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
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │       └── post_model_viz.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target='_blank' href='https://drivendata.github.io/cookiecutter-data-science/'>cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

--------

LESSON NOTES

I plot the scaled features to make sure distributions are normal, and they were not. 
I tried StandardScaler() or RobustScaler(), but either did not sufficiently normalize
the distrubutions, so I needed to use log transformations to handle heavily skewed 
distrubutions cuased by many zero values.

Clustering by product is more visually informative than by tags, but it is clearly 
biased towards number of products versus how many times a customer has signed 
(feature = "Sign In Count")

The reason is that the vast majority (73.5%!) have not signed in once, so I decided 
to drop this as a feature and instead apply PCA on the vectorized products feature.

Increasing the number of clusters from 4 to 6 seemed to improve
KMeans Silhouette score to 0.76 from 0.70.