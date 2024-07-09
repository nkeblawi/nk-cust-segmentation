import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

import os
import sys

sys.path.append(os.path.abspath(os.path.join("../")))
sys.path.append(os.path.abspath(os.path.join("../../")))
from src.data.make_dataset import DataPruner
from src.data.data_cleaning import DataCleaner
from src.features.build_features import FeatureBuilder


def create_pipeline():
    """
    A function to create a pipeline for a machine learning model.
    Returns:
        Pipeline: A scikit-learn pipeline object with defined steps.
    """
    # Define the pipeline steps up to PCA, then apply KMeans separately
    pipeline = Pipeline(
        [
            ("data_pruner", DataPruner()),
            ("data_cleaner", DataCleaner()),
            ("feature_builder", FeatureBuilder()),
            ("log_transform", LogTransformer()),
            ("binary_matrix", BinaryMatrixTransformer()),
            ("pca", PCA(n_components=3)),
        ]
    )
    return pipeline


def save_pipeline(pipeline, filename):
    joblib.dump(pipeline, filename)


def load_pipeline(filename):
    try:
        return joblib.load(filename)
    except ModuleNotFoundError:
        print("Module not found. Attempting to fix...")
        # Add the current directory to sys.path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        # Try loading again
        return joblib.load(filename)


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Apply log transformation to the 'Product_Count' and 'Tags_Count' columns of the input DataFrame.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the 'Product_Count' and 'Tags_Count' columns.

    Returns:
        pandas.DataFrame: The DataFrame with the 'Log_Product_Count' and 'Log_Tags_Count' columns added,
        and the 'Is_Member', 'Log_Product_Count', and 'Log_Tags_Count' columns scaled using RobustScaler.
    """

    def __init__(self):
        self.scaler = RobustScaler()

    def fit(self, df, y=None):
        # Apply log transformation
        df_log = df.copy()
        df_log["Log_Product_Count"] = np.log1p(df["Product_Count"])
        df_log["Log_Tags_Count"] = np.log1p(df["Tags_Count"])

        # Fit the scaler on the log-transformed data
        self.scaler.fit(df_log[["Log_Product_Count", "Log_Tags_Count", "Is_Member"]])
        return self

    # Apply log transformation to handle skewness and zero values
    def transform(self, df):
        # Create a copy to avoid modifying the original dataframe
        df_transformed = df.copy()
        df_transformed["Log_Product_Count"] = np.log1p(df_transformed["Product_Count"])
        df_transformed["Log_Tags_Count"] = np.log1p(df_transformed["Tags_Count"])

        # Include numerical features and standardize them (removed 'Log_Sign_In_Count' for now)
        scaled_features = self.scaler.transform(
            df_transformed[["Log_Product_Count", "Log_Tags_Count", "Is_Member"]]
        )

        # Create a new dataframe with scaled features
        df_scaled = pd.DataFrame(
            scaled_features,
            columns=[
                "Log_Product_Count",
                "Log_Tags_Count",
                "Is_Member",
            ],
            index=df_transformed.index,
        )

        # Add other columns that should be passed through
        for col in df.columns:
            if col not in df_scaled.columns and col not in [
                "Product_Count",
                "Tags_Count",
            ]:
                df_scaled[col] = df[col]

        # Convert the scaled features back to a DataFrame
        return df_scaled


# Instead of using lambda function, we can use tokenizer_func to pickle the pipeline
def tokenizer_func(x):
    return x.split(", ") if isinstance(x, str) else ["No Product"]


class BinaryMatrixTransformer(BaseEstimator, TransformerMixin):
    """
    Fit the model with the given data.

    Parameters:
        X (array-like): The input data.
        y (None, optional): Not used, defaults to None.

    Returns:
        self: The instance of the class.
    """

    def __init__(self):
        self.vectorizer = CountVectorizer(tokenizer=tokenizer_func, token_pattern=None)

    def fit(self, df, y=None):
        # Replace NaN values with "No Product"
        products = df["Products"].fillna("No Product")
        self.vectorizer.fit(products)
        return self

    # Convert the 'Products' column into a binary matrix using CountVectorizer
    def transform(self, df):
        # Replace NaN values with "No Product"
        products = df["Products"].fillna("No Product")

        # Create the binary matrix using CountVectorizer for products
        X_products = self.vectorizer.transform(products)

        # Convert the binary matrix to a DataFrame
        df_products = pd.DataFrame(
            X_products.toarray(),
            columns=self.vectorizer.get_feature_names_out(),
            index=df.index,
        )

        # Combine the binary matrix with the additional features
        df_binary = pd.concat(
            [df_products, df[["Log_Product_Count", "Log_Tags_Count", "Is_Member"]]],
            axis=1,
        )

        return df_binary
