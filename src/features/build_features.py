import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# Custom transformer for feature engineering
class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Add any initialization parameters here if needed
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        # Implement your feature engineering steps here
        df_filtered = create_additional_features(df)
        return df_filtered


# --------------------------------------------------
# Create additional features to use in modeling
# --------------------------------------------------


def create_additional_features(df):

    # Add the number of products and tags each person has
    df["Product_Count"] = df["Products"].apply(
        lambda x: 0 if x == "No Product" else len(x.split(", "))
    )
    df["Tags_Count"] = df["Tags"].apply(
        lambda x: 0 if x == "No Tag" else len(x.split(", "))
    )

    # Drop rows with zero products or tags, there is no information to use in modeling
    df_filtered = df[(df["Product_Count"] > 0) & (df["Tags_Count"] > 0)]

    return df_filtered
