import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# Custom transformer for data pruning
class DataPruner(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Add any initialization parameters here if needed
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        # Implement your data cleaning steps here
        df_pruned = df.copy()
        df_pruned = prune_dataset(df_pruned)
        return df_pruned


# Run all the functions to create a pruned dataset
def prune_dataset(df):
    df = remove_duplicates(df)
    df = select_features(df)
    df = label_members(df)
    return df


# --------------------------------------------------
# Create an interim dataset through feature addition, imputation and pruning
# --------------------------------------------------


# Check for duplicates on the 'Email' column and remove them, keeping the first occurrence
def remove_duplicates(df):
    duplicates = df["Email"].duplicated()
    num_duplicates = duplicates.sum()
    df_no_duplicates = df.drop_duplicates(subset="Email", keep="first")
    return df_no_duplicates


# Select the useful features without any identifying information (use ID column instead of name or email)
selected_columns = [
    "ID",
    "Products",
    "Tags",
    "Created At",
    "Sign In Count",
    "Last Activity",
    "Last Sign In At",
]


def select_features(df):
    df = df[selected_columns]
    return df


# --------------------------------------------------
# Tag customers who have active memberships with DDU
# --------------------------------------------------

# Label all members
member_tags = [
    "DDU Expert+ - Purchased",
    "DDU Skills+ - Purchased",
    "DDU Scale+ - Purchased",
    "DDU Insiders - Purchased Monthly",
    "DDU Insiders - Purchased Annual",
    "Measurement Marketing Academy - Purchased",
]
cancelled_tags = ["DDU Insiders - Cancelled"]


# Add a new column that labels all members and subtract those that cancelled
def label_members(df):
    df["Is_Member"] = df["Tags"].apply(lambda x: any(term in x for term in member_tags))
    df["Is_Member"] = df.apply(
        lambda row: (
            0
            if any(cancelled_tags[0] in tag for tag in row["Tags"])
            else row["Is_Member"]
        ),
        axis=1,
    )
    return df
