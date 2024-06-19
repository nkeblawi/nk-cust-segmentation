import pandas as pd
import numpy as np


# Import the cleaned dataset
df = pd.read_pickle("../../data/interim/DDU - Cleaned Kajabi Data.pkl")

# --------------------------------------------------
# Create additional features to use in modeling
# --------------------------------------------------

# Add the number of products and tags each person has
df["Product_Count"] = df["Products"].apply(
    lambda x: 0 if x == "No Product" else len(x.split(", "))
)
df["Tags_Count"] = df["Tags"].apply(
    lambda x: 0 if x == "No Tag" else len(x.split(", "))
)


# Drop rows with zero products or tags, there is no information
# to use in modeling
df_filtered = df[(df["Product_Count"] > 0) & (df["Tags_Count"] > 0)]


# Export the dataset with these new features
df_filtered.to_pickle("../../data/interim/DDU - Filtered Kajabi Data.pkl")
