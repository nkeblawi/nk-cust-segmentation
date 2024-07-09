import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin


# Custom transformer for data cleaning
class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Add any initialization parameters here if needed
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        # Implement your data cleaning steps here
        df_cleaned = df.copy()
        df_cleaned = clean_dataset(df_cleaned)
        return df_cleaned


# --------------------------------------------------
# Clean dataset sequentially and save
# --------------------------------------------------


def clean_dataset(df):
    df = impute_missing_values(df)
    df = convert_date_columns(df)
    df = clean_products(df)
    df = clean_tags(df)
    return df


# Function that imputes missing values under 'Last Activity' and
# 'Last Sign In At' columns using values from the Created At column
def impute_missing_values(df):
    df["Products"] = df["Products"].fillna("No Product")
    df["Tags"] = df["Tags"].fillna("No Tag")
    df["Last Activity"] = df["Last Activity"].fillna(df["Created At"])
    df["Last Sign In At"] = df["Last Sign In At"].fillna(df["Created At"])
    df["Sign In Count"] = df["Sign In Count"].fillna(0)
    return df


# Function to convert date columns from object to date formats
def convert_date_columns(df):
    df["Created At"] = pd.to_datetime(df["Created At"], utc=True)
    df["Last Activity"] = pd.to_datetime(df["Last Activity"], utc=True)
    df["Last Sign In At"] = pd.to_datetime(df["Last Sign In At"], utc=True)

    # Convert 'Sign In Count' to integer instead of floats
    df["Sign In Count"] = df["Sign In Count"].astype(int)

    # Reset the index of the dataframe
    df = df.reset_index(drop=True)

    return df


# List any products or tags that we don't need
products_not_needed = [
    "AppSumo Deal Information",
    "Brain Gainz",
    "Cheat Codes",
    "Dasboard Pack",
    "DD Insider Exclusive Resources",
    "DD Insider Exclusive Workshops",
    "DDU Tools+",
    "GA4 CYA Blueprint Community archived 1668007437",
    "Insider Perks",
    "Office Hours Schedule & Recordings",
    "Process Pack",
    "Road to Recurring Challenge",
]
tags_not_needed = [
    "[220901] Exit Intent Popup - Opted In",
    "[220901] Funnel - Do Not Disturb",
    "[220901] Sequence A - Completed",
    "[221024] Appsumo - Purchased",
    "[221125] BFCM Purchaser",
    "[221130] DDU+ - Purchased",
    "ALGBC - Purchaser",
    "ASBC - Purchaser",
    "DDIG - Purchaser",
    "DDIP - Purchaser",
    "DDIW - Purchaser",
    "DDU Scale+ Team Member",
    "DP - Purchaser",
    "FBAB - Purchaser",
    "FBBC - Purchaser",
    "ISA - Purchaser",
    "Lead Magnet - Non Buyers",
    "LJBC - Purchaser",
    "PAP - Purchaser",
    "RRC - Purchaser",
    "RRW - Purchaser",
    "SFZ - Purchaser",
    "Funnel - Do Not Disturb",
    "DDU+ - Purchased",
]


# Function for stripping cohort numbers and extra characters from products or tags
def clean_data(data):
    # Remove cohort numbers with optional preceding dash
    data = re.sub(r" -?Cohort \d+", "", data)

    # Remove numbered brackets
    data = re.sub(r"\[\d+\]", "", data)

    # Remove emojis
    emoj = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        re.UNICODE,
    )
    data = re.sub(emoj, "", data)

    # Remove products not needed
    data = re.sub(rf"\b({'|'.join(products_not_needed)})\b", "", data)
    data = re.sub(rf"\b({'|'.join(tags_not_needed)})\b", "", data)

    return data.strip()


# --------------------------------------------------
# Clean the 'Products' column
# --------------------------------------------------


def clean_products(df):

    # Explode the 'Products' column to handle each product individually
    df_exploded = df.explode("Products")

    # Clean each product
    df_exploded["Products"] = df_exploded["Products"].apply(
        lambda x: [clean_data(item) for item in x.split(",")]
    )

    # Remove duplicates
    df_exploded["Products"] = df_exploded["Products"].apply(lambda x: list(set(x)))

    # Group by the original index to reassemble the lists
    df_cleaned_products = df_exploded.groupby(level=0).agg(
        {"Products": lambda x: ", ".join(sum(x, []))}
    )
    df_cleaned_products["Products"] = df_cleaned_products["Products"].str.lstrip(", ")

    # Reassemble the cleaned products back into the DataFrame
    df["Products"] = df_cleaned_products["Products"]

    # Replace empty strings with 'No Product'
    df["Products"].replace("", "No Product", inplace=True)

    return df


# --------------------------------------------------
# Clean the 'Tags' column
# --------------------------------------------------


def clean_tags(df):

    # Explode the 'Products' column to handle each product individually
    df_exploded = df.explode("Tags")

    # Clean each product
    df_exploded["Tags"] = df_exploded["Tags"].apply(
        lambda x: [clean_data(item) for item in x.split(",")]
    )

    # Remove duplicates
    df_exploded["Tags"] = df_exploded["Tags"].apply(lambda x: list(set(x)))

    # Group by the original index to reassemble the lists
    df_cleaned_tags = df_exploded.groupby(level=0).agg(
        {"Tags": lambda x: ", ".join(sum(x, []))}
    )

    # Reassemble the cleaned products back into the DataFrame
    df["Tags"] = df_cleaned_tags["Tags"]

    # Replace empty strings with 'No Tag'
    df["Tags"].replace("", "No Tag", inplace=True)

    return df
