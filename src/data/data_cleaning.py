import pandas as pd
import numpy as np
import re

# Load the pruned dataset
df = pd.read_pickle("../../data/interim/DDU - Pruned Kajabi Data.pkl")
df_cleaned = df.copy()

###### DATA CLEANING ########

# Impute missing values under 'Last Activity' and 'Last Sign In At' columns
# using values from the Created At column
df_cleaned["Last Activity"] = df_cleaned["Last Activity"].fillna(
    df_cleaned["Created At"]
)
df_cleaned["Last Sign In At"] = df_cleaned["Last Sign In At"].fillna(
    df_cleaned["Created At"]
)
df_cleaned["Sign In Count"] = df_cleaned["Sign In Count"].fillna(0)

# Convert date columns from object to date formats
df_cleaned["Created At"] = pd.to_datetime(df_cleaned["Created At"], utc=True)
df_cleaned["Last Activity"] = pd.to_datetime(df_cleaned["Last Activity"], utc=True)
df_cleaned["Last Sign In At"] = pd.to_datetime(df_cleaned["Last Sign In At"], utc=True)

# Convert 'Sign In Count' to integer instead of floats
df_cleaned["Sign In Count"] = df_cleaned["Sign In Count"].astype(int)

# Reset the index of the dataframe
df_cleaned = df_cleaned.reset_index(drop=True)

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
df_cleaned["Products"] = df_cleaned_products["Products"]

# Replace empty strings with 'No Product'
df_cleaned["Products"].replace("", "No Product", inplace=True)

# --------------------------------------------------
# Clean the 'Tags' column
# --------------------------------------------------

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
df_cleaned["Tags"] = df_cleaned_tags["Tags"]

# Replace empty strings with 'No Tag'
df_cleaned["Tags"].replace("", "No Tag", inplace=True)


# --------------------------------------------------
# Save the cleaned dataset
# --------------------------------------------------

df_cleaned.to_pickle("../../data/interim/DDU - Cleaned Kajabi Data.pkl")


# --------------------------------------------------
# Export product and tag lists for use in feature engineering
# --------------------------------------------------

# Flatten the list of comma-delimited products and tags into individual items
flattened_products = [
    item.strip() for sublist in df_cleaned["Products"] for item in sublist.split(",")
]
unique_products = set(flattened_products)
unique_products_list = list(unique_products)

flattened_tags = [
    item.strip() for sublist in df_cleaned["Tags"] for item in sublist.split(",")
]
unique_tags = set(flattened_tags)
unique_tags_list = list(unique_tags)

pd.DataFrame({"Products": unique_products_list}).to_csv(
    "../../data/interim/DDU Product List.csv"
)
pd.DataFrame({"Tags": unique_tags_list}).to_csv("../../data/interim/DDU Tag List.csv")
