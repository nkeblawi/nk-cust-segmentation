import pandas as pd
import numpy as np

# Load the raw dataset
df = pd.read_csv("../../data/raw/DDU - Raw Kajabi Data.csv")

# --------------------------------------------------
# Create an interim dataset through feature addition, imputation and pruning
# --------------------------------------------------

# Check for duplicates on the 'Email' column
duplicates = df["Email"].duplicated()
num_duplicates = duplicates.sum()
print(f"Number of duplicate emails: {num_duplicates}")

# Remove rows with duplicate emails, keeping the first occurrence
df_no_duplicates = df.drop_duplicates(subset="Email", keep="first")

# Select the useful features without any identifying information (use ID column instead of name or email)
columns = [
    "ID",
    "Products",
    "Tags",
    "Created At",
    "Sign In Count",
    "Last Activity",
    "Last Sign In At",
]
df_pruned = df_no_duplicates[columns]

# Impute missing values under 'Last Activity' and 'Last Sign In At' columns
# using values from the Created At column
df_pruned["Last Activity"] = df_pruned["Last Activity"].fillna(df_pruned["Created At"])
df_pruned["Last Sign In At"] = df_pruned["Last Sign In At"].fillna(
    df_pruned["Created At"]
)
df_pruned["Sign In Count"] = df_pruned["Sign In Count"].fillna(0)

# Convert date columns from object to date formats
df_pruned["Created At"] = pd.to_datetime(df_pruned["Created At"], utc=True)
df_pruned["Last Activity"] = pd.to_datetime(df_pruned["Last Activity"], utc=True)
df_pruned["Last Sign In At"] = pd.to_datetime(df_pruned["Last Sign In At"], utc=True)

# Convert 'Sign In Count' to integer instead of floats
df_pruned["Sign In Count"] = df_pruned["Sign In Count"].astype(int)

# Reset the index of the dataframe
df_pruned = df_pruned.reset_index(drop=True)

# Impute missing values in 'Products' and 'Tags' columns
df_pruned["Products"] = df_pruned["Products"].fillna("No Product")
df_pruned["Tags"] = df_pruned["Tags"].fillna("No Tag")

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
df_pruned["Is_Member"] = (
    df_pruned["Tags"]
    .apply(lambda x: any(term in x for term in member_tags))
    .astype(int)
)
# using the cancelled_tags variable, turn Is_Member back to 0 for cancelled members
df_pruned["Is_Member"] = df_pruned.apply(
    lambda row: (
        0 if any(cancelled_tags[0] in tag for tag in row["Tags"]) else row["Is_Member"]
    ),
    axis=1,
)

# --------------------------------------------------
# Save the interim dataset - no missing values
# --------------------------------------------------

df_pruned.to_pickle("../../data/interim/DDU - Pruned Kajabi Data.pkl")


# --------------------------------------------------
# Export product and tag lists for use in feature engineering
# --------------------------------------------------

# Flatten the list of comma-delimited products and tags into individual items
flattened_products = [
    item.strip() for sublist in df_pruned["Products"] for item in sublist.split(",")
]
unique_products = set(flattened_products)
unique_products_list = list(unique_products)

flattened_tags = [
    item.strip() for sublist in df_pruned["Tags"] for item in sublist.split(",")
]
unique_tags = set(flattened_tags)
unique_tags_list = list(unique_tags)

pd.DataFrame({"Products": unique_products_list}).to_csv(
    "../../data/interim/DDU Product List.csv"
)
pd.DataFrame({"Tags": unique_tags_list}).to_csv("../../data/interim/DDU Tag List.csv")
