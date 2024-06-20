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

# --------------------------------------------------
# Tag customers who have active memberships with DDU
# --------------------------------------------------

# First, impute missing values in 'Products' and 'Tags' columns
df_pruned["Products"] = df_pruned["Products"].fillna("No Product")
df_pruned["Tags"] = df_pruned["Tags"].fillna("No Tag")

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
