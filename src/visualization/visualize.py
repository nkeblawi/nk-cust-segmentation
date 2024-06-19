import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

"""
    EXPLORATORY DATA ANALYSIS & PLOTTING
    
    Includes a variety of visualization techniques to explore the data.
    The data is loaded from the processed data file and visualized in different ways.
    The plots are saved in the ../../reports/figures/ folder.
    The file naming convention is: '{label}-{participant}.png'

"""

# --------------------------------------------------------------
# Load dataset
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/DDU - Filtered Kajabi Data.pkl")


# --------------------------------------------------------------
# Plot distributions of products and tags each person has
# --------------------------------------------------------------

num_products = df["Product_Count"].value_counts()
num_tags = df["Tags_Count"].value_counts()

# Plot a historgram of product_counts with matplotlib. The product index should be the x-axis and the count should be the y-axis
fig, ax = plt.subplots()
ax.bar(num_products.index, num_products.values)
ax.set_title("Product Count Histrogram")
ax.set_xlabel("Number of Products")
ax.set_ylabel("Number of Customers With X Number of Products")
plt.savefig("../../reports/figures/num_products.png")

# Plot a historgram of tag_counts with matplotlib. The tag index should be the x-axis and the count should be the y-axis
fig, ax = plt.subplots()
ax.bar(num_tags.index, num_tags.values)
ax.set_xticks(range(num_tags.index.min(), num_tags.index.max() + 1, 2))
ax.set_title("Tag Count Histrogram")
ax.set_xlabel("Number of Tags")
ax.set_ylabel("Number of Customers With X Number of Tags")
plt.savefig("../../reports/figures/num_tags.png")

### Number of products and tags need to be normalized using StandardScaler()

# --------------------------------------------------------------
# Plot number of sign-in counts for both members and non-members
# --------------------------------------------------------------

# Plot a historgram of sign-in counts for ALL customers.
# The index should be the x-axis and the count should be the y-axis
df["Sign In Count"].describe()
fig, ax = plt.subplots(figsize=(30, 10))
ax.hist(df["Sign In Count"], bins=df["Sign In Count"].max(), color="blue")
ax.set_title("Sign-in Count Histogram")
ax.set_xlabel("Number of Sign-ins")

### Number of sign-ins is heavily skewed, needs normalization as well

# Now plot a historgram of sign-in counts for members vs non-members
members = df[df["Is_Member"] == 1]["Sign In Count"]
non_members = df[df["Is_Member"] == 0]["Sign In Count"]

# Separate plots for members vs non-members
fig, ax = plt.subplots(figsize=(30, 10))
ax.hist(
    members, bins=df["Sign In Count"].max(), color="green", alpha=0.5, label="Members"
)
ax.set_title("Sign-in Count Histogram")
ax.set_xlabel("Number of Sign-ins")
ax.set_ylabel("Frequency")
ax.legend(loc="upper right")
plt.savefig("../../reports/figures/signin-frequency_members.png")

fig, ax = plt.subplots(figsize=(30, 10))
ax.hist(
    non_members,
    bins=df["Sign In Count"].max(),
    color="orange",
    alpha=0.5,
    label="Non-Members",
)
ax.set_title("Sign-in Count Histogram")
ax.set_xlabel("Number of Sign-ins")
ax.set_ylabel("Frequency")
ax.legend(loc="upper right")
plt.savefig("../../reports/figures/signin-frequency_non-members.png")

### Member sign-ins are more frequent than non-member sign-ins, which could be valuable
### information for marketing to non-members who sign in more frequently than others
