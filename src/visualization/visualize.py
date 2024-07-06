import pandas as pd
from plotter import HistogramPlotter

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

# Plot a historgram of product_counts with matplotlib. The product index should be the x-axis and the count should be the y-axis
num_products = df["Product_Count"]
HistogramPlotter(
    data=num_products,
    bins=range(1, num_products.max() + 2),
    title="Product Count Histogram",
    xlabel="Number of Products",
    ylabel="Number of Customers With X Number of Products",
    filename="../../reports/figures/num_products.png",
).plot()

# Plot a historgram of tag_counts with matplotlib. The tag index should be the x-axis and the count should be the y-axis
num_tags = df["Tags_Count"]
HistogramPlotter(
    data=num_tags,
    bins=range(1, num_tags.max() + 2),
    title="Tag Count Histogram",
    xlabel="Number of Tags",
    ylabel="Number of Customers With X Number of Tags",
    filename="../../reports/figures/num_tags.png",
    xticks=range(1, num_tags.max() + 2),
).plot()


### FINDING: Number of products and tags need to be normalized using StandardScaler()

# --------------------------------------------------------------
# Plot number of sign-in counts for both members and non-members
# --------------------------------------------------------------

# Plot a historgram of sign-in counts for ALL customers.
# The index should be the x-axis and the count should be the y-axis

HistogramPlotter(
    data=df["Sign In Count"],
    bins=df["Sign In Count"].max(),
    title="Sign-in Count Histogram",
    xlabel="Number of Sign-ins",
    ylabel="Total Number of Customers",
    color="blue",
    figsize=(30, 10),
    filename="../../reports/figures/signin-frequency_all.png",
).plot()

### FINDING: Number of sign-ins is heavily skewed, needs normalization as well

# Now plot a historgram of sign-in counts for members vs non-members
members = df[df["Is_Member"] == 1]["Sign In Count"]
HistogramPlotter(
    data=members,
    title="Sign-in Count Histogram",
    xlabel="Number of Sign-ins",
    ylabel="Number of Members",
    filename="../../reports/figures/signin-frequency_members.png",
    bins=df["Sign In Count"].max(),
    color="green",
    figsize=(30, 10),
    alpha=0.5,
    label="Members",
).plot()

non_members = df[df["Is_Member"] == 0]["Sign In Count"]
HistogramPlotter(
    data=non_members,
    title="Sign-in Count Histogram",
    xlabel="Number of Sign-ins",
    ylabel="Number of Non-members",
    filename="../../reports/figures/signin-frequency_non-members.png",
    bins=df["Sign In Count"].max(),
    color="orange",
    figsize=(30, 10),
    alpha=0.5,
    label="Non-Members",
).plot()

### Member sign-ins are more frequent than non-member sign-ins, which could be valuable
### information for marketing to non-members who sign in more frequently than others
