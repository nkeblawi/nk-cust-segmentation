import pandas as pd
from plotter import HistogramPlotter


# --------------------------------------------------------------
# Plot distributions of scaled features AFTER running the KMeans model
# --------------------------------------------------------------

# Load the datasets
df = pd.read_pickle("../../data/interim/DDU - Filtered Kajabi Data.pkl")
df_scaled = pd.read_pickle("../../data/interim/DDU - Scaled Kajabi Data.pkl")

# Plot distributions of scaled features
for column in df_scaled.columns:
    HistogramPlotter(
        data=df_scaled[column],
        title=f"{column} Histogram (Scaled)",
        xlabel=f"Scaled {column}",
        ylabel="Frequency",
        filename=f"../../reports/figures/{column.lower()}_scaled.png",
        bins=20,
        color="blue",
        figsize=(10, 6),  # Adjust the figsize as needed
        alpha=0.7,
    ).plot()
