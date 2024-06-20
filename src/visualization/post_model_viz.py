import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display


# --------------------------------------------------------------
# Plot distributions of scaled features AFTER running the KMeans model
# --------------------------------------------------------------

# Load the datasets
df = pd.read_pickle("../../data/interim/DDU - Filtered Kajabi Data.pkl")
df_scaled = pd.read_pickle("../../data/interim/DDU - Scaled Kajabi Data.pkl")

# Plot distributions of scaled features
for column in df_scaled.columns:
    fig, ax = plt.subplots()
    ax.hist(df_scaled[column], bins=20, color="blue", alpha=0.7)
    ax.set_title(f"{column} Histogram (Scaled)")
    ax.set_xlabel(f"Scaled {column}")
    ax.set_ylabel("Frequency")
    plt.savefig(f"../../reports/figures/{column.lower()}_scaled.png")

# Plot a histogram of sign-in counts for members vs non-members (scaled)
members_scaled = df_scaled[df["Is_Member"] == 1]["Log_Sign_In_Count"]
non_members_scaled = df_scaled[df["Is_Member"] == 0]["Log_Sign_In_Count"]

# Separate plots for members vs non-members (scaled)
fig, ax = plt.subplots(figsize=(30, 10))
ax.hist(members_scaled, bins=20, color="green", alpha=0.5, label="Members")
ax.set_title("Sign-in Count Histogram (Scaled) - Members")
ax.set_xlabel("Scaled Number of Sign-ins")
ax.set_ylabel("Frequency")
ax.legend(loc="upper right")
plt.savefig("../../reports/figures/signin-frequency_members_scaled.png")

fig, ax = plt.subplots(figsize=(30, 10))
ax.hist(non_members_scaled, bins=20, color="orange", alpha=0.5, label="Non-Members")
ax.set_title("Sign-in Count Histogram (Scaled) - Non-Members")
ax.set_xlabel("Scaled Number of Sign-ins")
ax.set_ylabel("Frequency")
ax.legend(loc="upper right")
plt.savefig("../../reports/figures/signin-frequency_non-members_scaled.png")
