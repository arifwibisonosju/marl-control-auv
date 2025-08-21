import pandas as pd
import matplotlib.pyplot as plt

# Load the MADDPG eval data
file_path_maddpg = "/mnt/data/eval_maddpg_1755759790.csv"
df_maddpg = pd.read_csv(file_path_maddpg)

# Plot histogram with default style
plt.figure(figsize=(10, 5))
plt.hist(df_maddpg["return"], bins=20)
plt.title("Distribution of Returns (MADDPG)")
plt.xlabel("Return")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
