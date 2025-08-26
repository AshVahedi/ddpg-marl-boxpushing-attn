import numpy as np
import matplotlib.pyplot as plt

# Load the saved weights
weights = np.load("results/weights.npy")  # shape: [episodes, 4]

# Get the last 100 rows
last_100 = weights[-300:]  # shape: [100, 4]

# Transpose so each box in boxplot represents one weight index
# Each column becomes a distribution over 100 episodes
data_for_plot = last_100.T  # shape: [4, 100]

# Create boxplot
plt.figure(figsize=(6, 10))
plt.boxplot(data_for_plot.T, labels=["w1", "w2", "w3", "w4"])
plt.title("Boxplot of Last 100 Weight Values")
plt.xlabel("Weight Index")
plt.ylabel("Weight Value")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/weights_boxplot.png")

plt.show()
