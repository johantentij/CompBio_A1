import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Kinetics.csv")

S2_vals = np.unique(data["S2"])

groups = []
for S2 in S2_vals:
    groups.append(data[data["S2"] == S2])

fig, (ax1, ax2) = plt.subplots(1, 2)

for i, g in enumerate(groups):
    ax1.scatter(g["S1"], g["Rate"], label=str(S2_vals[i]))
    ax2.scatter(1 / g["S1"], 1 / g["Rate"], label=str(S2_vals[i]))

ax1.legend()
ax2.legend()
plt.show()