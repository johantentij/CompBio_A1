import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import curve_fit
from scipy.stats import chi2

data = pd.read_csv("Kinetics.csv")

S2_vals = np.unique(data["S2"])

groups = []
for S2 in S2_vals:
    groups.append(data[data["S2"] == S2])

def sequentialMechanism(S, Vmax, K_ia, K_A, K_B):
    S1, S2 = S
    return Vmax * S1 * S2 / (K_ia * K_B + K_B * S1 + K_A * S2 + S1 * S2)

N_bootstrap = 1000
K_ia = []

for i in range(N_bootstrap):
    sample = data.sample(frac=1, replace=True)

    try:
        popt, pcov = curve_fit(
            f=sequentialMechanism,
            xdata=(sample["S1"], sample["S2"]),
            ydata=sample["Rate"],
            p0=[
                sample["Rate"].max(),
                1e-3,
                1.0,
                1.0
            ],
            bounds=([0, 0, 0, 0], np.inf)
        )
    except RuntimeError:
        print("oeps")
        continue

    K_ia.append(popt[1])

plt.hist(K_ia, density=True)
plt.xlabel("$K_{ia}$")
plt.ylabel("Density of occurence")
print("95% CI K_ia:", np.percentile(K_ia, [2.5, 97.5]))

popt, pcov = curve_fit(
    f=sequentialMechanism,
    xdata=(data["S1"], data["S2"]),
    ydata=data["Rate"],
    p0=[
        data["Rate"].max(),
        1e-1,
        1.0,
        1.0
    ]
)

print(popt)

fig, (ax1, ax2) = plt.subplots(1, 2)

for i, g in enumerate(groups):
    fit = sequentialMechanism((g["S1"], g["S2"]), *popt)

    ax1.plot(g["S1"], fit, linestyle="dashed", color="grey")
    ax1.scatter(g["S1"], g["Rate"], label=str(S2_vals[i]))
    ax2.scatter(1 / g["S1"], 1 / g["Rate"], label=str(S2_vals[i]))


ax1.legend()
ax2.legend()
plt.show()