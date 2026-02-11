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

def makePlots():
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for i, g in enumerate(groups):
        ax1.scatter(g["S1"], g["Rate"], label="$S_2=$"+str(S2_vals[i]))
        ax2.scatter(1 / g["S1"], 1 / g["Rate"], label="$S_2=$"+str(S2_vals[i]))

    ax1.set_xlabel("$S_1$")
    ax2.set_ylabel("Rate")
    ax1.legend()
    ax2.legend()
    plt.show()

    return

def sequentialMechanism(S, Vmax, K_ia, K_A, K_B):
    S1, S2 = S
    return Vmax * S1 * S2 / (K_ia * K_B + K_B * S1 + K_A * S2 + S1 * S2)

def bootstrapK_ia(N_bootstrap=1000, plot=False):
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

    if plot:
        plt.hist(K_ia, density=True)
        plt.xlabel("$K_{ia}$")
        plt.ylabel("Density of occurence")
        plt.show()

    print("95% CI K_ia:", np.percentile(K_ia, [2.5, 97.5]))  

    return

def dState_dt(state, params):
    v = sequentialMechanism(state, *params)

    return np.array([-v, -v])

def RK4(state, params, T_max=10, steps=1000):
    dt = T_max / steps
    t = np.arange(steps) * dt

    stateHist = np.empty((steps, 2))
    stateHist[0] = state

    for i in range(1, steps):
        k1 = dState_dt(state, params)
        k2 = dState_dt(state + .5 * k1 * dt, params)
        k3 = dState_dt(state + .5 * k2 * dt, params)
        k4 = dState_dt(state + k3 * dt, params)

        state += (k1 + 2 * (k2 + k3) + k4) * dt / 6

        stateHist[i] = state

    return t, stateHist

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

initState = np.array([
    100 / 150,
    1000
])

targetConcentration = 1 / 150
t, stateHist = RK4(initState, popt, T_max=2, steps=int(1e4))

T_target = t[stateHist[:, 0] < targetConcentration]
T_target = T_target[0]

print("Concentration below 1 g/L at t =", T_target)

plt.plot(t, stateHist[:, 0])
plt.hlines(targetConcentration, np.min(t), np.max(t), linestyles="dashed", color="grey")
plt.xlabel("Time (s)")
plt.ylabel("$S_1$ (mol / L)")
plt.show()