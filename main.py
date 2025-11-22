import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(42)

T = 50
pre = 20
controls = 10

trend = np.linspace(0, 4, T)
control_data = np.array([
    trend + np.random.normal(0, 0.4, T) + i*0.2
    for i in range(controls)
])

treated = trend + np.random.normal(0, 0.4, T)
treated[20:] += 5

df = pd.DataFrame(control_data.T, columns=[f"C{i}" for i in range(1, controls+1)])
df["Treated"] = treated

X0 = df.iloc[:pre, :-1].values
X1 = df.iloc[:pre, -1].values

def loss(w):
    return np.sum((X1 - X0 @ w)**2)

cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1)] * controls

res = minimize(loss, np.ones(controls)/controls, bounds=bounds, constraints=cons)
w = res.x

synthetic = df.iloc[:, :-1].values @ w
df["Synthetic"] = synthetic
df["ATT"] = df["Treated"] - synthetic

rmse = float(np.sqrt(np.mean((df["Treated"][:pre] - df["Synthetic"][:pre])**2)))
mae = float(np.mean(np.abs(df["Treated"][:pre] - df["Synthetic"][:pre])))

# Sensitivity analysis
perturbed = df.iloc[:, :-1].values * 1.01
synthetic2 = perturbed @ w
ATT2 = df["Treated"].values - synthetic2

df.to_csv("results.csv", index=False)

plt.figure()
plt.plot(df["Treated"], label="Treated")
plt.plot(df["Synthetic"], label="Synthetic Control")
plt.title("Treated vs Synthetic")
plt.legend()
plt.savefig("trend_plot.png")
plt.close()

plt.figure()
plt.plot(df["ATT"], label="Baseline ATT")
plt.plot(ATT2, label="Perturbed ATT", linestyle="--")
plt.title("ATT Sensitivity Analysis")
plt.legend()
plt.savefig("att_plot.png")
plt.close()

with open("metrics.txt", "w") as f:
    f.write(f"RMSE: {rmse}\nMAE: {mae}\nWeights: {w.tolist()}\n")

print("Completed with sensitivity analysis. Files saved.")
