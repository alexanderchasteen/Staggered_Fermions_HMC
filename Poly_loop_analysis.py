import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Configuration & Parameters ---
N_T = 6.0
target_coupling = 4 # Select coupling_idx to analyze
max_r = 8.0           # Truncate fit window at r <= 5 (r*T <= 0.625)

# --- Load data using Pandas ---
df = pd.read_csv("Poly_Corr_Real_Data.csv")

# Extract distance columns (r_1 to r_7) and parse distance values r = [1.0, 2.0, ...]
r_cols = [c for c in df.columns if c.startswith('r_')]
r = np.array([float(c.split('_')[1]) for c in r_cols])

# Filter configurations for the selected coupling index
coupling_df = df[df['coupling_idx'] == target_coupling]

if coupling_df.empty:
    raise ValueError(f"coupling_idx {target_coupling} not found in the dataset.")

configs = coupling_df[r_cols].values /3.0  # Shape: (N_cfg, N_r)
N_cfg, N_r = configs.shape

# --- Compute mean Polyakov loop across sweeps ---
means = np.mean(configs, axis=0)

# Mask non-positive values AND enforce distance truncation r <= max_r
valid_mask = (means > 0) & (r > 0) & (r <= max_r)

if np.sum(valid_mask) < 3:
    raise ValueError(
        f"Fewer than 3 valid points for coupling_idx={target_coupling} with r <= {max_r}. "
        "Need at least 3 points to fit Cornell parameters (A, B, sigma)."
    )

r_fit = r[valid_mask]
means_fit = means[valid_mask]
configs_fit = configs[:, valid_mask]

# --- Option A: Linear Propagation of Standard Error ---
P_sem = np.std(configs_fit, axis=0, ddof=1) / np.sqrt(N_cfg)
V = -np.log(means_fit) / N_T
V_err = P_sem / (N_T * means_fit)

# --- Option B: Bootstrap Resampling ---
N_boot = 500
boot_V = np.zeros((N_boot, len(r_fit)))
for b in range(N_boot):
    idx = np.random.choice(N_cfg, size=N_cfg, replace=True)
    boot_means = np.mean(configs_fit[idx, :], axis=0)
    boot_V[b, :] = -np.log(boot_means) / N_T

V_err_boot = np.std(boot_V, axis=0, ddof=1)

# --- Cornell potential model ---
def potential(r, A, B, sigma):
    return A + B / r + sigma * r

# --- Fit using Bootstrap errors ---
params, cov = curve_fit(potential, r_fit, V, sigma=V_err_boot, absolute_sigma=True)
A_fit, B_fit, sigma_fit = params
A_err, B_err, sigma_err = np.sqrt(np.diag(cov))

print(f"Fitted parameters for coupling_idx = {target_coupling} (r <= {max_r}):")
print(f"  A      = {A_fit:.4f} ± {A_err:.4f}")
print(f"  B      = {B_fit:.4f} ± {B_err:.4f}")
print(f"  sigma  = {sigma_fit:.4f} ± {sigma_err:.4f}")

# --- Continuous curve for plot ---
r_dense = np.linspace(r_fit.min(), r_fit.max(), 200)
V_dense = potential(r_dense, *params)

# --- Plot: Potential with Error Bars ---
plt.figure(figsize=(7, 4))
plt.title(f"Polyakov Loop Potential (Cornell Fit, coupling_idx={target_coupling}, r <= {max_r})")
plt.errorbar(r_fit, V, yerr=V_err_boot, fmt='o', markersize=4, capsize=3,
             alpha=0.8, label="Data (Bootstrap SEM)")
plt.plot(r_dense, V_dense, "-", lw=2, label=f"Fit (σ = {sigma_fit:.4f})")
plt.xlabel("Distance r (lattice units)")
plt.ylabel("Potential V(r)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# --- Save complete potential data ---
output_file = f"Polyakov_Potential_coupling_{target_coupling}_rmax_{int(max_r)}.txt"
output = np.column_stack((r_fit, V, V_err_boot))
np.savetxt(output_file, output,
           header="r             V(r)          V_err",
           fmt="%.8f")

print(f"\nSaved processed potential data to '{output_file}'.")