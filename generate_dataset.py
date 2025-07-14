# ==============================================================================
#                      FINAL SCRIPT (v2): generate_dataset.py
# ==============================================================================
# This script generates a high-fidelity dataset for the dynamic response of a
# beam on a Pasternak foundation using a corrected and verified analytical model.

import pandas as pd
import numpy as np
from scipy.stats import qmc
import time

# --- Correct Physical Constants from Thesis Chapter 4 ---
L = 12.192
EI = 6041658
m = 2758.27
P = 10500

# --- FINAL, VERIFIED Core Calculation Function ---
def calculate_w_max(k0, k1, damping_ratio, v):
    """
    Calculates the maximum dynamic deflection at the beam's midpoint (x=L/2).
    """
    t_end = L / v
    t = np.linspace(0, t_end, num=500)
    x_mid = L / 2
    total_deflection = np.zeros_like(t)
    N_modes = 100

    for n in range(1, N_modes + 1):
        sin_nx_L = np.sin(n * np.pi * x_mid / L)
        if sin_nx_L == 0: continue

        omega_n_sq = (EI * (n * np.pi / L)**4 + k1 * (n * np.pi / L)**2 + k0) / m
        omega_n = np.sqrt(omega_n_sq)
        omega_v = (n * np.pi * v) / L

        if np.isclose(omega_n, omega_v):
            time_term = (np.sin(omega_n * t) - omega_n * t * np.cos(omega_n * t)) / (2 * omega_n**3)
        else:
            time_term = (np.sin(omega_v * t) - (omega_v / omega_n) * np.sin(omega_n * t)) / (omega_n**2 - omega_v**2)

        mode_contribution = ((2 * P) / (m * L)) * time_term * sin_nx_L
        total_deflection += mode_contribution

    return np.max(np.abs(total_deflection))

# --- Data Generation Setup ---
# Damping is now removed from the sampling process.
param_limits = {
    'k0': [0, 50e6],
    'k1': [10e3, 500e3],
    'velocity': [10, 80]
}
num_samples = 2000

# Sampler dimension is now 3 (for k0, k1, velocity).
sampler = qmc.LatinHypercube(d=len(param_limits))
sample_points = sampler.random(n=num_samples)
scaled_samples = qmc.scale(sample_points,
                          [lim[0] for lim in param_limits.values()],
                          [lim[1] for lim in param_limits.values()])

# --- Run the Data Generation Loop ---
dataset = []
start_time = time.time()
print(f"Generating {num_samples} data points with the FINAL script...")
for i, params in enumerate(scaled_samples):
    # Damping is now fixed to 0.
    k0, k1, velocity = params
    damping = 0
    
    w_max = calculate_w_max(k0, k1, damping, velocity)
    dataset.append([k0, k1, damping, velocity, w_max])
    if (i + 1) % 100 == 0: print(f"   ...{i + 1}/{num_samples} complete.")

# --- Save the Final Dataset ---
columns = ['k0', 'k1', 'damping', 'velocity', 'w_max']
df = pd.DataFrame(dataset, columns=columns)
df.to_csv('beam_deflection_dataset.csv', index=False)
print(f"\nDataset generation complete in {time.time() - start_time:.2f} seconds.")
print("\n--- New Dataset Statistics ---")
print(df['w_max'].describe())