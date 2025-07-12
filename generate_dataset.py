# This script generates a dataset for the dynamic response of an
# Euler-Bernoulli beam on a Pasternak foundation. It solves the analytical
# equations from Attama (2025) for a wide range of physical parameters and
# saves the results to a CSV file for machine learning model training.


import pandas as pd
import numpy as np
from scipy.stats import qmc
import time

# --- Constants from Thesis Table 4.1 ---
L = 12.192      # Length of the beam (m)
E = 2.1e11      # Modulus of Elasticity (N/m^2)
I = 2.87e-5     # Moment of Inertia (m^4)
EI = E * I      # Flexural rigidity
m = 116.53      # Mass per unit length (kg/m)
P = 10500       # Magnitude of the concentrated load (N)
g = 9.81        # Acceleration due to gravity (m/s^2)

# --- The Core Calculation Function (from Action 1.2) ---
def calculate_w_max(k0, k1, damping_ratio, v):
    t_end = L / v
    t = np.linspace(0, t_end, num=200)
    total_deflection = np.zeros_like(t)
    N_modes = 50 
    
    for n in range(1, N_modes + 1):
        lambda_n_sq = ((n * np.pi) / L)**2
        omega_n_sq = (EI * (lambda_n_sq**2) + k0 + k1 * lambda_n_sq) / m
        omega_n = np.sqrt(omega_n_sq)
        omega_d = omega_n * np.sqrt(1 - damping_ratio**2)
        omega_v = (n * np.pi * v) / L
        
        if np.isclose(omega_d, omega_v): omega_v += 1e-6
        
        term1 = (1 / (omega_n_sq - omega_v**2)) * np.sin(omega_v * t)
        term2 = (omega_v / (omega_d * (omega_n_sq - omega_v**2))) * \
                np.exp(-damping_ratio * omega_n * t) * np.sin(omega_d * t)
        
        mode_deflection = (2 * P / (m * L)) * (term1 - term2)
        total_deflection += mode_deflection

    return np.max(np.abs(total_deflection))

# --- Data Generation Setup ---
param_limits = {
    'k0': [1e6, 50e6],        # REDUCED: Lower and Wider Foundation Stiffness range
    'k1': [10e3, 200e3],        # Shear Modulus
    'damping': [0.01, 0.20],    # Damping Ratio (1% to 20%)
    'velocity': [10, 80]        # Load Velocity (m/s)
}

num_samples = 2000 # Let's start with 2000 to make it faster. Can increase later.
sampler = qmc.LatinHypercube(d=len(param_limits))
sample_points = sampler.random(n=num_samples)
scaled_samples = qmc.scale(sample_points, 
                          [lim[0] for lim in param_limits.values()], 
                          [lim[1] for lim in param_limits.values()])

# --- Run the Data Generation Loop ---
dataset = []
start_time = time.time()
print(f"Generating {num_samples} data points...")

for i, params in enumerate(scaled_samples):
    k0, k1, damping, velocity = params
    w_max = calculate_w_max(k0, k1, damping, velocity)
    dataset.append([k0, k1, damping, velocity, w_max])
    
    if (i + 1) % 100 == 0:
        print(f"   ...{i + 1}/{num_samples} complete.")

# --- Save the Final Dataset ---
columns = ['k0', 'k1', 'damping', 'velocity', 'w_max']
df = pd.DataFrame(dataset, columns=columns)
df.to_csv('beam_deflection_dataset.csv', index=False)

end_time = time.time()
print("\nDataset generation complete.")
print(f"Total time taken: {end_time - start_time:.2f} seconds.")
print(f"File 'beam_deflection_dataset.csv' saved with {len(df)} samples.")
print("\nFirst 5 rows of the dataset:")
print(df.head())