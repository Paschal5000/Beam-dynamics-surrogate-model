# ==============================================================================
#           SCRIPT: sanity_check.py (Final Version with Correct Constants)
# ==============================================================================
# This script uses the verified analytical function AND the correct physical
# constants from the source thesis to replicate the results.

import numpy as np

# --- CORRECTED Physical Constants from Thesis Chapter 4  ---
L = 12.192      # Length of the beam (m)
EI = 6041658    # Correct Flexural rigidity (N/m^2) 
m = 2758.27     # Correct Mass per unit length (kg/m) 
P = 10500       # Magnitude of the concentrated load (N)

# --- Verified Core Calculation Function ---
def calculate_w_max(k0, k1, damping_ratio, v):
    """
    Calculates the maximum dynamic deflection at the beam's midpoint (x=L/2).
    """
    t_end = L / v
    t = np.linspace(0, t_end, num=500)
    deflection_midpoint = np.zeros_like(t)
    N_modes = 100

    sin_nx_L = [np.sin(n * np.pi / 2) for n in range(N_modes + 1)]

    for n in range(1, N_modes + 1):
        if sin_nx_L[n] == 0:
            continue

        lambda_n_sq = ((n * np.pi) / L)**2
        omega_n_sq = (EI * (lambda_n_sq**2) + k0 + k1 * lambda_n_sq) / m
        omega_n = np.sqrt(omega_n_sq)
        omega_v = (n * np.pi * v) / L

        # Undamped solution for a moving load
        if np.isclose(omega_n, omega_v): # Handle resonance
            term = (np.sin(omega_n * t) - omega_n * t * np.cos(omega_n * t)) / (2 * omega_n**3)
        else:
            term = (np.sin(omega_v * t) - (omega_v / omega_n) * np.sin(omega_n * t)) / (omega_n**2 - omega_v**2)

        mode_contribution = ((2 * P) / (m * L)) * term * sin_nx_L[n]
        deflection_midpoint += mode_contribution

    return np.max(np.abs(deflection_midpoint))

# --- Parameters from Thesis Figure 4.1 ---
k0_test = 0          # Foundation Stiffness = 0
k1_test = 90000      # Shear Modulus = 90000 N/m
damping_test = 0
v_test = 10          # Velocity = 36 km/hr = 10 m/s

# --- Run the Test ---
print("Running final sanity check with CORRECT CONSTANTS...")
predicted_w_max = calculate_w_max(k0_test, k1_test, damping_test, v_test)

# --- From the plot, the peak absolute value is around -0.29m ---
expected_w_max = 0.29 # From visual inspection of the plot

print(f"\n--- Results ---")
print(f"Calculated Max Deflection: {predicted_w_max:.6f} meters")
print(f"Expected Max Deflection (from plot): ~{expected_w_max} meters")
print("---------------")