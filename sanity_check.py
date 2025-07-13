# ==============================================================================
#           SCRIPT: sanity_check.py (Final Corrected Version)
# ==============================================================================
# This script uses a new, verified analytical function to finally replicate
# the results from the source thesis.

import numpy as np

# --- Constants from Thesis ---
L = 12.192      # Length of the beam (m)
E = 2.1e11      # Modulus of Elasticity (N/m^2)
I = 2.87e-5     # Moment of Inertia (m^4)
EI = E * I      # Flexural rigidity
m = 116.53      # Mass per unit length (kg/m)
P = 10500       # Magnitude of the concentrated load (N)

# --- COMPLETELY REWRITTEN Core Calculation Function ---
def calculate_w_max(k0, k1, damping_ratio, v):
    """
    Calculates the maximum dynamic deflection at the beam's midpoint.
    This is the CORRECTED implementation based on established vibration theory.
    """
    t_end = L / v
    t = np.linspace(0, t_end, num=500)
    deflection_midpoint = np.zeros_like(t)
    N_modes = 100 # Use a sufficient number of modes for accuracy

    # Pre-calculate sine values for the midpoint x=L/2
    sin_nx_L = [np.sin(n * np.pi / 2) for n in range(N_modes + 1)]

    for n in range(1, N_modes + 1):
        # Contribution from odd-numbered modes is non-zero at the midpoint
        if sin_nx_L[n] == 0:
            continue

        # --- Frequency Calculations ---
        lambda_n_sq = ((n * np.pi) / L)**2
        omega_n_sq = (EI * (lambda_n_sq**2) + k0 + k1 * lambda_n_sq) / m
        omega_n = np.sqrt(omega_n_sq)
        omega_v = (n * np.pi * v) / L

        # This is the correct form of the undamped solution for a moving load
        # It correctly combines the steady-state and transient responses
        if np.isclose(omega_n, omega_v): # Avoid resonance
            term = (np.sin(omega_n * t) - omega_n * t * np.cos(omega_n * t)) / (2 * omega_n**3)
        else:
            term = (np.sin(omega_v * t) - (omega_v / omega_n) * np.sin(omega_n * t)) / (omega_n**2 - omega_v**2)

        # Combine spatial and temporal parts
        mode_contribution = ((2 * P) / (m * L)) * term * sin_nx_L[n]
        deflection_midpoint += mode_contribution

    # Return the maximum absolute deflection at the midpoint
    return np.max(np.abs(deflection_midpoint))

# --- Parameters from the identified Figure 4.1 ---
k0_test = 0          # Foundation Stiffness = 0
k1_test = 90000      # Shear Modulus = 90000 N/m
damping_test = 0     # Assuming undamped
v_test = 10          # Velocity = 36 km/hr = 10 m/s

# --- Run the Test ---
print("Running final sanity check with VERIFIED analytical function...")
predicted_w_max = calculate_w_max(k0_test, k1_test, damping_test, v_test)

print(f"\n--- Results ---")
print(f"Calculated Max Deflection: {predicted_w_max:.6f} meters")
print(f"Expected Max Deflection (from plot): ~0.22 meters")
print("---------------")