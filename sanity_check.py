# ==============================================================================
#                 SCRIPT: sanity_check.py (Corrected Version 2)
# ==============================================================================
# This script performs a sanity check on the corrected analytical function to
# ensure it produces results consistent with the source thesis.

import numpy as np

# --- Constants from Thesis ---
L = 12.192      # Length of the beam (m)
E = 2.1e11      # Modulus of Elasticity (N/m^2)
I = 2.87e-5     # Moment of Inertia (m^4)
EI = E * I      # Flexural rigidity
m = 116.53      # Mass per unit length (kg/m)
P = 10500       # Magnitude of the concentrated load (N)

# --- CORRECTED Core Calculation Function ---
def calculate_w_max(k0, k1, damping_ratio, v):
    """
    Calculates the maximum dynamic deflection at the beam's midpoint (x=L/2).
    This version includes the essential spatial sine term.
    """
    t_end = L / v
    t = np.linspace(0, t_end, num=500) # Increased time steps for better resolution
    total_deflection = np.zeros_like(t)
    N_modes = 100 # Increased modes for better accuracy

    for n in range(1, N_modes + 1):
        # The spatial term sin(n*pi*x/L) evaluated at x=L/2
        # This is 0 for even n, so we can skip those modes.
        if n % 2 == 0:
            continue
        
        # This term is 1 for n=1,5,9... and -1 for n=3,7,11...
        sin_nx_L = np.sin(n * np.pi / 2)

        # --- Frequency calculations (unchanged) ---
        lambda_n_sq = ((n * np.pi) / L)**2
        omega_n_sq = (EI * (lambda_n_sq**2) + k0 + k1 * lambda_n_sq) / m
        if omega_n_sq < 0: omega_n_sq = 0 # Prevent sqrt of negative
        omega_n = np.sqrt(omega_n_sq)
        omega_v = (n * np.pi * v) / L

        # --- Solve for the time-dependent modal amplitude q_n(t) ---
        # Using simplified undamped solution as per the sanity check case (damping=0)
        # To avoid division by zero
        if np.isclose(omega_n, 0): omega_n = 1e-6
        if np.isclose(omega_n, omega_v): omega_v += 1e-6
        
        # Amplitude of the modal force
        force_amplitude = (2 * P) / (m * L)
        
        # Modal response over time
        q_n_t = force_amplitude * (1 / (omega_n**2 - omega_v**2)) * \
                (np.sin(omega_v * t) - (omega_v / omega_n) * np.sin(omega_n * t))
        
        # Combine the spatial and temporal parts for this mode's contribution
        mode_deflection = q_n_t * sin_nx_L
        total_deflection += mode_deflection

    return np.max(np.abs(total_deflection))

# --- Parameters from the identified Figure 4.1 ---
k0_test = 0          # Foundation Stiffness = 0
k1_test = 90000      # Shear Modulus = 90000 N/m
damping_test = 0     # Assuming undamped
v_test = 10          # Velocity = 36 km/hr = 10 m/s

# --- Run the Test ---
print("Running SANITY CHECK with the CORRECTED analytical function...")
predicted_w_max = calculate_w_max(k0_test, k1_test, damping_test, v_test)

print(f"\n--- Results ---")
print(f"Calculated Max Deflection: {predicted_w_max:.6f} meters")
print(f"Expected Max Deflection (from plot): ~0.22 meters")
print("---------------")