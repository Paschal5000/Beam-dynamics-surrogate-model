# ==============================================================================
#                      SCRIPT: sanity_check.py (Updated)
# ==============================================================================
# This script performs a sanity check on the analytical function to ensure
# it produces results consistent with the source thesis, using parameters
# from the user-identified Figure 4.1.

import numpy as np

# --- Constants from Thesis ---
L = 12.192      # Length of the beam (m)
E = 2.1e11      # Modulus of Elasticity (N/m^2)
I = 2.87e-5     # Moment of Inertia (m^4)
EI = E * I      # Flexural rigidity
m = 116.53      # Mass per unit length (kg/m)
P = 10500       # Magnitude of the concentrated load (N)

# --- The Core Calculation Function ---
def calculate_w_max(k0, k1, damping_ratio, v):
    t_end = L / v
    t = np.linspace(0, t_end, num=200)
    total_deflection = np.zeros_like(t)
    N_modes = 50

    for n in range(1, N_modes + 1):
        lambda_n_sq = ((n * np.pi) / L)**2
        # Check if damping is near zero to avoid division by zero in omega_d
        if np.isclose(damping_ratio, 0):
             omega_n_sq = (EI * (lambda_n_sq**2) + k0 + k1 * lambda_n_sq) / m
             omega_n = np.sqrt(omega_n_sq)
             omega_d = omega_n # For undamped case, omega_d = omega_n
        else:
            omega_n_sq = (EI * (lambda_n_sq**2) + k0 + k1 * lambda_n_sq) / m
            omega_n = np.sqrt(omega_n_sq)
            omega_d = omega_n * np.sqrt(1 - damping_ratio**2)
        
        omega_v = (n * np.pi * v) / L

        # To prevent division by zero
        if np.isclose(omega_n_sq, omega_v**2): omega_n_sq += 1e-6
        if np.isclose(omega_d, 0): omega_d = 1e-6

        # Simplified solution for undamped case if damping_ratio is zero
        if np.isclose(damping_ratio, 0):
            term1 = (1 / (omega_n_sq - omega_v**2)) * np.sin(omega_v * t)
            term2 = (omega_v / (omega_n * (omega_n_sq - omega_v**2))) * np.sin(omega_n * t)
        else:
            term1 = (1 / (omega_n_sq - omega_v**2)) * np.sin(omega_v * t)
            term2 = (omega_v / (omega_d * (omega_n_sq - omega_v**2))) * \
                    np.exp(-damping_ratio * omega_n * t) * np.sin(omega_d * t)

        mode_deflection = (2 * P / (m * L)) * (term1 - term2)
        total_deflection += mode_deflection

    return np.max(np.abs(total_deflection))

# --- Parameters from the identified Figure 4.1 ---
k0_test = 0          # Foundation Stiffness = 0
k1_test = 90000      # Shear Modulus = 90000 N/m
damping_test = 0     # Assuming undamped as it's not specified
v_test = 10          # Velocity = 36 km/hr = 10 m/s

# --- Run the Test ---
print("Running sanity check with parameters from the identified Figure 4.1...")
predicted_w_max = calculate_w_max(k0_test, k1_test, damping_test, v_test)

print(f"\n--- Results ---")
print(f"Calculated Max Deflection: {predicted_w_max:.6f} m"eters")
print(f"Now, compare this value to the peak deflection shown on the plot in the thesis.")
print("------------------")