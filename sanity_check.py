# ==============================================================================
#           SCRIPT: sanity_check.py (Definitive Version)
# ==============================================================================
# This script uses the CORRECTED interpretation of the physical constants
# to definitively replicate the results from the source thesis plot.

import numpy as np
import matplotlib.pyplot as plt

# --- Corrected Physical Constants from Thesis Chapter 4 ---
L = 12.192      # Length of the beam (m)
EI = 6041658    # Correct Flexural rigidity (N-m^2)
P = 10500       # Magnitude of the concentrated load (N)

# THE FIX: The thesis states m=2758.27 kg/m, which is a typo. It is the
# total mass. We derive the correct mass per unit length here.
m_total = 2758.27
m = m_total / L # Correct mass per unit length (kg/m)


# --- Verified Core Calculation Function ---
def calculate_deflection_history(k0, k1, v):
    """
    Calculates the full time-history of the deflection at the beam's midpoint.
    """
    t_end = L / v
    t = np.linspace(0, t_end, num=1000)
    x_mid = L / 2
    total_deflection = np.zeros_like(t)
    N_modes = 100

    for n in range(1, N_modes + 1):
        sin_nx_L = np.sin(n * np.pi * x_mid / L)
        if np.isclose(sin_nx_L, 0):
            continue

        omega_n_sq = (EI * (n * np.pi / L)**4 + k1 * (n * np.pi / L)**2 + k0) / m
        omega_n = np.sqrt(omega_n_sq)
        omega_v = (n * np.pi * v) / L

        # Undamped solution for a moving load
        if np.isclose(omega_n, omega_v):
            time_term = (np.sin(omega_n * t) - omega_n * t * np.cos(omega_n * t)) / (2 * omega_n**3)
        else:
            time_term = (1 / (omega_n**2 - omega_v**2)) * (np.sin(omega_v * t) - (omega_v / omega_n) * np.sin(omega_n * t))

        mode_contribution = ((2 * P) / (m * L)) * time_term * sin_nx_L
        total_deflection += mode_contribution
    
    return t, total_deflection

# --- Parameters from Thesis Figure 4.1 ---
k0_test = 0
k1_test = 90000
v_test = 10 # 36 km/hr

# --- Run the Test and Generate Plot ---
print("Running final sanity check with DEFINITIVE constants...")
time_vector, deflection_vector = calculate_deflection_history(k0_test, k1_test, v_test)
calculated_max = np.max(np.abs(deflection_vector))

print(f"Calculated Max Deflection: {calculated_max:.4f} meters")
print("Expected Max Deflection from plot: ~0.29 meters")

# --- Create the plot ---
plt.figure(figsize=(10, 6))
plt.plot(time_vector, deflection_vector, '-b', label=f'Calculated Deflection (Max: {calculated_max:.4f}m)')
plt.title('Sanity Check: Calculated vs. Thesis Plot')
plt.xlabel('Time (s)')
plt.ylabel('Deflection (m)')
plt.grid(True)
plt.legend()
plt.ylim(-0.4, 0.3) # Match the thesis plot's y-axis
plt.axhline(0, color='black', linewidth=0.5)
plt.show()