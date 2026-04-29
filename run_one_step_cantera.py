import numpy as np

# - equation: H + O2 <=> O + OH  # Reaction 1
#   rate-constant: {A: 1.0399e+14, b: 0.0, Ea: 1.531e+04}

A = 1.0399e+14
b = 0.0
Ea = 1.531e+04
R = 8.314  # J/(mol*K)
T = 1200  # K

k_f = A * (T ** b) * np.exp(-Ea / (R * T))