import numpy as np
from models.parameter_identification import identify_parameters

# Your time series data
T_in = np.array([...])  # Indoor Temperature
T_set = np.array([...])  # Set Temperature
T_out = np.array([...])  # Outdoor Temperature

# Identify parameters
a, b = identify_parameters(T_in, T_set, T_out)
print(f"Identified parameters: \n a = {a:.4f}, \n b = {b:.4f}")


