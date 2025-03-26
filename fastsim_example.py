import numpy as np
import matplotlib.pyplot as plt
from fastsim import FASTSIM, ContactPatch, CreepageParameters, MaterialParameters

# Define contact patch
contact_patch = ContactPatch(
    a=0.006,  # 6 mm semi-axis in rolling direction
    b=0.004,  # 4 mm semi-axis in lateral direction
    area=np.pi * 0.006 * 0.004,  # Elliptical area
    normal_force=10000  # 10 kN normal force
)

# Define creepage parameters
creepage = CreepageParameters(
    xi_x=0.001,  # 0.1% longitudinal creepage
    xi_y=0.0005,  # 0.05% lateral creepage
    phi=0.1  # 0.1 rad/m spin creepage
)

# Define material parameters
material = MaterialParameters(
    G=8.4e10,  # 84 GPa shear modulus
    poisson=0.28,  # Poisson's ratio
    mu=0.3  # Friction coefficient
)

# Create FASTSIM instance
fastsim = FASTSIM(contact_patch, creepage, material, discretization=(50, 30))

# Run algorithm
results = fastsim.run()

# Print results
print(f"Longitudinal force (Fx): {results['Fx']:.2f} N")
print(f"Lateral force (Fy): {results['Fy']:.2f} N")
print(f"Spin moment (Mz): {results['Mz']:.2f} N·m")

# Plot results
fastsim.plot_results()
plt.show()