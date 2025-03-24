import numpy as np
from kp_model import KpModel, WheelRailProfiles, MaterialProperties

# Create wheel and rail profiles
# Wheel profile (simplified circular arc)
x_wheel = np.linspace(-0.05, 0.05, 100)
R_wheel = 0.46  # 460 mm wheel radius
y_wheel = np.sqrt(R_wheel**2 - x_wheel**2) - R_wheel + 0.01
wheel_profile = np.column_stack((x_wheel, y_wheel))

# Rail profile (simplified flat surface with rounded corners)
x_rail = np.linspace(-0.05, 0.05, 100)
y_rail = np.zeros_like(x_rail)
corner_indices = np.where(abs(x_rail) > 0.03)[0]
y_rail[corner_indices] = -0.001 * (abs(x_rail[corner_indices]) - 0.03)**2
rail_profile = np.column_stack((x_rail, y_rail))

# Create profiles object
profiles = WheelRailProfiles(
    wheel_profile=wheel_profile,
    rail_profile=rail_profile,
    wheel_radius=R_wheel
)

# Create material properties
material = MaterialProperties(
    E=210e9,  # 210 GPa Young's modulus
    nu=0.28   # Poisson's ratio
)

# Create Kp model
kp_model = KpModel(
    profiles=profiles,
    material=material,
    penetration=0.0001,  # 0.1 mm penetration
    yaw_angle=0.0,
    discretization=100
)

# Run model
results = kp_model.run()

# Print results
print(f"Contact patch dimensions: a = {results['contact_patch']['a']*1000:.2f} mm, b = {results['contact_patch']['b']*1000:.2f} mm")
print(f"Contact patch area: {results['contact_patch']['area']*1e6:.2f} mm²")
print(f"Maximum pressure: {results['max_pressure']/1e6:.2f} MPa")
print(f"Normal force: {results['normal_force']:.2f} N")

# Plot results
kp_model.plot_results(results)