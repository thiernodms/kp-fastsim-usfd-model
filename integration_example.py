import numpy as np
from kp_fastsim_usfd_integration import KpFastsimUsfdIntegration

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

# Define material properties
material_properties = {
    'E': 210e9,  # 210 GPa Young's modulus
    'nu': 0.28,  # Poisson's ratio
    'G': 80e9    # 80 GPa shear modulus
}

# Define simulation parameters
simulation_params = {
    'wheel_radius': R_wheel,
    'penetration': 0.0001,  # 0.1 mm penetration
    'discretization': 100,
    'fastsim_discretization': (50, 30),
    'friction_coefficient': 0.3,
    'creepages': {
        'xi_x': 0.001,  # Longitudinal creepage
        'xi_y': 0.0005,  # Lateral creepage
        'phi': 0.1  # Spin creepage (1/m)
    },
    'running_distance': 1000  # m
}

# Create integrated model
integrated_model = KpFastsimUsfdIntegration(
    wheel_profile=wheel_profile,
    rail_profile=rail_profile,
    material_properties=material_properties,
    simulation_params=simulation_params
)

# Run simulation
results = integrated_model.run()

if results['status'] == 'success':
    # Print summary results
    print("\nContact Patch Dimensions:")
    print(f"a = {results['kp_results']['contact_patch']['a']*1000:.2f} mm")
    print(f"b = {results['kp_results']['contact_patch']['b']*1000:.2f} mm")
    print(f"Area = {results['kp_results']['contact_patch']['area']*1e6:.2f} mm²")
    
    print("\nNormal Contact:")
    print(f"Normal Force = {results['kp_results']['normal_force']:.2f} N")
    print(f"Maximum Pressure = {results['kp_results']['max_pressure']/1e6:.2f} MPa")
    
    print("\nTangential Contact:")
    print(f"Longitudinal Force (Fx) = {results['fastsim_results']['Fx']:.2f} N")
    print(f"Lateral Force (Fy) = {results['fastsim_results']['Fy']:.2f} N")
    print(f"Spin Moment (Mz) = {results['fastsim_results']['Mz']:.2f} N·m")
    
    print("\nWear Prediction:")
    print(f"T-gamma = {results['wear_results']['T_gamma']:.2f} N/mm²")
    print(f"Wear Regime = {results['wear_results']['wear_regime']}")
    print(f"Wear Rate = {results['wear_results']['wear_rate']:.2f} μg/m/mm²")
    print(f"Material Loss = {results['wear_results']['material_loss']:.2f} μg over {simulation_params['running_distance']} m")
    
    # Plot results
    integrated_model.plot_results()
else:
    print("Simulation failed. No contact detected.")