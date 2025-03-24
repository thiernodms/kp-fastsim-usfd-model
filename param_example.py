import numpy as np
import matplotlib.pyplot as plt
from kp_fastsim_usfd_integration import KpFastsimUsfdIntegration

# Define base parameters
material_properties = {
    'E': 210e9,
    'nu': 0.28,
    'G': 80e9
}

simulation_params = {
    'wheel_radius': 0.46,
    'penetration': 0.0001,
    'discretization': 100,
    'fastsim_discretization': (50, 30),
    'friction_coefficient': 0.3,
    'creepages': {
        'xi_x': 0.001,
        'xi_y': 0.0005,
        'phi': 0.1
    },
    'running_distance': 1000
}

# Create wheel and rail profiles
# (code for creating profiles as shown earlier)

# Perform parametric study on creepage
creepage_values = np.linspace(0.0001, 0.01, 10)
wear_rates = []

for creep in creepage_values:
    # Update creepage
    simulation_params['creepages']['xi_x'] = creep
    
    # Create model
    model = KpFastsimUsfdIntegration(
        wheel_profile=wheel_profile,
        rail_profile=rail_profile,
        material_properties=material_properties,
        simulation_params=simulation_params
    )
    
    # Run simulation
    results = model.run()
    
    if results['status'] == 'success':
        wear_rates.append(results['wear_results']['wear_rate'])
    else:
        wear_rates.append(0)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(creepage_values, wear_rates, 'o-')
plt.xlabel('Longitudinal Creepage')
plt.ylabel('Wear Rate (μg/m/mm²)')
plt.title('Effect of Creepage on Wear Rate')
plt.grid(True)
plt.show()