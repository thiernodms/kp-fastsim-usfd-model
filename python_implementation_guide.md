# Python Implementation Guide for Kp+FASTSIM+USFD

## Introduction

This guide provides a comprehensive approach to implementing the Kp+FASTSIM+USFD combination for wheel-rail contact modeling and wear prediction in railway engineering using Python. This implementation combines three key components:

1. **Kp (Kik-Piotrowski)**: A non-Hertzian contact model for normal contact analysis
2. **FASTSIM**: Kalker's simplified theory algorithm for tangential contact calculation
3. **USFD**: University of Sheffield wear model for predicting material loss

The implementation follows a modular approach, with each component implemented as a separate Python module that can be developed, tested, and integrated together. This guide includes code explanations, usage examples, and integration strategies.

## System Architecture

The overall architecture of the Kp+FASTSIM+USFD Python implementation consists of the following modules:

1. **kp_model.py**: Implements the Kik-Piotrowski non-Hertzian contact model
2. **fastsim.py**: Implements Kalker's FASTSIM algorithm for tangential contact
3. **usfd_wear_model.py**: Implements the USFD wear model for wear prediction
4. **kp_fastsim_usfd_integration.py**: Integrates the three components into a unified framework

The data flow between these modules is sequential:

```
Wheel/Rail Profiles → Kp Model → FASTSIM Algorithm → USFD Wear Model → Results
```

## Implementation Requirements

### Software Requirements

- Python 3.6 or higher
- NumPy: For numerical calculations
- Matplotlib: For visualization
- (Optional) SciPy: For additional mathematical functions

### Installation

To set up the environment for the Kp+FASTSIM+USFD implementation, follow these steps:

1. Ensure Python 3.6+ is installed
2. Install required packages:

```bash
pip install numpy matplotlib scipy
```

3. Download the implementation files:
   - kp_model.py
   - fastsim.py
   - usfd_wear_model.py
   - kp_fastsim_usfd_integration.py

## Component 1: Kp (Kik-Piotrowski) Contact Model

The Kp model is implemented in `kp_model.py` and provides a non-Hertzian contact analysis for wheel-rail interaction.

### Key Classes and Functions

1. **WheelRailProfiles**: Data class for wheel and rail profile coordinates
2. **MaterialProperties**: Data class for material properties (Young's modulus, Poisson's ratio)
3. **KpModel**: Main class implementing the Kik-Piotrowski algorithm

### Usage Example

```python
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
```

### Key Algorithms

The Kp model implementation includes the following key algorithms:

1. **equalize_profiles()**: Ensures wheel and rail profiles have the same number of points
2. **calculate_interpenetration()**: Calculates the overlap between wheel and rail profiles
3. **determine_contact_patch()**: Determines contact patch dimensions based on interpenetration
4. **calculate_pressure_distribution()**: Calculates normal pressure distribution

## Component 2: FASTSIM Algorithm

The FASTSIM algorithm is implemented in `fastsim.py` and calculates tangential forces and creepages in wheel-rail contact.

### Key Classes and Functions

1. **ContactPatch**: Data class for contact patch geometry
2. **CreepageParameters**: Data class for creepage parameters
3. **MaterialParameters**: Data class for material properties
4. **FASTSIM**: Main class implementing Kalker's FASTSIM algorithm

### Usage Example

```python
import numpy as np
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
```

### Key Algorithms

The FASTSIM implementation includes the following key algorithms:

1. **_calculate_kalker_coefficients()**: Calculates Kalker's coefficients based on a/b ratio
2. **_calculate_pressure_distribution()**: Calculates normal pressure distribution (elliptical)
3. **run()**: Main algorithm that processes the contact patch strip by strip

## Component 3: USFD Wear Model

The USFD wear model is implemented in `usfd_wear_model.py` and predicts material loss based on energy dissipation.

### Key Classes and Functions

1. **WearCoefficients**: Data class for wear coefficients in different regimes
2. **USFDWearModel**: Main class implementing the USFD wear model

### Usage Example

```python
from usfd_wear_model import USFDWearModel, WearCoefficients

# Create USFD wear model with default coefficients
usfd_model = USFDWearModel()

# Calculate T-gamma for a sample case
Fx = -2500  # Longitudinal force (N)
Fy = -1200  # Lateral force (N)
gamma_x = 0.001  # Longitudinal creepage
gamma_y = 0.0005  # Lateral creepage

T_gamma = usfd_model.calculate_t_gamma(Fx, Fy, gamma_x, gamma_y)

# Calculate wear rate
wear_rate, regime = usfd_model.calculate_wear_rate(T_gamma)

# Calculate material loss
contact_area = 75  # Contact patch area (mm²)
distance = 1000  # Running distance (m)

material_loss = usfd_model.calculate_material_loss(wear_rate, contact_area, distance)

# Print results
print(f"T-gamma: {T_gamma:.2f} N/mm²")
print(f"Wear regime: {regime} ({'Mild' if regime == 1 else 'Severe' if regime == 2 else 'Catastrophic'})")
print(f"Wear rate: {wear_rate:.2f} μg/m/mm²")
print(f"Material loss: {material_loss:.2f} μg over {distance} m")

# Plot wear function
usfd_model.plot_wear_function()
```

### Key Algorithms

The USFD wear model implementation includes the following key algorithms:

1. **calculate_t_gamma()**: Calculates energy dissipation per unit area
2. **determine_wear_regime()**: Determines wear regime based on T-gamma value
3. **calculate_wear_rate()**: Calculates wear rate based on T-gamma value
4. **calculate_material_loss()**: Calculates total material loss

## Integration of Kp+FASTSIM+USFD

The integration of the three components is implemented in `kp_fastsim_usfd_integration.py`.

### Key Classes and Functions

1. **KpFastsimUsfdIntegration**: Main class that integrates the three components

### Usage Example

```python
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
```

### Integration Workflow

The integration workflow consists of the following steps:

1. **Initialize components**: Create instances of Kp model and USFD wear model
2. **Run Kp model**: Calculate contact patch and normal pressure
3. **Initialize and run FASTSIM**: Use contact patch from Kp model to calculate tangential forces
4. **Calculate wear**: Use forces from FASTSIM to predict wear with USFD model
5. **Compile and visualize results**: Combine results from all components

## Working with Real Wheel-Rail Profiles

In real applications, wheel and rail profiles are typically loaded from measurement files. Here's how to load profiles from files:

```python
import numpy as np

def load_wheel_profile(file_path):
    """Load wheel profile from file."""
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    return data

def load_rail_profile(file_path):
    """Load rail profile from file."""
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    return data

# Load profiles
wheel_profile = load_wheel_profile('wheel_profile.csv')
rail_profile = load_rail_profile('rail_profile.csv')

# Use profiles in the integrated model
integrated_model = KpFastsimUsfdIntegration(
    wheel_profile=wheel_profile,
    rail_profile=rail_profile,
    material_properties=material_properties,
    simulation_params=simulation_params
)
```

## Parametric Studies

The modular implementation allows for easy parametric studies by varying input parameters:

```python
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
```

## Validation and Testing

To ensure the correctness of the implementation, it's important to validate each component and the integrated system against known results.

### Component Testing

Each component includes example usage functions that can be used for basic testing:

```python
# Test Kp model
from kp_model import example_usage as kp_example
kp_example()

# Test FASTSIM
from fastsim import example_usage as fastsim_example
fastsim_example()

# Test USFD wear model
from usfd_wear_model import example_usage as usfd_example
usfd_example()

# Test integrated model
from kp_fastsim_usfd_integration import example_usage as integration_example
integration_example()
```

### Validation Test Cases

For more comprehensive validation, the implementation includes a test script for FASTSIM (`test_fastsim.py`) that tests different scenarios:

1. **Standard wheel-rail contact with moderate creepage**
2. **High creepage leading to full slip condition**
3. **Circular contact patch with equal creepages**

Run the test script to validate the FASTSIM implementation:

```bash
python test_fastsim.py
```

## Performance Optimization

For large-scale simulations, performance optimization is crucial. Here are some strategies:

1. **Vectorization**: Use NumPy's vectorized operations instead of loops where possible
2. **Discretization**: Adjust discretization parameters to balance accuracy and performance
3. **Parallel Processing**: For multiple simulations, use Python's multiprocessing module

Example of parallel processing for parametric studies:

```python
import multiprocessing as mp
from functools import partial

def run_simulation(creep, wheel_profile, rail_profile, material_properties, base_params):
    """Run a single simulation with given creepage."""
    simulation_params = base_params.copy()
    simulation_params['creepages']['xi_x'] = creep
    
    model = KpFastsimUsfdIntegration(
        wheel_profile=wheel_profile,
        rail_profile=rail_profile,
        material_properties=material_properties,
        simulation_params=simulation_params
    )
    
    results = model.run()
    
    if results['status'] == 'success':
        return results['wear_results']['wear_rate']
    else:
        return 0

# Set up parallel processing
creepage_values = np.linspace(0.0001, 0.01, 10)
run_sim_partial = partial(
    run_simulation,
    wheel_profile=wheel_profile,
    rail_profile=rail_profile,
    material_properties=material_properties,
    base_params=simulation_params
)

# Run simulations in parallel
with mp.Pool(processes=mp.cpu_count()) as pool:
    wear_rates = pool.map(run_sim_partial, creepage_values)
```

## Conclusion

This Python implementation of the Kp+FASTSIM+USFD combination provides a comprehensive framework for wheel-rail contact modeling and wear prediction in railway engineering. The modular design allows for flexibility in development, testing, and application.

Key advantages of this implementation:

1. **Pure Python**: Easy to understand, modify, and extend
2. **Modular Design**: Each component can be used independently or together
3. **Visualization**: Built-in plotting functions for results analysis
4. **Flexibility**: Suitable for both simple examples and complex real-world applications

## References

1. Kik, W., & Piotrowski, J. (1996). A fast approximate method to calculate normal load at contact between wheel and rail and creep forces during rolling. In Proceedings of the 2nd mini-conference on contact mechanics and wear of rail/wheel systems.

2. Kalker, J. J. (1982). A fast algorithm for the simplified theory of rolling contact. Vehicle System Dynamics, 11(1), 1-13.

3. Lewis, R., & Dwyer-Joyce, R. S. (2004). Wear mechanisms and transitions in railway wheel steels. Proceedings of the Institution of Mechanical Engineers, Part J: Journal of Engineering Tribology, 218(6), 467-478.

4. Network Rail (2019). Guide to calculating Tgamma values.
