from usfd_wear_model import USFDWearModel, WearCoefficients
import matplotlib.pyplot as plt
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
plt.show()