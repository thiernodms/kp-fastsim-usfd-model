"""
Kp (Kik-Piotrowski) contact model implementation in Python.
This module implements the non-Hertzian contact model for wheel-rail applications.

References:
- Piotrowski, J., & Kik, W. (2008). A simplified model of wheel/rail contact mechanics for
  non-Hertzian problems and its application in rail vehicle dynamic simulations.
  Vehicle System Dynamics, 46(1-2), 27-48.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class WheelRailProfiles:
    """Class representing wheel and rail profiles."""
    wheel_profile: np.ndarray  # Array of [x, y] coordinates for wheel profile
    rail_profile: np.ndarray   # Array of [x, y] coordinates for rail profile
    wheel_radius: float        # Nominal wheel rolling radius (m)


@dataclass
class MaterialProperties:
    """Class representing material properties."""
    E: float      # Young's modulus (N/m²)
    nu: float     # Poisson's ratio
    G: float = None  # Shear modulus (N/m²), calculated if not provided
    
    def __post_init__(self):
        """Calculate shear modulus if not provided."""
        if self.G is None:
            self.G = self.E / (2 * (1 + self.nu))


class KpModel:
    """
    Implementation of the Kik-Piotrowski non-Hertzian contact model.
    """
    
    def __init__(self, profiles, material, penetration, yaw_angle=0.0, discretization=100):
        """
        Initialize the Kp model.
        
        Parameters:
        profiles -- WheelRailProfiles object with wheel and rail geometry
        material -- MaterialProperties object with material properties
        penetration -- Penetration depth (m)
        yaw_angle -- Yaw angle (rad)
        discretization -- Number of points for discretization
        """
        self.profiles = profiles
        self.material = material
        self.penetration = penetration
        self.yaw_angle = yaw_angle
        self.discretization = discretization
        
        # Results
        self.contact_patch = None
        self.max_pressure = None
        self.pressure_distribution = None
        self.normal_force = None
        
    def equalize_profiles(self):
        """
        Ensure wheel and rail profiles have the same number of points
        and are properly aligned for comparison.
        """
        # This is a simplified implementation
        # In a real application, this would involve interpolation and alignment
        
        wheel = self.profiles.wheel_profile
        rail = self.profiles.rail_profile
        
        # Create common x-coordinates for both profiles
        x_min = max(np.min(wheel[:, 0]), np.min(rail[:, 0]))
        x_max = min(np.max(wheel[:, 0]), np.max(rail[:, 0]))
        
        x_common = np.linspace(x_min, x_max, self.discretization)
        
        # Interpolate wheel and rail profiles to common x-coordinates
        wheel_y = np.interp(x_common, wheel[:, 0], wheel[:, 1])
        rail_y = np.interp(x_common, rail[:, 0], rail[:, 1])
        
        # Create new profile arrays
        wheel_profile_eq = np.column_stack((x_common, wheel_y))
        rail_profile_eq = np.column_stack((x_common, rail_y))
        
        return wheel_profile_eq, rail_profile_eq
    
    def calculate_interpenetration(self, wheel_profile, rail_profile):
        """
        Calculate interpenetration between wheel and rail profiles.
        
        Parameters:
        wheel_profile -- Equalized wheel profile
        rail_profile -- Equalized rail profile
        
        Returns:
        interpenetration -- Array of interpenetration values
        """
        # Calculate vertical distance between profiles
        interpenetration = wheel_profile[:, 1] - rail_profile[:, 1]
        
        # Apply global penetration
        interpenetration += self.penetration
        
        # Set negative values (no contact) to zero
        interpenetration = np.maximum(interpenetration, 0)
        
        return interpenetration
    
    def determine_contact_patch(self, x_coords, interpenetration):
        """
        Determine contact patch dimensions based on interpenetration.
        
        Parameters:
        x_coords -- x-coordinates of profiles
        interpenetration -- Array of interpenetration values
        
        Returns:
        contact_patch -- Dictionary with contact patch information
        """
        # Find contact points (where interpenetration > 0)
        contact_indices = np.where(interpenetration > 0)[0]
        
        if len(contact_indices) == 0:
            return None
        
        # Get x-coordinates of contact points
        contact_x = x_coords[contact_indices]
        
        # Calculate contact patch dimensions
        x_min = np.min(contact_x)
        x_max = np.max(contact_x)
        
        # Calculate semi-axis in rolling direction (a)
        a = (x_max - x_min) / 2
        
        # Calculate semi-axis in lateral direction (b) using Kik-Piotrowski formula
        # This is a simplified approximation
        max_penetration = np.max(interpenetration)
        b = np.sqrt(4 * self.profiles.wheel_radius * max_penetration)
        
        # Calculate contact patch area
        area = np.pi * a * b
        
        # Create contact patch dictionary
        contact_patch = {
            'a': a,
            'b': b,
            'area': area,
            'center_x': (x_max + x_min) / 2,
            'max_penetration': max_penetration,
            'contact_indices': contact_indices
        }
        
        return contact_patch
    
    def calculate_pressure_distribution(self, x_coords, interpenetration, contact_patch):
        """
        Calculate pressure distribution based on interpenetration.
        
        Parameters:
        x_coords -- x-coordinates of profiles
        interpenetration -- Array of interpenetration values
        contact_patch -- Dictionary with contact patch information
        
        Returns:
        pressure -- Array of pressure values
        max_pressure -- Maximum pressure value
        normal_force -- Total normal force
        """
        # Initialize pressure array
        pressure = np.zeros_like(interpenetration)
        
        # Calculate pressure only for contact points
        contact_indices = contact_patch['contact_indices']
        
        # Calculate effective radius
        R_eff = self.profiles.wheel_radius
        
        # Calculate maximum pressure using Kik-Piotrowski formula
        E_star = self.material.E / (1 - self.material.nu**2)
        max_pressure = E_star * np.sqrt(contact_patch['max_penetration'] / R_eff)
        
        # Calculate semi-elliptical pressure distribution
        for i in contact_indices:
            x_rel = (x_coords[i] - contact_patch['center_x']) / contact_patch['a']
            
            # Semi-elliptical distribution in x-direction
            if abs(x_rel) <= 1:
                pressure[i] = max_pressure * np.sqrt(1 - x_rel**2)
        
        # Calculate normal force by integrating pressure over contact area
        # This is a simplified calculation
        normal_force = np.sum(pressure) * (x_coords[1] - x_coords[0]) * 2 * contact_patch['b']
        
        return pressure, max_pressure, normal_force
    
    def run(self):
        """
        Run the Kp model to calculate contact patch and pressure distribution.
        
        Returns:
        results -- Dictionary with contact results
        """
        # Step 1: Equalize profiles
        wheel_profile_eq, rail_profile_eq = self.equalize_profiles()
        
        # Step 2: Calculate interpenetration
        interpenetration = self.calculate_interpenetration(wheel_profile_eq, rail_profile_eq)
        
        # Step 3: Determine contact patch
        x_coords = wheel_profile_eq[:, 0]
        self.contact_patch = self.determine_contact_patch(x_coords, interpenetration)
        
        if self.contact_patch is None:
            return {
                'contact_patch': None,
                'max_pressure': 0,
                'normal_force': 0,
                'pressure_distribution': np.zeros_like(x_coords)
            }
        
        # Step 4: Calculate pressure distribution
        self.pressure_distribution, self.max_pressure, self.normal_force = \
            self.calculate_pressure_distribution(x_coords, interpenetration, self.contact_patch)
        
        # Update contact patch with normal force
        self.contact_patch['normal_force'] = self.normal_force
        
        # Return results
        return {
            'contact_patch': self.contact_patch,
            'max_pressure': self.max_pressure,
            'normal_force': self.normal_force,
            'pressure_distribution': self.pressure_distribution,
            'x_coords': x_coords,
            'interpenetration': interpenetration
        }
    
    def plot_results(self, results, save_path=None):
        """
        Plot the results of the Kp model.
        
        Parameters:
        results -- Dictionary with contact results
        save_path -- Path to save the figure (optional)
        """
        if results['contact_patch'] is None:
            print("No contact detected. Nothing to plot.")
            return
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot profiles and interpenetration
        axs[0].plot(results['x_coords'], self.profiles.wheel_profile[:, 1], 'b-', label='Wheel')
        axs[0].plot(results['x_coords'], self.profiles.rail_profile[:, 1], 'r-', label='Rail')
        axs[0].set_title('Wheel-Rail Profiles')
        axs[0].set_xlabel('Lateral position (m)')
        axs[0].set_ylabel('Vertical position (m)')
        axs[0].legend()
        
        # Plot pressure distribution
        axs[1].plot(results['x_coords'], results['pressure_distribution'], 'g-')
        axs[1].set_title('Pressure Distribution')
        axs[1].set_xlabel('Lateral position (m)')
        axs[1].set_ylabel('Pressure (Pa)')
        axs[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig


def example_usage():
    """Example usage of the Kp model."""
    # Create simplified wheel and rail profiles
    # In a real application, these would be loaded from files
    
    # Wheel profile (simplified circular arc)
    x_wheel = np.linspace(-0.05, 0.05, 100)
    R_wheel = 0.46  # 460 mm wheel radius
    y_wheel = np.sqrt(R_wheel**2 - x_wheel**2) - R_wheel + 0.01  # Vertical offset
    wheel_profile = np.column_stack((x_wheel, y_wheel))
    
    # Rail profile (simplified flat surface with rounded corners)
    x_rail = np.linspace(-0.05, 0.05, 100)
    y_rail = np.zeros_like(x_rail)
    # Add rounded corners
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
    plt.show()


if __name__ == "__main__":
    example_usage()
