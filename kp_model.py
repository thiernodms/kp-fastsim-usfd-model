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
from scipy import interpolate, integrate


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
    
    def __init__(self, profiles, material, penetration, virtual_penetration=None, yaw_angle=0.0, discretization=100):
        """
        Initialize the Kp model.
        
        Parameters:
        profiles -- WheelRailProfiles object with wheel and rail geometry
        material -- MaterialProperties object with material properties
        penetration -- Penetration depth (m)
        virtual_penetration -- Virtual penetration depth (m), if None, equals to penetration
        yaw_angle -- Yaw angle (rad)
        discretization -- Number of points for discretization
        """
        self.profiles = profiles
        self.material = material
        self.penetration = penetration
        self.virtual_penetration = virtual_penetration if virtual_penetration is not None else penetration
        self.yaw_angle = yaw_angle
        self.discretization = discretization
        
        # Results
        self.contact_patch = None
        self.max_pressure = None
        self.pressure_distribution = None
        self.normal_force = None
        
    def get_profiles(self):
        """
        Get wheel and rail profiles.
        
        Returns:
        wheel_profile, rail_profile -- Arrays of wheel and rail profile coordinates
        """
        return self.profiles.wheel_profile, self.profiles.rail_profile
    
    def equalize_profiles(self):
        """
        Ensure wheel and rail profiles have the same number of points
        and are properly aligned for comparison.
        
        Returns:
        wheel_profile_eq, rail_profile_eq -- Equalized wheel and rail profiles
        """
        wheel, rail = self.get_profiles()
        
        # Create common x-coordinates for both profiles
        x_min = max(np.min(wheel[:, 0]), np.min(rail[:, 0]))
        x_max = min(np.max(wheel[:, 0]), np.max(rail[:, 0]))
        
        x_common = np.linspace(x_min, x_max, self.discretization)
        
        # Interpolate wheel and rail profiles to common x-coordinates
        # Use scipy.interpolate for more accurate interpolation
        wheel_interp = interpolate.interp1d(wheel[:, 0], wheel[:, 1], kind='linear')
        rail_interp = interpolate.interp1d(rail[:, 0], rail[:, 1], kind='linear')
        
        wheel_y = wheel_interp(x_common)
        rail_y = rail_interp(x_common)
        
        # Create new profile arrays
        wheel_profile_eq = np.column_stack((x_common, wheel_y))
        rail_profile_eq = np.column_stack((x_common, rail_y))
        
        return wheel_profile_eq, rail_profile_eq
    
    def separation_of_profiles(self, wheel_profile, rail_profile):
        """
        Compute distance between points of two profiles.
        
        Parameters:
        wheel_profile -- Wheel profile coordinates
        rail_profile -- Rail profile coordinates
        
        Returns:
        separation -- Array of separation values
        """
        # Calculate vertical distance between profiles
        sep = wheel_profile[:, 1] - rail_profile[:, 1]
        
        # Correct separation if profiles overlap or do not touch
        min_sep = np.min(sep)
        
        return sep - min_sep
    
    def calculate_interpenetration(self, wheel_profile, rail_profile):
        """
        Calculate interpenetration between wheel and rail profiles.
        
        Parameters:
        wheel_profile -- Equalized wheel profile
        rail_profile -- Equalized rail profile
        
        Returns:
        interpenetration -- Array of interpenetration values
        """
        # Calculate separation between profiles
        sep = self.separation_of_profiles(wheel_profile, rail_profile)
        
        # Calculate interpenetration
        interp_array = self.virtual_penetration - sep
        
        # Set negative values (no contact) to zero
        for i, interp in enumerate(interp_array):
            if interp > 0:
                interp_array[i] = interp
            else:
                interp_array[i] = 0
        
        return interp_array
    
    def nonzero_runs(self, a):
        """
        Returns (n,2) array where n is number of runs of non-zeros.
        The first column is the index of the first non-zero in each run,
        and the second is the index of the first zero element after the run.
        
        Parameters:
        a -- 1d array
        
        Returns:
        ranges -- Array of run ranges
        """
        # Create an array that's 1 where a isn't 0, and pad each end with an extra 0
        nonzero = np.concatenate(([0], np.not_equal(a, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(nonzero))  # Calculate a[n+1] - a[n] for all n
        
        # Runs start and end where absdiff is 1
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges
    
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
        
        # Find runs of non-zero interpenetration
        runs = self.nonzero_runs(interpenetration)
        
        # Create contact patches for each run
        contact_patches = []
        
        for run_start, run_end in runs:
            # Calculate contact patch dimensions for this run
            x_min = x_coords[run_start]
            x_max = x_coords[run_end - 1]
            
            # Calculate semi-axis in rolling direction (a)
            # Ensure a minimum value to avoid division by zero
            a = max((x_max - x_min) / 2, 1e-6)  # Minimum value of 1 micron
            
            # Calculate center of contact patch
            center_x = (x_max + x_min) / 2
            
            # Get maximum penetration in this run
            max_penetration = np.max(interpenetration[run_start:run_end])
            
            # Calculate semi-axis in lateral direction (b) using Kik-Piotrowski formula
            b = np.sqrt(4 * self.profiles.wheel_radius * max_penetration)
            
            # Calculate contact patch area
            area = np.pi * a * b
            
            # Create contact patch dictionary
            contact_patch = {
                'a': a,
                'b': b,
                'area': area,
                'center_x': center_x,
                'max_penetration': max_penetration,
                'contact_indices': np.arange(run_start, run_end),
                'run_start': run_start,
                'run_end': run_end
            }
            
            contact_patches.append(contact_patch)
        
        # For simplicity, return the patch with the largest penetration
        if contact_patches:
            return max(contact_patches, key=lambda x: x['max_penetration'])
        else:
            return None
    
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
        
        # Get contact patch information
        run_start = contact_patch['run_start']
        run_end = contact_patch['run_end']
        max_penetration = contact_patch['max_penetration']
        a = contact_patch['a']  # Semi-axis in rolling direction
        
        # Convert units for pressure calculation
        # Reference implementation uses mm, our implementation uses m
        # Convert wheel_radius from m to mm for consistency with reference
        wheel_radius_mm = self.profiles.wheel_radius * 1000.0
        # Convert penetration from m to mm
        penetration_mm = self.penetration * 1000.0
        # Convert max_penetration from m to mm
        max_penetration_mm = max_penetration * 1000.0
        
        # Calculate coefficient for pressure calculation
        # Using the formula from the reference implementation
        # E is in MPa in the reference implementation, convert from Pa to MPa
        E_MPa = self.material.E / 1e6
        
        # Calculate maximum pressure using the formula from the reference implementation
        # This is based on equation 13 in the original article
        # The result will be in MPa
        max_pressure_MPa = 0.5 * np.pi * E_MPa * penetration_mm / (1.0 - self.material.nu**2) * np.sqrt(max_penetration_mm / wheel_radius_mm)
        
        # Convert max_pressure from MPa to Pa for consistency with our implementation
        max_pressure = max_pressure_MPa * 1e6
        
        # Calculate semi-elliptical pressure distribution
        for i in range(run_start, run_end):
            if interpenetration[i] > 0:
                # Calculate relative position in contact patch
                # Avoid division by zero by ensuring a is not too small
                if a > 1e-6:  # Check if a is large enough to avoid division issues
                    x_rel = (x_coords[i] - contact_patch['center_x']) / a
                    
                    # Semi-elliptical distribution in x-direction
                    if abs(x_rel) <= 1:
                        pressure[i] = max_pressure * np.sqrt(1 - x_rel**2)
                else:
                    # If a is too small, use a simplified approach
                    # Just set the pressure at this point to the maximum pressure
                    pressure[i] = max_pressure
        
        # Calculate normal force by integrating pressure over contact area
        dx = x_coords[1] - x_coords[0]
        normal_force = np.sum(pressure) * dx * 2 * contact_patch['b']
        
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
            'interpenetration': interpenetration,
            'wheel_profile_eq': wheel_profile_eq,
            'rail_profile_eq': rail_profile_eq
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
        
        # Create figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # 1. Plot original profiles with correct scaling (mm)
        # Convert to mm for better visualization if profiles are in m
        scale_factor = 1000.0  # Convert from m to mm
        
        # Plot original wheel profile
        wheel_x = self.profiles.wheel_profile[:, 0] * scale_factor
        wheel_y = self.profiles.wheel_profile[:, 1] * scale_factor
        axs[0].plot(wheel_x, wheel_y, 'b-', label='Wheel')
        
        # Plot original rail profile
        rail_x = self.profiles.rail_profile[:, 0] * scale_factor
        rail_y = self.profiles.rail_profile[:, 1] * scale_factor
        axs[0].plot(rail_x, rail_y, 'r-', label='Rail')
        
        axs[0].set_title('Original Wheel-Rail Profiles')
        axs[0].set_xlabel('Lateral position (mm)')
        axs[0].set_ylabel('Vertical position (mm)')
        axs[0].legend()
        axs[0].grid(True)
        
        # Set equal aspect ratio to preserve shape
        axs[0].set_aspect('equal')
        
        # 2. Plot equalized profiles used for contact calculation
        x_eq = results['x_coords'] * scale_factor
        wheel_y_eq = results['wheel_profile_eq'][:, 1] * scale_factor
        rail_y_eq = results['rail_profile_eq'][:, 1] * scale_factor
        
        axs[1].plot(x_eq, wheel_y_eq, 'b--', label='Wheel (equalized)')
        axs[1].plot(x_eq, rail_y_eq, 'r--', label='Rail (equalized)')
        axs[1].set_title('Equalized Profiles for Contact Calculation')
        axs[1].set_xlabel('Lateral position (mm)')
        axs[1].set_ylabel('Vertical position (mm)')
        axs[1].legend()
        axs[1].grid(True)
        
        # 3. Plot pressure distribution
        axs[2].plot(x_eq, results['pressure_distribution'] / 1e6, 'g-')  # Convert to MPa
        axs[2].set_title(f'Pressure Distribution (Max: {results["max_pressure"]/1e6:.2f} MPa)')
        axs[2].set_xlabel('Lateral position (mm)')
        axs[2].set_ylabel('Pressure (MPa)')
        axs[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig


def example_usage():
    """Example usage of the Kp model."""
    # Create simplified wheel and rail profiles
    # In a real application, these would be loaded from files
    
    import numpy as np
    import os
    
    def load_wheel_profile(file_path):
        """Load wheel profile from file."""
        data = np.loadtxt(file_path, delimiter=None, skiprows=1)
        # Point z-axis upwards
        data[:, 1] = -data[:, 1]
        return data

    def load_rail_profile(file_path):
        """Load rail profile from file."""
        data = np.loadtxt(file_path, delimiter=None, skiprows=1)
        # Point z-axis upwards
        data[:, 1] = -data[:, 1]
        return data

    # Create profiles directory if it doesn't exist
    os.makedirs('profiles', exist_ok=True)
    
    # Check if profiles exist, if not create dummy profiles
    wheel_file = 'profiles/S1002.wheel'
    rail_file = 'profiles/uic60i00.rail'
    
    if not os.path.exists(wheel_file) or not os.path.exists(rail_file):
        print("Profile files not found. Creating dummy profiles.")
        # Create dummy wheel profile (circular arc)
        wheel_radius = 0.46  # m
        x_wheel = np.linspace(-0.05, 0.05, 100)
        y_wheel = np.sqrt(wheel_radius**2 - x_wheel**2) - wheel_radius
        wheel_profile = np.column_stack((x_wheel, y_wheel))
        
        # Create dummy rail profile (flat with rounded edges)
        x_rail = np.linspace(-0.05, 0.05, 100)
        y_rail = np.zeros_like(x_rail)
        y_rail[x_rail < -0.03] = -0.01 * np.sqrt(1 - ((x_rail[x_rail < -0.03] + 0.03) / 0.02)**2)
        y_rail[x_rail > 0.03] = -0.01 * np.sqrt(1 - ((x_rail[x_rail > 0.03] - 0.03) / 0.02)**2)
        rail_profile = np.column_stack((x_rail, y_rail))
        
        # Save dummy profiles
        os.makedirs('profiles', exist_ok=True)
        np.savetxt(wheel_file, wheel_profile, header='Dummy wheel profile')
        np.savetxt(rail_file, rail_profile, header='Dummy rail profile')
    else:
        # Load profiles
        wheel_profile = load_wheel_profile(wheel_file)
        rail_profile = load_rail_profile(rail_file)
    
    # Create profiles object
    profiles = WheelRailProfiles(
        wheel_profile=wheel_profile,
        rail_profile=rail_profile,
        wheel_radius=0.46  # 460 mm converted to m
    )
    
    # Create material properties
    material = MaterialProperties(
        E=210e9,  # 210 GPa Young's modulus
        nu=0.28   # Poisson's ratio
    )
    
    # Create Kp model with parameters matching the reference implementation
    penetration = 0.0001  # 0.1 mm penetration in meters
    virtual_penetration = 0.00018  # Virtual penetration in meters
    
    kp_model = KpModel(
        profiles=profiles,
        material=material,
        penetration=penetration,
        virtual_penetration=virtual_penetration,
        yaw_angle=0.0,
        discretization=100
    )
    
    # Run model
    results = kp_model.run()
    
    if results['contact_patch'] is not None:
        # Print results
        print(f"Contact patch dimensions: a = {results['contact_patch']['a']*1000:.2f} mm, b = {results['contact_patch']['b']*1000:.2f} mm")
        print(f"Contact patch area: {results['contact_patch']['area']*1e6:.2f} mm²")
        print(f"Maximum pressure: {results['max_pressure']/1e6:.2f} MPa")
        print(f"Normal force: {results['normal_force']:.2f} N")
        
        # Plot results
        kp_model.plot_results(results)
        plt.show()
    else:
        print("No contact detected.")


if __name__ == "__main__":
    example_usage()
