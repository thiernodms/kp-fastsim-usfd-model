"""
Integration module for Kp+FASTSIM+USFD wheel-rail contact and wear modeling.
This module combines the three components into a unified framework.

References:
- Kik-Piotrowski contact model
- Kalker's FASTSIM algorithm
- USFD wear model
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from kp_model import KpModel, WheelRailProfiles, MaterialProperties
from fastsim import FASTSIM, ContactPatch, CreepageParameters, MaterialParameters
from usfd_wear_model import USFDWearModel, WearCoefficients


class KpFastsimUsfdIntegration:
    """
    Integration class for Kp+FASTSIM+USFD wheel-rail contact and wear modeling.
    """
    
    def __init__(self, wheel_profile, rail_profile, material_properties, simulation_params):
        """
        Initialize the integrated Kp+FASTSIM+USFD model.
        
        Parameters:
        wheel_profile -- Wheel profile coordinates (numpy array of [x, y] points)
        rail_profile -- Rail profile coordinates (numpy array of [x, y] points)
        material_properties -- Dictionary containing material properties (E, nu, etc.)
        simulation_params -- Dictionary containing simulation parameters
        """
        self.wheel_profile = wheel_profile
        self.rail_profile = rail_profile
        self.material_properties = material_properties
        self.simulation_params = simulation_params
        
        # Create profiles object for Kp model
        self.profiles = WheelRailProfiles(
            wheel_profile=wheel_profile,
            rail_profile=rail_profile,
            wheel_radius=simulation_params['wheel_radius']
        )
        
        # Create material properties object for Kp model
        self.kp_material = MaterialProperties(
            E=material_properties['E'],
            nu=material_properties['nu']
        )
        
        # Initialize Kp model
        self.kp_model = KpModel(
            profiles=self.profiles,
            material=self.kp_material,
            penetration=simulation_params['penetration'],
            yaw_angle=simulation_params.get('yaw_angle', 0.0),
            discretization=simulation_params.get('discretization', 100)
        )
        
        # Initialize USFD wear model
        self.usfd_model = USFDWearModel()
        
        # FASTSIM will be initialized after Kp model results
        self.fastsim = None
        
        # Results
        self.results = None
    
    def run(self):
        """
        Run the integrated Kp+FASTSIM+USFD simulation.
        
        Returns:
        results -- Dictionary with simulation results
        """
        # Step 1: Run Kp model to get contact patch and normal pressure
        kp_results = self.kp_model.run()
        
        if kp_results['contact_patch'] is None:
            print("No contact detected. Simulation aborted.")
            return {
                'status': 'no_contact',
                'kp_results': kp_results,
                'fastsim_results': None,
                'wear_results': None
            }
        
        # Step 2: Create contact patch object for FASTSIM
        contact_patch = ContactPatch(
            a=kp_results['contact_patch']['a'],
            b=kp_results['contact_patch']['b'],
            area=kp_results['contact_patch']['area'],
            normal_force=kp_results['normal_force']
        )
        
        # Create creepage parameters for FASTSIM
        creepage = CreepageParameters(
            xi_x=self.simulation_params['creepages']['xi_x'],
            xi_y=self.simulation_params['creepages']['xi_y'],
            phi=self.simulation_params['creepages'].get('phi', 0.0)
        )
        
        # Create material parameters for FASTSIM
        material = MaterialParameters(
            G=self.material_properties['G'],
            poisson=self.material_properties['nu'],
            mu=self.simulation_params['friction_coefficient']
        )
        
        # Initialize FASTSIM
        self.fastsim = FASTSIM(
            contact_patch=contact_patch,
            creepage=creepage,
            material=material,
            discretization=self.simulation_params.get('fastsim_discretization', (50, 30))
        )
        
        # Run FASTSIM to get tangential forces
        fastsim_results = self.fastsim.run()
        
        # Step 3: Calculate T-gamma using USFD model
        T_gamma = self.usfd_model.calculate_t_gamma(
            Fx=fastsim_results['Fx'],
            Fy=fastsim_results['Fy'],
            gamma_x=creepage.xi_x,
            gamma_y=creepage.xi_y
        )
        
        # Calculate wear rate
        wear_rate, wear_regime = self.usfd_model.calculate_wear_rate(T_gamma)
        
        # Calculate material loss
        material_loss = self.usfd_model.calculate_material_loss(
            wear_rate=wear_rate,
            contact_area=contact_patch.area * 1e6,  # Convert m² to mm²
            distance=self.simulation_params['running_distance']
        )
        
        # Compile results
        wear_results = {
            'T_gamma': T_gamma,
            'wear_rate': wear_rate,
            'wear_regime': wear_regime,
            'material_loss': material_loss
        }
        
        self.results = {
            'status': 'success',
            'kp_results': kp_results,
            'fastsim_results': fastsim_results,
            'wear_results': wear_results
        }
        
        return self.results
    
    def plot_results(self, save_dir=None):
        """
        Plot the results of the integrated simulation.
        
        Parameters:
        save_dir -- Directory to save the figures (optional)
        
        Returns:
        figs -- List of matplotlib figure objects
        """
        if self.results is None or self.results['status'] != 'success':
            print("No valid results to plot. Run the simulation first.")
            return []
        
        figs = []
        
        # Plot Kp model results
        kp_fig = self.kp_model.plot_results(self.results['kp_results'])
        figs.append(kp_fig)
        
        if save_dir:
            kp_fig.savefig(f"{save_dir}/kp_results.png")
        
        # Plot FASTSIM results
        fastsim_fig = self.fastsim.plot_results()
        figs.append(fastsim_fig)
        
        if save_dir:
            fastsim_fig.savefig(f"{save_dir}/fastsim_results.png")
        
        # Plot USFD wear function
        usfd_fig = self.usfd_model.plot_wear_function()
        figs.append(usfd_fig)
        
        if save_dir:
            usfd_fig.savefig(f"{save_dir}/usfd_wear_function.png")
        
        # Plot summary results
        summary_fig = self.plot_summary_results()
        figs.append(summary_fig)
        
        if save_dir:
            summary_fig.savefig(f"{save_dir}/summary_results.png")
        
        return figs
    
    def plot_summary_results(self):
        """
        Plot summary results from the integrated simulation.
        
        Returns:
        fig -- Matplotlib figure object
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot normal pressure distribution
        kp_results = self.results['kp_results']
        axs[0, 0].plot(kp_results['x_coords'], kp_results['pressure_distribution'], 'b-')
        axs[0, 0].set_title('Normal Pressure Distribution')
        axs[0, 0].set_xlabel('Lateral position (m)')
        axs[0, 0].set_ylabel('Pressure (Pa)')
        axs[0, 0].grid(True)
        
        # Plot tangential stress distribution
        fastsim_results = self.results['fastsim_results']
        x_grid = self.fastsim.x_grid
        y_grid = self.fastsim.y_grid
        
        # Calculate resultant tangential stress
        px = fastsim_results['px']
        py = fastsim_results['py']
        p_tangential = np.sqrt(px**2 + py**2)
        
        # Plot as contour
        im = axs[0, 1].contourf(x_grid, y_grid, p_tangential.T, cmap='viridis')
        axs[0, 1].set_title('Tangential Stress Distribution')
        axs[0, 1].set_xlabel('x (m)')
        axs[0, 1].set_ylabel('y (m)')
        fig.colorbar(im, ax=axs[0, 1], label='Stress (Pa)')
        
        # Plot slip status
        slip_status = fastsim_results['slip_status']
        im2 = axs[1, 0].contourf(x_grid, y_grid, slip_status.T, cmap='binary')
        axs[1, 0].set_title('Slip Status (white = slip, black = stick)')
        axs[1, 0].set_xlabel('x (m)')
        axs[1, 0].set_ylabel('y (m)')
        
        # Plot wear results
        wear_results = self.results['wear_results']
        
        # Create a bar chart for forces and wear
        labels = ['Fx (kN)', 'Fy (kN)', 'T-gamma (N/mm²)', 'Wear Rate (μg/m/mm²)']
        values = [
            fastsim_results['Fx'] / 1000,  # Convert to kN
            fastsim_results['Fy'] / 1000,  # Convert to kN
            wear_results['T_gamma'],
            wear_results['wear_rate']
        ]
        
        colors = ['blue', 'green', 'orange', 'red']
        
        axs[1, 1].bar(labels, values, color=colors)
        axs[1, 1].set_title('Forces and Wear Results')
        axs[1, 1].set_ylabel('Value')
        axs[1, 1].grid(True, axis='y')
        
        # Add text with material loss
        axs[1, 1].text(0.5, -0.2, 
                      f"Material Loss: {wear_results['material_loss']:.2f} μg over {self.simulation_params['running_distance']} m",
                      horizontalalignment='center', transform=axs[1, 1].transAxes)
        
        plt.tight_layout()
        
        return fig


def example_usage():
    """Example usage of the integrated Kp+FASTSIM+USFD model."""
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
        print(f"Wear Regime = {results['wear_results']['wear_regime']} " + 
              f"({'Mild' if results['wear_results']['wear_regime'] == 1 else 'Severe' if results['wear_results']['wear_regime'] == 2 else 'Catastrophic'})")
        print(f"Wear Rate = {results['wear_results']['wear_rate']:.2f} μg/m/mm²")
        print(f"Material Loss = {results['wear_results']['material_loss']:.2f} μg over {simulation_params['running_distance']} m")
        
        # Plot results
        integrated_model.plot_results()
        plt.show()
    else:
        print("Simulation failed. No contact detected.")


if __name__ == "__main__":
    example_usage()
