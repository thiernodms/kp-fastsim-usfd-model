"""
FASTSIM Algorithm Implementation in Python
Based on Kalker's simplified theory of rolling contact for wheel-rail applications

This module implements the FASTSIM algorithm for calculating tangential forces
in wheel-rail contact. The implementation follows Kalker's simplified theory
and is designed to be integrated with the Kp (Kik-Piotrowski) contact model
and USFD wear model.

References:
- Kalker, J. J. (1982). A fast algorithm for the simplified theory of rolling contact.
  Vehicle System Dynamics, 11(1), 1-13.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class ContactPatch:
    """Class representing the contact patch geometry."""
    a: float  # Semi-axis in rolling direction (m)
    b: float  # Semi-axis in lateral direction (m)
    area: float  # Contact patch area (m²)
    normal_force: float  # Normal force (N)


@dataclass
class CreepageParameters:
    """Class representing creepage parameters."""
    xi_x: float  # Longitudinal creepage
    xi_y: float  # Lateral creepage
    phi: float  # Spin creepage (1/m)


@dataclass
class MaterialParameters:
    """Class representing material parameters."""
    G: float  # Shear modulus (N/m²)
    poisson: float  # Poisson's ratio
    mu: float  # Friction coefficient


class FASTSIM:
    """
    Implementation of Kalker's FASTSIM algorithm for tangential contact forces.
    """
    
    def __init__(self, contact_patch, creepage, material, discretization=(50, 50)):
        """
        Initialize the FASTSIM algorithm.
        
        Parameters:
        contact_patch -- ContactPatch object with contact geometry
        creepage -- CreepageParameters object with creepage values
        material -- MaterialParameters object with material properties
        discretization -- Tuple (nx, ny) with number of grid points in x and y directions
        """
        self.contact_patch = contact_patch
        self.creepage = creepage
        self.material = material
        self.nx, self.ny = discretization
        
        # Calculate Kalker coefficients
        self.c11, self.c22, self.c23 = self._calculate_kalker_coefficients()
        
        # Calculate flexibility parameters
        self.L1 = (8 * self.a) / (3 * self.G * self.c11)
        self.L2 = (8 * self.a) / (3 * self.G * self.c22)
        self.L3 = (8 * self.a) / (3 * self.G * np.pi * self.a * self.c23)
        
        # Initialize grid
        self.dx = 2 * self.a / self.nx
        self.dy = 2 * self.b / self.ny
        self.x_grid = np.linspace(-self.a + self.dx/2, self.a - self.dx/2, self.nx)
        self.y_grid = np.linspace(-self.b + self.dy/2, self.b - self.dy/2, self.ny)
        
        # Initialize pressure distribution (elliptical)
        self.p_grid = np.zeros((self.nx, self.ny))
        self._calculate_pressure_distribution()
        
        # Results
        self.px_grid = np.zeros((self.nx, self.ny))  # Tangential stress in x direction
        self.py_grid = np.zeros((self.nx, self.ny))  # Tangential stress in y direction
        self.slip_grid = np.zeros((self.nx, self.ny), dtype=bool)  # Slip status (True = slip, False = stick)
        
        # Forces
        self.Fx = 0.0  # Longitudinal force
        self.Fy = 0.0  # Lateral force
        self.Mz = 0.0  # Spin moment
        
    @property
    def a(self):
        """Semi-axis in rolling direction."""
        return self.contact_patch.a
        
    @property
    def b(self):
        """Semi-axis in lateral direction."""
        return self.contact_patch.b
        
    @property
    def G(self):
        """Shear modulus."""
        return self.material.G
        
    def _calculate_kalker_coefficients(self):
        """
        Calculate Kalker's coefficients based on a/b ratio.
        
        These are simplified approximations of Kalker's coefficients.
        For more accurate values, refer to Kalker's tables.
        """
        ab_ratio = self.a / self.b
        
        # Simplified approximations based on Kalker's tables
        if ab_ratio <= 0.1:
            c11 = 2.51 * ab_ratio
            c22 = 2.51
            c23 = 1.17
        elif ab_ratio <= 1.0:
            c11 = 2.51 * ab_ratio
            c22 = 2.51
            c23 = 0.25 + 0.92 * ab_ratio
        elif ab_ratio <= 10.0:
            c11 = 2.51
            c22 = 2.51 / ab_ratio
            c23 = 0.25 + 0.92 / ab_ratio
        else:
            c11 = 2.51
            c22 = 2.51 / ab_ratio
            c23 = 1.17 / ab_ratio
            
        # Adjust for Poisson's ratio (simplified)
        poisson_factor = (2 - self.material.poisson) / (2 * (1 - self.material.poisson))
        c11 *= poisson_factor
        c22 *= poisson_factor
        c23 *= poisson_factor
        
        return c11, c22, c23
        
    def _calculate_pressure_distribution(self):
        """Calculate normal pressure distribution (elliptical)."""
        for i, x in enumerate(self.x_grid):
            for j, y in enumerate(self.y_grid):
                # Elliptical pressure distribution
                r_squared = (x/self.a)**2 + (y/self.b)**2
                if r_squared < 1.0:
                    self.p_grid[i, j] = (3 * self.contact_patch.normal_force) / (2 * np.pi * self.a * self.b) * np.sqrt(1 - r_squared)
                else:
                    self.p_grid[i, j] = 0.0
    
    def _is_in_contact(self, x, y):
        """Check if a point is within the contact ellipse."""
        return (x/self.a)**2 + (y/self.b)**2 < 1.0
    
    def run(self):
        """
        Run the FASTSIM algorithm to calculate tangential stresses and forces.
        
        The algorithm processes the contact patch strip by strip in the
        rolling direction, from leading edge to trailing edge.
        """
        # Reset forces
        self.Fx = 0.0
        self.Fy = 0.0
        self.Mz = 0.0
        
        # Process each strip in y direction
        for j, y in enumerate(self.y_grid):
            # Process each element in x direction (from leading to trailing edge)
            for i in range(self.nx-1, -1, -1):
                x = self.x_grid[i]
                
                if not self._is_in_contact(x, y):
                    continue
                
                # Calculate rigid slip
                rigid_slip_x = self.creepage.xi_x - self.creepage.phi * y
                rigid_slip_y = self.creepage.xi_y + self.creepage.phi * x
                
                # Calculate elastic displacement
                if i < self.nx - 1:
                    # Not at leading edge, consider previous point's stress
                    prev_x = self.x_grid[i+1]
                    dx = x - prev_x
                    
                    # Elastic displacement from previous point plus rigid slip
                    u_x = self.px_grid[i+1, j] * self.L1 + rigid_slip_x * dx
                    u_y = self.py_grid[i+1, j] * self.L2 + rigid_slip_y * dx
                else:
                    # At leading edge, only rigid slip
                    dx = self.dx
                    u_x = rigid_slip_x * dx
                    u_y = rigid_slip_y * dx
                
                # Calculate trial stress (assuming stick)
                trial_px = u_x / self.L1
                trial_py = u_y / self.L2
                
                # Calculate traction bound
                traction_bound = self.material.mu * self.p_grid[i, j]
                
                # Check slip condition
                trial_stress_magnitude = np.sqrt(trial_px**2 + trial_py**2)
                
                if trial_stress_magnitude > traction_bound:
                    # Slip condition
                    self.slip_grid[i, j] = True
                    
                    # Scale back to traction bound
                    scale_factor = traction_bound / trial_stress_magnitude
                    self.px_grid[i, j] = trial_px * scale_factor
                    self.py_grid[i, j] = trial_py * scale_factor
                else:
                    # Stick condition
                    self.slip_grid[i, j] = False
                    self.px_grid[i, j] = trial_px
                    self.py_grid[i, j] = trial_py
                
                # Accumulate forces and moment
                dA = self.dx * self.dy
                self.Fx += self.px_grid[i, j] * dA
                self.Fy += self.py_grid[i, j] * dA
                self.Mz += (y * self.px_grid[i, j] - x * self.py_grid[i, j]) * dA
        
        return {
            'Fx': self.Fx,
            'Fy': self.Fy,
            'Mz': self.Mz,
            'px': self.px_grid,
            'py': self.py_grid,
            'slip_status': self.slip_grid,
            'pressure': self.p_grid
        }
    
    def plot_results(self, save_path=None):
        """
        Plot the results of the FASTSIM algorithm.
        
        Parameters:
        save_path -- Path to save the figure (optional)
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot normal pressure
        im0 = axs[0, 0].contourf(self.x_grid, self.y_grid, self.p_grid.T, cmap='viridis')
        axs[0, 0].set_title('Normal Pressure Distribution')
        axs[0, 0].set_xlabel('x (m)')
        axs[0, 0].set_ylabel('y (m)')
        fig.colorbar(im0, ax=axs[0, 0], label='Pressure (Pa)')
        
        # Plot tangential stress in x direction
        im1 = axs[0, 1].contourf(self.x_grid, self.y_grid, self.px_grid.T, cmap='coolwarm')
        axs[0, 1].set_title('Tangential Stress (x-direction)')
        axs[0, 1].set_xlabel('x (m)')
        axs[0, 1].set_ylabel('y (m)')
        fig.colorbar(im1, ax=axs[0, 1], label='Stress (Pa)')
        
        # Plot tangential stress in y direction
        im2 = axs[1, 0].contourf(self.x_grid, self.y_grid, self.py_grid.T, cmap='coolwarm')
        axs[1, 0].set_title('Tangential Stress (y-direction)')
        axs[1, 0].set_xlabel('x (m)')
        axs[1, 0].set_ylabel('y (m)')
        fig.colorbar(im2, ax=axs[1, 0], label='Stress (Pa)')
        
        # Plot slip status
        im3 = axs[1, 1].contourf(self.x_grid, self.y_grid, self.slip_grid.T, cmap='binary')
        axs[1, 1].set_title('Slip Status (white = slip, black = stick)')
        axs[1, 1].set_xlabel('x (m)')
        axs[1, 1].set_ylabel('y (m)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig


def example_usage():
    """Example usage of the FASTSIM algorithm."""
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


if __name__ == "__main__":
    example_usage()
