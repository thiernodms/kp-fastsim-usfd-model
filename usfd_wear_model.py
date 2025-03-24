"""
USFD (University of Sheffield) wear model implementation in Python.
This module implements the T-gamma approach for predicting wear in wheel-rail contact.

References:
- Lewis, R., & Dwyer-Joyce, R. S. (2004). Wear mechanisms and transitions in railway wheel steels.
  Proceedings of the Institution of Mechanical Engineers, Part J: Journal of Engineering Tribology, 218(6), 467-478.
- Network Rail (2019). Guide to calculating Tgamma values.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class WearCoefficients:
    """Class representing wear coefficients for different regimes."""
    K_mild: float        # Mild wear regime coefficient (μg/m/mm²)
    K_severe: float      # Severe wear regime coefficient (μg/m/mm²)
    K_catastrophic: float  # Catastrophic wear regime coefficient (μg/m/mm²)
    
    # T-gamma thresholds for different regimes (N/mm²)
    T_gamma_threshold_1: float  # Threshold between mild and severe
    T_gamma_threshold_2: float  # Threshold between severe and catastrophic


class USFDWearModel:
    """
    Implementation of the USFD wear model based on T-gamma approach.
    """
    
    def __init__(self, wear_coefficients=None):
        """
        Initialize the USFD wear model.
        
        Parameters:
        wear_coefficients -- WearCoefficients object with wear parameters
        """
        # Default wear coefficients based on USFD research if not provided
        if wear_coefficients is None:
            self.wear_coefficients = WearCoefficients(
                K_mild=0.25,         # μg/m/mm²
                K_severe=25.0,       # μg/m/mm²
                K_catastrophic=100.0,  # μg/m/mm²
                T_gamma_threshold_1=10.4,  # N/mm²
                T_gamma_threshold_2=77.2   # N/mm²
            )
        else:
            self.wear_coefficients = wear_coefficients
    
    def calculate_t_gamma(self, Fx, Fy, gamma_x, gamma_y):
        """
        Calculate T-gamma value (energy dissipation per unit area).
        
        Parameters:
        Fx -- Longitudinal creep force (N)
        Fy -- Lateral creep force (N)
        gamma_x -- Longitudinal creepage
        gamma_y -- Lateral creepage
        
        Returns:
        T_gamma -- Energy dissipation per unit area (N/mm²)
        """
        # Calculate T-gamma as the product of forces and creepages
        T_gamma = abs(Fx * gamma_x + Fy * gamma_y)
        return T_gamma
    
    def determine_wear_regime(self, T_gamma):
        """
        Determine wear regime based on T-gamma value.
        
        Parameters:
        T_gamma -- Energy dissipation per unit area (N/mm²)
        
        Returns:
        regime -- Wear regime (1: mild, 2: severe, 3: catastrophic)
        K -- Wear coefficient for the determined regime
        """
        if T_gamma < self.wear_coefficients.T_gamma_threshold_1:
            return 1, self.wear_coefficients.K_mild
        elif T_gamma < self.wear_coefficients.T_gamma_threshold_2:
            return 2, self.wear_coefficients.K_severe
        else:
            return 3, self.wear_coefficients.K_catastrophic
    
    def calculate_wear_rate(self, T_gamma):
        """
        Calculate wear rate based on T-gamma value.
        
        Parameters:
        T_gamma -- Energy dissipation per unit area (N/mm²)
        
        Returns:
        wear_rate -- Material loss per unit area (μg/m/mm²)
        regime -- Wear regime (1: mild, 2: severe, 3: catastrophic)
        """
        regime, K = self.determine_wear_regime(T_gamma)
        wear_rate = K * T_gamma
        return wear_rate, regime
    
    def calculate_material_loss(self, wear_rate, contact_area, distance):
        """
        Calculate total material loss.
        
        Parameters:
        wear_rate -- Material loss per unit area (μg/m/mm²)
        contact_area -- Contact patch area (mm²)
        distance -- Running distance (m)
        
        Returns:
        material_loss -- Total material loss (μg)
        """
        material_loss = wear_rate * contact_area * distance
        return material_loss
    
    def plot_wear_function(self, t_gamma_range=None, save_path=None):
        """
        Plot the USFD wear function.
        
        Parameters:
        t_gamma_range -- Range of T-gamma values to plot (optional)
        save_path -- Path to save the figure (optional)
        
        Returns:
        fig -- Matplotlib figure object
        """
        if t_gamma_range is None:
            t_gamma_range = np.linspace(0, 100, 1000)
        
        wear_rates = []
        regimes = []
        
        for t_gamma in t_gamma_range:
            wear_rate, regime = self.calculate_wear_rate(t_gamma)
            wear_rates.append(wear_rate)
            regimes.append(regime)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot wear rate vs T-gamma
        ax.plot(t_gamma_range, wear_rates, 'b-', linewidth=2)
        
        # Highlight different regimes
        mild_indices = np.where(np.array(regimes) == 1)[0]
        severe_indices = np.where(np.array(regimes) == 2)[0]
        catastrophic_indices = np.where(np.array(regimes) == 3)[0]
        
        if len(mild_indices) > 0:
            ax.fill_between(t_gamma_range[mild_indices], 0, np.array(wear_rates)[mild_indices], 
                           color='green', alpha=0.3, label='Mild')
        
        if len(severe_indices) > 0:
            ax.fill_between(t_gamma_range[severe_indices], 0, np.array(wear_rates)[severe_indices], 
                           color='orange', alpha=0.3, label='Severe')
        
        if len(catastrophic_indices) > 0:
            ax.fill_between(t_gamma_range[catastrophic_indices], 0, np.array(wear_rates)[catastrophic_indices], 
                           color='red', alpha=0.3, label='Catastrophic')
        
        # Add vertical lines at thresholds
        ax.axvline(x=self.wear_coefficients.T_gamma_threshold_1, color='k', linestyle='--', 
                  label=f'Threshold 1: {self.wear_coefficients.T_gamma_threshold_1} N/mm²')
        ax.axvline(x=self.wear_coefficients.T_gamma_threshold_2, color='k', linestyle=':', 
                  label=f'Threshold 2: {self.wear_coefficients.T_gamma_threshold_2} N/mm²')
        
        ax.set_title('USFD Wear Function')
        ax.set_xlabel('T-gamma (N/mm²)')
        ax.set_ylabel('Wear Rate (μg/m/mm²)')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig


def example_usage():
    """Example usage of the USFD wear model."""
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


if __name__ == "__main__":
    example_usage()
