"""
Kp (Kik-Piotrowski) contact model implementation in Python.
This module implements the non-Hertzian contact model for wheel-rail applications.
Direct adaptation from https://github.com/bytebunny/piotrowski-kik_model

References:
- Piotrowski, J., & Kik, W. (2008). A simplified model of wheel/rail contact mechanics for
  non-Hertzian problems and its application in rail vehicle dynamic simulations.
  Vehicle System Dynamics, 46(1-2), 27-48.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as spi
from scipy import integrate as spint


def get_profiles(rail_path='', wheel_path=''):
    """Returns rail and wheel profiles from given paths with Z-axis upwards.
    
    If no path is given, returns empty array.
    
    Parameters
    ----------
    rail_path : string
        path to rail profile.
    wheel_path : string
        path to wheel profile.
        
    Returns
    -------
    2d array
        rail profile.
    2d array
        wheel profile.
    """
    rail = []
    if rail_path:
        rail = np.loadtxt(rail_path)
        rail[:,1] = - rail[:,1]  # Point z-axis upwards.
        
    wheel = []
    if wheel_path:
        wheel = np.loadtxt(wheel_path, skiprows=2)
        wheel[:,1] = - wheel[:,1]  # Point z-axis upwards.
        
    return rail, wheel


def plot_profiles(profile1, profile2=[], contact_point=[]):
    """Plot profile(s).
    
    Parameters
    ----------
    profile1 : 2d array
        coordinates in solid blue.
    [profile2] : 2d array
        coordinates in dashed red.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('$y$, [mm]')
    plt.ylabel('$z$, [mm]')
    
    ax.plot(profile1[:,0], profile1[:,1], 'b-')
    
    if len(profile2) != 0:
        ax.plot(profile2[:,0], profile2[:,1], 'r--')
        
    if len(contact_point) != 0:
        ax.plot(contact_point[0], contact_point[1], 'ko')
        
    plt.tight_layout()  # Adjust margins to fit tick and axis labels, and titles.
    plt.show()


def equal_points(profile1, profile2):
    """Interpolate *profile1* with same number of points as in *profile2*.
    
    Parameters
    ----------
    profile1 : 2d array
        coordinates to be modified.
    profile2 : 2d array
        reference coordinates.
        
    Returns
    -------
    2d array
        interpolated profile.
    """
    # Find common range to avoid interpolation errors
    min_x1, max_x1 = np.min(profile1[:,0]), np.max(profile1[:,0])
    min_x2, max_x2 = np.min(profile2[:,0]), np.max(profile2[:,0])
    
    common_min_x = max(min_x1, min_x2)
    common_max_x = min(max_x1, max_x2)
    
    # Filter profile2 to be within the common range
    mask = (profile2[:,0] >= common_min_x) & (profile2[:,0] <= common_max_x)
    filtered_profile2 = profile2[mask]
    
    # If no points remain after filtering, create a new x array
    if len(filtered_profile2) < 2:
        x_new = np.linspace(common_min_x, common_max_x, 100)
        filtered_profile2 = np.column_stack((x_new, np.zeros_like(x_new)))
    
    # Create interpolation function
    itp = spi.interp1d(profile1[:,0], profile1[:,1], kind='linear', bounds_error=False, fill_value="extrapolate")
    
    # Interpolate profile1 at profile2's x coordinates
    return np.array([filtered_profile2[:,0], itp(filtered_profile2[:,0])]).T


def separation_of_profiles(wheel, rail):
    """Compute distance between points of two profiles f(y).
    
    Profiles need to be defined in a common coordinate system. The top profile
    (wheel) needs to be the first one in the list of arguments.
    
    Parameters
    ----------
    wheel : 2d array
        coordinates of the top profile.
    rail : 2d array
        coordinates of the bottom profile.
        
    Returns
    -------
    1d array
        distance between points of the two profiles.
    """
    sep = wheel[:,1] - rail[:,1]
    
    # Correct separation if profiles overlap or do not touch:
    min_sep = min(sep)
    
    return sep - min_sep


def interpenetration(wheel, rail, delta0):
    """Compute interpenetration function.
    
    The interpenetration function is defined by eq. 7 in the original article.
    
    Parameters
    ----------
    wheel : 2d array
        coordinates of the wheel profile.
    rail : 2d array
        coordinates of the rail profile.
    delta0 : float 
        virtual penetration.
        
    Returns
    -------
    1d array
        values of interpenetration function.
    """
    sep = separation_of_profiles(wheel, rail)
    
    interp_array = delta0 - sep
    
    ind = 0
    for interp in interp_array:
        if interp > 0:
            interp_array[ind] = interp
        else:
            interp_array[ind] = 0
        ind += 1
        
    return interp_array


def nonzero_runs(a):
    """Returns (n,2) array where n is number of runs of non-zeros.
    
    The first column is the index of the first non-zero in each run,
    and the second is the index of the first zero element after the run.
    This indexing pattern matches, for example, how slicing works and how
    the range function works.
    
    Parameters
    ----------
    a : 1d array
        input.
        
    Returns
    -------
    2d array
        output.
    """
    # Create an array that's 1 where a isn't 0, and pad each end with an extra 0.
    notzero = np.concatenate(([0], np.not_equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(notzero))  # Calculate a[n+1] - a[n] for all.
    
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    
    return ranges


def max_pressure(wheel, g_array, radius, E, nu, delta, delta0):
    """Compute maximum pressures for all contact patches.
    
    Each entry of the returned array is an evaluated eq. 13 in
    the original article.
    
    Parameters
    ----------
    wheel : 2d array
        coordinates of the wheel.
    g_array : 1d array
        interpenetration array.
    radius : float
        wheel nominal rolling radius.
    E : float
        Young's modulus.
    nu : float 
        Poisson's ratio.
    delta : float
        penetration.
    delta0 : float
        virtual penetration.
        
    Returns
    -------
    1d array
        array of maximum contact pressures for each contact patch.
    """
    y_array, z_array = wheel[:,0], wheel[:,1]
    coef = 0.5 * np.pi * E * delta / (1. - nu * nu)
    
    # Function to compute x coordinate of the front edge of the
    # interpenetration region using in situ rolling radius:
    x_front_edge = lambda ind: np.sqrt(2. * radius * g_array[ind])
    
    # 1st integrand:
    f1 = lambda x,y,ind: np.sqrt(x_front_edge(ind) ** 2 - x * x) / \
                         np.sqrt(x * x + y * y + 1.e-10)
    
    # 2nd integrand:
    f2 = lambda x,ind: np.sqrt(x_front_edge(ind) ** 2 - x * x)
    
    # Identify regions with positive interpenetration function:
    region_array = nonzero_runs(g_array)
    
    pmax_array = []
    for region in region_array:
        ind_l, ind_u = region[0], region[1]
        int2_f1 = 0
        int2_f2 = 0
        for ind in range(ind_l, ind_u):
            x_f = x_front_edge(ind)
            int2_f1 += spint.quad(lambda x: f1(x,y_array[ind],ind),
                                 - x_f, x_f,
                                 limit=100)[0]
            int2_f2 += spint.quad(lambda x: f2(x,ind),
                                 - x_f, x_f)[0]
        
        load = coef / int2_f1 * int2_f2
        pmax = load * np.sqrt(2. * radius * delta0) / int2_f2
        pmax_array.append(pmax)
            
    return np.array(pmax_array)


class KpModel:
    """
    Implementation of the Kik-Piotrowski non-Hertzian contact model.
    Direct adaptation from https://github.com/bytebunny/piotrowski-kik_model
    """
    
    def __init__(self, wheel_path='', rail_path='', wheel_radius=460, 
                 E=183e3, nu=0.3, virtual_penetration=0.01, 
                 penetration_reduction_factor=0.55):
        """
        Initialize the Kp model.
        
        Parameters:
        wheel_path -- Path to wheel profile file
        rail_path -- Path to rail profile file
        wheel_radius -- Nominal wheel rolling radius (mm)
        E -- Young's modulus (MPa)
        nu -- Poisson's ratio
        virtual_penetration -- Virtual penetration (mm)
        penetration_reduction_factor -- Ratio between penetration and virtual penetration
        """
        self.wheel_path = wheel_path
        self.rail_path = rail_path
        self.wheel_radius = wheel_radius
        self.E = E
        self.nu = nu
        self.virtual_penetration = virtual_penetration
        self.penetration_reduction_factor = penetration_reduction_factor
        self.penetration = virtual_penetration / penetration_reduction_factor
        
        # Results
        self.wheel = None
        self.rail = None
        self.g_array = None
        self.pmax_array = None
        self.region_array = None
        
    def load_profiles(self):
        """
        Load wheel and rail profiles from files.
        
        Returns:
        wheel, rail -- Wheel and rail profile arrays
        """
        self.rail, self.wheel = get_profiles(self.rail_path, self.wheel_path)
        
        if len(self.wheel) == 0 or len(self.rail) == 0:
            raise ValueError("Failed to load wheel or rail profiles")
            
        return self.wheel, self.rail
    
    def equalize_profiles(self):
        """
        Ensure wheel and rail profiles have the same number of points.
        
        Returns:
        wheel, rail -- Equalized wheel and rail profiles
        """
        if self.wheel is None or self.rail is None:
            self.load_profiles()
            
        # Make sure both profiles have the same number of points
        if len(self.wheel) != len(self.rail):
            if len(self.wheel) > len(self.rail):
                self.rail = equal_points(self.rail, self.wheel)
            else:
                self.wheel = equal_points(self.wheel, self.rail)
                
        return self.wheel, self.rail
    
    def calculate_interpenetration(self):
        """
        Calculate interpenetration between wheel and rail profiles.
        
        Returns:
        g_array -- Array of interpenetration values
        """
        self.equalize_profiles()
        
        # Calculate interpenetration
        self.g_array = interpenetration(self.wheel, self.rail, self.virtual_penetration)
        
        return self.g_array
    
    def calculate_max_pressure(self):
        """
        Calculate maximum pressure for all contact patches.
        
        Returns:
        pmax_array -- Array of maximum pressure values for each contact patch
        """
        if self.g_array is None:
            self.calculate_interpenetration()
            
        # Calculate maximum pressure
        self.pmax_array = max_pressure(
            self.wheel, 
            self.g_array, 
            self.wheel_radius, 
            self.E, 
            self.nu, 
            self.penetration, 
            self.virtual_penetration
        )
        
        # Identify regions with positive interpenetration
        self.region_array = nonzero_runs(self.g_array)
        
        return self.pmax_array
    
    def run(self):
        """
        Run the Kp model to calculate contact patch and pressure distribution.
        
        Returns:
        results -- Dictionary with contact results
        """
        # Calculate maximum pressure
        pmax_array = self.calculate_max_pressure()
        
        if len(pmax_array) == 0:
            print("No contact detected.")
            return None
        
        # Create results dictionary
        results = {
            'wheel': self.wheel,
            'rail': self.rail,
            'g_array': self.g_array,
            'pmax_array': self.pmax_array,
            'region_array': self.region_array,
            'wheel_radius': self.wheel_radius,
            'E': self.E,
            'nu': self.nu,
            'penetration': self.penetration,
            'virtual_penetration': self.virtual_penetration
        }
        
        return results
    
    def plot_results(self, results=None):
        """
        Plot the results of the Kp model.
        
        Parameters:
        results -- Dictionary with contact results (optional)
        """
        if results is None:
            results = self.run()
            
        if results is None:
            return
        
        # Create figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # 1. Plot original profiles
        axs[0].plot(self.wheel[:,0], self.wheel[:,1], 'b-', label='Wheel')
        axs[0].plot(self.rail[:,0], self.rail[:,1], 'r-', label='Rail')
        axs[0].set_title('Original Wheel-Rail Profiles')
        axs[0].set_xlabel('Lateral position (mm)')
        axs[0].set_ylabel('Vertical position (mm)')
        axs[0].legend()
        axs[0].grid(True)
        
        # 2. Plot interpenetration function
        axs[1].plot(self.wheel[:,0], self.g_array, 'g-')
        axs[1].set_title('Interpenetration Function')
        axs[1].set_xlabel('Lateral position (mm)')
        axs[1].set_ylabel('Interpenetration (mm)')
        axs[1].grid(True)
        
       #3. Plot contact patches and maximum pressures
        # First, determine the range to display
        if len(self.region_array) > 0:
            y_min_global = float('inf')
            y_max_global = float('-inf')
            
            for region in self.region_array:
                ind_l, ind_u = region[0], region[1]
                y_min = self.wheel[ind_l,0]
                y_max = self.wheel[ind_u-1,0]
                
                # Add some margin
                y_min_global = min(y_min_global, y_min - 2)
                y_max_global = max(y_max_global, y_max + 2)
            
            # Set the x-axis limits to focus on the contact area
            axs[2].set_xlim(y_min_global, y_max_global)
            
            # Find the maximum pressure value for scaling
            max_pressure_value = np.max(self.pmax_array) if len(self.pmax_array) > 0 else 1.0
            
            # Create a pressure distribution visualization for each contact patch
            for i, region in enumerate(self.region_array):
                ind_l, ind_u = region[0], region[1]
                y_min, y_max = self.wheel[ind_l,0], self.wheel[ind_u-1,0]
                y_center = (y_min + y_max) / 2
                
                # Plot contact patch region
                axs[2].axvspan(y_min, y_max, alpha=0.3, color='blue')
                
                # Create a semi-elliptical pressure distribution
                a = (y_max - y_min) / 2  # Semi-axis in y direction
                
                # Create more points for smooth visualization
                y_fine = np.linspace(y_min, y_max, 100)
                
                # Calculate semi-elliptical pressure distribution
                p_distribution = np.zeros_like(y_fine)
                for j, y in enumerate(y_fine):
                    # Normalized position in contact patch
                    y_rel = (y - y_center) / a if a > 0 else 0
                    
                    # Semi-elliptical distribution
                    if abs(y_rel) <= 1:
                        p_distribution[j] = self.pmax_array[i] * np.sqrt(1 - y_rel**2)
                
                # Plot the pressure distribution
                axs[2].plot(y_fine, p_distribution, 'r-', linewidth=2)
                
                # Fill the area under the curve
                axs[2].fill_between(y_fine, p_distribution, alpha=0.3, color='red')
                
                # Add text label for maximum pressure with improved visibility
                axs[2].text(y_center, self.pmax_array[i] * 1.1, 
                           f'{self.pmax_array[i]:.2f} MPa', 
                           ha='center', fontweight='bold', 
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))
            
            # Set appropriate y-axis limit
            axs[2].set_ylim(0, max_pressure_value * 1.3)
        
        axs[2].set_title('Contact Patches and Maximum Pressures')
        axs[2].set_xlabel('Lateral position (mm)')
        axs[2].set_ylabel('Pressure (MPa)')
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return fig


def example_usage():
    """Example usage of the Kp model."""
    import os
    
    # Create profiles directory if it doesn't exist
    os.makedirs('profiles', exist_ok=True)
    
    # Check if profiles exist, if not create dummy profiles
    wheel_file = 'profiles/S1002.wheel'
    rail_file = 'profiles/uic60i00.rail'
    
    if not os.path.exists(wheel_file) or not os.path.exists(rail_file):
        print("Profile files not found. Creating dummy profiles.")
        # Create dummy wheel profile (circular arc)
        wheel_radius = 460  # mm
        x_wheel = np.linspace(-50, 50, 100)
        y_wheel = np.sqrt(wheel_radius**2 - x_wheel**2) - wheel_radius
        wheel_profile = np.column_stack((x_wheel, y_wheel))
        
        # Create dummy rail profile (flat with rounded edges)
        x_rail = np.linspace(-50, 50, 100)
        y_rail = np.zeros_like(x_rail)
        y_rail[x_rail < -30] = -10 * np.sqrt(1 - ((x_rail[x_rail < -30] + 30) / 20)**2)
        y_rail[x_rail > 30] = -10 * np.sqrt(1 - ((x_rail[x_rail > 30] - 30) / 20)**2)
        rail_profile = np.column_stack((x_rail, y_rail))
        
        # Save dummy profiles
        os.makedirs('profiles', exist_ok=True)
        np.savetxt(wheel_file, wheel_profile, header='Dummy wheel profile')
        np.savetxt(rail_file, rail_profile, header='Dummy rail profile')
    
    # Create Kp model with default parameters
    kp_model = KpModel(
        wheel_path=wheel_file,
        rail_path=rail_file,
        wheel_radius=460,  # mm
        E=183e3,  # MPa
        nu=0.3,
        virtual_penetration=0.01,  # mm
        penetration_reduction_factor=0.55
    )
    
    # Run model
    results = kp_model.run()
    
    if results is not None:
        # Print results
        print(f"Number of contact patches: {len(results['pmax_array'])}")
        for i, pmax in enumerate(results['pmax_array']):
            region = results['region_array'][i]
            y_min = results['wheel'][region[0], 0]
            y_max = results['wheel'][region[1]-1, 0]
            print(f"Contact patch {i+1}:")
            print(f"  Lateral position: {y_min:.2f} to {y_max:.2f} mm")
            print(f"  Maximum pressure: {pmax:.2f} MPa")
        
        # Plot results
        kp_model.plot_results(results)
    else:
        print("No contact detected.")


if __name__ == "__main__":
    example_usage()
