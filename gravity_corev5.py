# gravity_core.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def schwarzschild_radius(M, M_D=1.0, n=2):
    """
    Calculates the Schwarzschild radius in D = 4 + n dimensions.

    Parameters:
    - M: mass of the black hole (in units of M_D)
    - M_D: Planck mass in D dimensions (default = 1 for scaling)
    - n: number of extra spatial dimensions

    Returns:
    - r_s: Schwarzschild radius in natural units
    """
    prefactor = (1 / M_D) * (M / M_D)**(1 / (n + 1))
    dim_term = (2**n * np.pi**((n - 3) / 2) * gamma((n + 3) / 2) / (n + 2))**(1 / (n + 1))
    return prefactor * dim_term

def plot_r_s_curve(M_range, dims):
    """
    Plots Schwarzschild radius vs mass for different extra dimensions.

    Parameters:
    - M_range: range of black hole masses
    - dims: list of extra dimension counts (n)
    """
    plt.figure(figsize=(10, 6))
    for n in dims:
        r_values = [schwarzschild_radius(M, n=n) for M in M_range]
        plt.plot(M_range, r_values, label=f'n = {n}')
    
    plt.title('Schwarzschild Radius vs Black Hole Mass')
    plt.xlabel('Black Hole Mass (in units of $M_D$)')
    plt.ylabel('Schwarzschild Radius')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

