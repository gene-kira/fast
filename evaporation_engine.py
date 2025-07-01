# evaporation_engine.py

import numpy as np

def hawking_temperature(M, kappa=1.0):
    """
    Computes Hawking temperature assuming natural units (ℏ = c = G = k_B = 1).

    Parameters:
    - M: black hole mass
    - kappa: dimensional correction factor, default = 1 (adjustable for D ≠ 4)

    Returns:
    - T_H: Hawking temperature
    """
    return kappa / (8 * np.pi * M)

def evaporation_time(M, n=0):
    """
    Estimates black hole lifetime due to Hawking evaporation.

    Parameters:
    - M: initial black hole mass
    - n: number of extra dimensions (affects decay rate scaling)

    Returns:
    - lifetime (in Planck time units)
    """
    decay_exponent = 3 / (1 + n)
    return M**decay_exponent  # Simplified scaling model for dimensionality

def decay_chain_particles(M_initial):
    """
    Mock decay chain model for particle burst during evaporation.
    Returns list of particles emitted.

    Placeholder: real model would use Monte Carlo sampling.

    Parameters:
    - M_initial: mass of the evaporating black hole

    Returns:
    - List of particle types
    """
    particles = ['quark', 'gluon', 'lepton', 'photon', 'neutrino']
    return np.random.choice(particles, size=int(10 + M_initial*5)).tolist()

