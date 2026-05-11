"""SI model component preset.

Assembles the Susceptible and Infectious compartments with SI-style direct
transmission (S → I, no recovery) for use as a standalone SI model.
"""

from laser.cohorts import Susceptible
from laser.cohorts import Infectious
from laser.cohorts import TransmissionSI as Transmission

__all__ = ["Infectious", "Susceptible", "Transmission"]
