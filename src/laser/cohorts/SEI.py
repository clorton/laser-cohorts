"""SEI model component preset.

Assembles the Susceptible, Exposed, and Infectious compartments with SE-style
transmission (S → E → I, no recovery) for use as a standalone SEI model.
"""

from laser.cohorts import Susceptible
from laser.cohorts import Exposed
from laser.cohorts import Infectious
from laser.cohorts import TransmissionSE as Transmission

__all__ = ["Exposed", "Infectious", "Susceptible", "Transmission"]
