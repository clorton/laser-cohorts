"""SEIS model component preset.

Assembles the Susceptible, Exposed, and Infectious (recovery returns to S)
compartments with SE-style transmission for use as a standalone SEIS model.
"""

from laser.cohorts import Susceptible
from laser.cohorts import Exposed
from laser.cohorts import InfectiousToSusceptible as Infectious
from laser.cohorts import TransmissionSE as Transmission

__all__ = ["Exposed", "Infectious", "Susceptible", "Transmission"]
