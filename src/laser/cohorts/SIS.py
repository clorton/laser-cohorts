"""SIS model component preset.

Assembles the Susceptible and Infectious (recovery returns to S) compartments
with SI-style transmission for use as a standalone SIS model.
"""

from laser.cohorts import Susceptible
from laser.cohorts import InfectiousToSusceptible as Infectious
from laser.cohorts import TransmissionSI as Transmission

__all__ = ["Infectious", "Susceptible", "Transmission"]
