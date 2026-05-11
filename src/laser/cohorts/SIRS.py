"""SIRS model component preset.

Assembles the Susceptible, Infectious (recovery to R), and Recovered (waning
immunity back to S) compartments with SI-style transmission for use as a
standalone SIRS model.
"""

from laser.cohorts import Susceptible
from laser.cohorts import InfectiousToRecovered as Infectious
from laser.cohorts import RecoveredToSusceptible as Recovered
from laser.cohorts import TransmissionSI as Transmission

__all__ = ["Infectious", "Recovered", "Susceptible", "Transmission"]
