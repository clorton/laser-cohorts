"""SIR model component preset.

Assembles the Susceptible, Infectious (with recovery to R), and Recovered
compartments with SI-style transmission for use as a standalone SIR model.
"""

from laser.cohorts import Susceptible
from laser.cohorts import InfectiousToRecovered as Infectious
from laser.cohorts import Recovered
from laser.cohorts import TransmissionSI as Transmission

__all__ = ["Infectious", "Recovered", "Susceptible", "Transmission"]
