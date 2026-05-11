"""SEIR model component preset.

Assembles the Susceptible, Exposed, Infectious (recovery to R), and Recovered
compartments with SE-style transmission for use as a standalone SEIR model.
"""

from laser.cohorts import Exposed
from laser.cohorts import InfectiousToRecovered as Infectious
from laser.cohorts import Recovered
from laser.cohorts import Susceptible
from laser.cohorts import TransmissionSE as Transmission

__all__ = ["Exposed", "Infectious", "Recovered", "Susceptible", "Transmission"]
