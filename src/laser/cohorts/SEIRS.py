"""SEIRS model component preset.

Assembles the Susceptible, Exposed, Infectious (recovery to R), and Recovered
(waning immunity back to S) compartments with SE-style transmission for use as
a standalone SEIRS model.
"""

from laser.cohorts import Exposed
from laser.cohorts import InfectiousToRecovered as Infectious
from laser.cohorts import RecoveredToSusceptible as Recovered
from laser.cohorts import Susceptible
from laser.cohorts import TransmissionSE as Transmission

__all__ = ["Exposed", "Infectious", "Recovered", "Susceptible", "Transmission"]
