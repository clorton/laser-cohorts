"""Cohort-based compartmental disease modelling for the LASER framework.

Provides composable epidemiological compartments and transmission components
for constructing SI, SIR, SEIR, SIS, SIRS, SEI, SEIS, and SEIRS models.
"""

from importlib.metadata import version

__version__ = version("laser.cohorts")

from .components import Exposed
from .components import Infectious
from .components import InfectiousToRecovered
from .components import InfectiousToSusceptible
from .components import Recovered
from .components import RecoveredToSusceptible
from .components import Susceptible
from .components import TransmissionSI
from .components import TransmissionSE
from .model import Model
from .statearray import StateArray

__all__ = [
    "Exposed",
    "Infectious",
    "InfectiousToRecovered",
    "InfectiousToSusceptible",
    "Model",
    "Recovered",
    "RecoveredToSusceptible",
    "StateArray",
    "Susceptible",
    "TransmissionSE",
    "TransmissionSI",
]
