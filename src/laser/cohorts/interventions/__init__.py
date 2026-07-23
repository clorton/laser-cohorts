"""Vaccination and immunization implementations for laser.cohorts.

Contains both campaign-dispatched ``Intervention`` subclasses (e.g.
``Vaccination``) and standalone vaccination components (e.g.
``RoutineImmunization``) that are added directly to ``model.components``.
"""

from .routine_immunization import RoutineImmunization
from .vaccination import Vaccination

__all__ = ["RoutineImmunization", "Vaccination"]
