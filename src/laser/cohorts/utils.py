"""Utility helpers for the laser.cohorts package."""

from typing import Type

import numpy as np

PropertyType = tuple[str, int, Type[int] | Type[float] | np.dtype, int | float]
