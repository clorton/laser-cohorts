"""Ad-hoc script for manual inspection of component properties and states.

This file is not a pytest test module; it is a standalone script intended for
interactive exploration.  Consider converting to a proper test if assertions are
desired.
"""

import laser.core as core
import laser.core.utils as utils

# import laser.generic as generic
import laser.cohorts as cohorts

scenario = utils.grid(M=3, N=3)
params = core.PropertySet({"nticks": 730})
model = cohorts.Model(scenario, params)
sus = cohorts.Susceptible(model)

print(f"{sus.states=}")
print(f"{sus.properties=}")
