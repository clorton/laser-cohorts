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
