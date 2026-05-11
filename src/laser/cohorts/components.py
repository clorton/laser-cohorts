"""Epidemiological compartment components for cohort-based simulation.

Each class represents one compartment or transition rule that can be composed
into a complete compartmental model (SI, SIR, SEIR, etc.).  Components are
registered on a `Model` instance and invoked each tick via `setup`,
`start_step`, `step`, and `end_step`.
"""

from typing import Optional

from laser.cohorts.model import Model
from laser.cohorts.utils import PropertyType

import numpy as np
from laser.generic.utils import ValuesMap


class Susceptible:
    """Susceptible (S) compartment component.

    Tracks the susceptible population and applies non-disease mortality each
    time step.  Initial S counts are read from the scenario at setup.
    """

    def __init__(self, model: Model, validating: bool = False):
        """Initialize the Susceptible component.

        Args:
            model (Model): The parent model instance.
            validating (bool): Enable validation checks during simulation.
        """
        self.model = model
        self.validating = validating

        return

    def setup(self) -> None:
        """Initialize state S at tick 0 from the scenario S column."""
        self.model.states.S[0] = self.model.scenario.S
        return

    def start_step(self, tick: int) -> None:
        """Carry forward S counts from the previous tick.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        sus = self.model.states.S
        sus[tick + 1] = sus[tick]
        return

    def step(self, tick: int) -> None:
        """No-op step hook for the S compartment.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    def end_step(self, tick: int) -> None:
        """No-op end-of-step hook for the S compartment.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    @property
    def properties(self) -> list[PropertyType]:
        """Return node properties required by this component.

        Returns:
            list[PropertyType]: Empty list; mortality tracking belongs to
                ``NonDiseaseMortality``.
        """
        return []

    @property
    def states(self) -> list[str]:
        """Return the compartment state names managed by this component.

        Returns:
            list[str]: ``["S"]``
        """
        return ["S"]


class Exposed:
    """Exposed (E) compartment component.

    Tracks individuals who are infected but not yet infectious.  Applies
    non-disease mortality and progression from E to I each time step.
    Initial E counts are read from the scenario at setup.
    """

    def __init__(self, model: Model, r_progression: ValuesMap, validating: bool = False):
        """Initialize the Exposed component.

        Args:
            model (Model): The parent model instance.
            r_progression (ValuesMap): Per-tick, per-node rate of progression from E to I.
            validating (bool): Enable validation checks during simulation.
        """
        self.model = model
        self.r_progression = r_progression
        self.validating = validating

        return

    def setup(self) -> None:
        """Initialize state E at tick 0 from the scenario E column."""
        self.model.states.E[0] = self.model.scenario.E
        return

    def start_step(self, tick: int) -> None:
        """Carry forward E counts from the previous tick.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        exp = self.model.states.E
        exp[tick + 1] = exp[tick]
        return

    def step(self, tick: int) -> None:
        """Apply disease progression to exposed individuals.

        Draws newly infectious individuals from a binomial using `r_progression`,
        moving them from E to I and recording the flow in `nodes.newly_infectious`.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        exp = self.model.states.E
        inf = self.model.states.I
        probability = -np.expm1(-self.r_progression[tick])
        newly_infectious = np.random.binomial(exp[tick + 1], probability)
        self.model.nodes.newly_infectious[tick] += newly_infectious
        exp[tick + 1] -= newly_infectious
        inf[tick + 1] += newly_infectious

        return

    def end_step(self, tick: int) -> None:
        """No-op end-of-step hook for the E compartment.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    @property
    def properties(self) -> list[PropertyType]:
        """Return node properties required by this component.

        Returns:
            list[PropertyType]: ``[("newly_infectious", nticks, np.int32, 0)]``
        """
        return [
            ("newly_infectious", self.model.params.nticks, np.int32, 0),
        ]

    @property
    def states(self) -> list[str]:
        """Return the compartment state names managed by this component.

        Returns:
            list[str]: ``["E", "I"]``
        """
        return ["E", "I"]


class Infectious:
    """Infectious (I) compartment component.

    Tracks the infectious population and applies non-disease mortality each
    time step.  Initial I counts are read from the scenario at setup.
    """

    def __init__(self, model: Model, validating: bool = False):
        """Initialize the Infectious component.

        Args:
            model (Model): The parent model instance.
            validating (bool): Enable validation checks during simulation.
        """
        self.model = model
        self.validating = validating

        return

    def setup(self) -> None:
        """Initialize state I at tick 0 from the scenario I column."""
        self.model.states.I[0] = self.model.scenario.I
        return

    def start_step(self, tick: int) -> None:
        """Carry forward I counts from the previous tick.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        inf = self.model.states.I
        inf[tick + 1] = inf[tick]
        return

    def step(self, tick: int) -> None:
        """No-op step hook for the I compartment.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    def end_step(self, tick: int) -> None:
        """No-op end-of-step hook for the I compartment.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    @property
    def properties(self) -> list[PropertyType]:
        """Return node properties required by this component.

        Returns:
            list[PropertyType]: Empty list; mortality tracking belongs to
                ``NonDiseaseMortality``.
        """
        return []

    @property
    def states(self) -> list[str]:
        """Return the compartment state names managed by this component.

        Returns:
            list[str]: ``["I"]``
        """
        return ["I"]


class InfectiousToRecovered(Infectious):
    """Infectious (I) compartment with recovery to R.

    Extends `Infectious` by drawing newly recovered individuals each tick and
    moving them from I to R.  Used in SIR and SEIR model configurations.
    """

    def __init__(self, model: Model, r_recovery: Optional[ValuesMap] = None, validating: bool = False) -> None:
        """Initialize the InfectiousToRecovered component.

        Args:
            model (Model): The parent model instance.
            r_recovery (ValuesMap | None): Per-tick, per-node recovery rate (I → R).
                Defaults to zero if not provided.
            validating (bool): Enable validation checks during simulation.
        """
        super().__init__(model, validating)
        self.r_recovery = r_recovery if r_recovery is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))
        return

    # def setup(self) -> None:
    #     super().setup()
    #     return

    # def start_step(self, tick: int) -> None:
    #     return super().start_step(tick)

    def step(self, tick: int) -> None:
        """Apply recovery (I → R) to infectious individuals.

        Draws newly recovered individuals from a binomial using `r_recovery`, moving
        them from I to R and recording the flow in `nodes.newly_recovered`.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        inf = self.model.states.I
        probability = -np.expm1(-self.r_recovery[tick])
        newly_recovered = np.random.binomial(inf[tick + 1], probability).astype(inf.dtype)
        self.model.nodes.newly_recovered[tick] += newly_recovered
        inf[tick + 1] -= newly_recovered
        self.model.states.R[tick + 1] += newly_recovered

        return

    # def end_step(self, tick: int) -> None:
    #     return super().end_step(tick)

    @property
    def properties(self) -> list[PropertyType]:
        """Return node properties required by this component.

        Returns:
            list[PropertyType]: Parent properties plus
                ``("newly_recovered", nticks, np.int32, 0)``.
        """
        return super().properties + [
            ("newly_recovered", self.model.params.nticks, np.int32, 0),
        ]

    @property
    def states(self) -> list[str]:
        """Return the compartment state names managed by this component.

        Returns:
            list[str]: Parent states plus ``["R"]``.
        """
        return super().states + ["R"]


class InfectiousToSusceptible(Infectious):
    """Infectious (I) compartment with recovery back to S.

    Extends `Infectious` by drawing newly recovered-susceptible individuals
    each tick and moving them from I to S.  Used in SIS model configurations
    where immunity is not retained after infection.
    """

    def __init__(self, model: Model, r_recovery: Optional[ValuesMap] = None, validating: bool = False) -> None:
        """Initialize the InfectiousToSusceptible component.

        Args:
            model (Model): The parent model instance.
            r_recovery (ValuesMap | None): Per-tick, per-node rate of recovery to S
                (I → S).  Defaults to zero if not provided.
            validating (bool): Enable validation checks during simulation.
        """
        super().__init__(model, validating)
        self.r_recovery = r_recovery if r_recovery is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))
        return

    # def setup(self) -> None:
    #     super().setup()
    #     return

    # def start_step(self, tick: int) -> None:
    #     return super().start_step(tick)

    def step(self, tick: int) -> None:
        """Apply I → S recovery to infectious individuals.

        Draws newly susceptible individuals from a binomial using `r_recovery`,
        moving them from I to S and recording the flow in `nodes.newly_susceptible`.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        inf = self.model.states.I
        probability = -np.expm1(-self.r_recovery[tick])
        newly_susceptible = np.random.binomial(inf[tick + 1], probability).astype(inf.dtype)
        self.model.nodes.newly_susceptible[tick] += newly_susceptible
        inf[tick + 1] -= newly_susceptible
        self.model.states.S[tick + 1] += newly_susceptible

        return

    # def end_step(self, tick: int) -> None:
    #     return super().end_step(tick)

    @property
    def properties(self) -> list[PropertyType]:
        """Return node properties required by this component.

        Returns:
            list[PropertyType]: Parent properties plus
                ``("newly_susceptible", nticks, np.int32, 0)``.
        """
        return super().properties + [
            ("newly_susceptible", self.model.params.nticks, np.int32, 0),
        ]

    @property
    def states(self) -> list[str]:
        """Return the compartment state names managed by this component.

        Returns:
            list[str]: Parent states plus ``["S"]``.
        """
        # + ["S"] almost certainly unnecessary, but for completeness
        return super().states + ["S"]


class Recovered:
    """Recovered (R) compartment component.

    Tracks the recovered population and applies non-disease mortality each
    time step.  Initial R counts are read from the scenario at setup.
    """

    def __init__(self, model: Model, validating: bool = False):
        """Initialize the Recovered component.

        Args:
            model (Model): The parent model instance.
            validating (bool): Enable validation checks during simulation.
        """
        self.model = model
        self.validating = validating

        return

    def setup(self) -> None:
        """Initialize state R at tick 0 from the scenario R column."""
        self.model.states.R[0] = self.model.scenario.R
        return

    def start_step(self, tick: int) -> None:
        """Carry forward R counts from the previous tick.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        rec = self.model.states.R
        rec[tick + 1] = rec[tick]
        return

    def step(self, tick: int) -> None:
        """No-op step hook for the R compartment.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    def end_step(self, tick: int) -> None:
        """No-op end-of-step hook for the R compartment.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    @property
    def properties(self) -> list[PropertyType]:
        """Return node properties required by this component.

        Returns:
            list[PropertyType]: Empty list; mortality tracking belongs to
                ``NonDiseaseMortality``.
        """
        return []

    @property
    def states(self) -> list[str]:
        """Return the compartment state names managed by this component.

        Returns:
            list[str]: ``["R"]``
        """
        return ["R"]


class RecoveredToSusceptible(Recovered):
    """Recovered (R) compartment with waning immunity back to S.

    Extends `Recovered` by drawing individuals with waned immunity each tick
    and moving them from R to S.  Used in SIRS model configurations.
    """

    def __init__(self, model: Model, r_waning: Optional[ValuesMap] = None, validating: bool = False) -> None:
        """Initialize the RecoveredToSusceptible component.

        Args:
            model (Model): The parent model instance.
            r_waning (ValuesMap | None): Per-tick, per-node rate of waning immunity
                (R → S).  Defaults to zero if not provided.
            validating (bool): Enable validation checks during simulation.
        """
        super().__init__(model, validating)

        self.r_waning = r_waning if r_waning is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))

        return

    # def setup(self) -> None:
    #     super().setup()
    #     return

    # def start_step(self, tick: int) -> None:
    #     return super().start_step(tick)

    def step(self, tick: int) -> None:
        """Apply waning immunity (R → S) to recovered individuals.

        Draws individuals with waned immunity from a binomial using `r_waning`,
        moving them from R to S and recording the flow in `nodes.newly_susceptible`.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        rec = self.model.states.R
        probability = -np.expm1(-self.r_waning[tick])
        newly_susceptible = np.random.binomial(rec[tick + 1], probability).astype(rec.dtype)
        self.model.nodes.newly_susceptible[tick] += newly_susceptible
        rec[tick + 1] -= newly_susceptible
        self.model.states.S[tick + 1] += newly_susceptible

        return

    # def end_step(self, tick: int) -> None:
    #     return super().end_step(tick)

    @property
    def properties(self) -> list[PropertyType]:
        """Return node properties required by this component.

        Returns:
            list[PropertyType]: Parent properties plus
                ``("newly_susceptible", nticks, np.int32, 0)``.
        """
        return super().properties + [
            ("newly_susceptible", self.model.params.nticks, np.int32, 0),
        ]

    @property
    def states(self) -> list[str]:
        """Return the compartment state names managed by this component.

        Returns:
            list[str]: Parent states plus ``["S"]``.
        """
        # + ["S"] almost certainly unnecessary, but for completeness
        return super().states + ["S"]


class TransmissionCommon:
    """Base class for stochastic transmission dynamics.

    Computes a frequency-dependent force of infection from the current I and N
    counts, draws newly infected individuals from a binomial distribution, moves
    them from S into a configurable sink compartment, and records the flow in a
    named node property.
    """

    def __init__(self, model: Model, beta: ValuesMap, sink_name: str, flow_name: str, validating: bool = False) -> None:
        """Initialize the transmission component.

        Args:
            model (Model): The parent model instance.
            beta (ValuesMap): Per-tick, per-node transmission rate.
            sink_name (str): Name of the destination state for newly infected
                individuals (e.g., ``"I"`` or ``"E"``).
            flow_name (str): Name of the node property that accumulates the
                newly infected flow count each tick.
            validating (bool): Enable validation checks during simulation.
        """
        self.model = model
        self.beta = beta
        self.validating = validating

        self.sink_name = sink_name
        self.flow_name = flow_name

        return

    def setup(self) -> None:
        """Cache references to the sink state and flow node property arrays."""
        self._sink = getattr(self.model.states, self.sink_name)
        self._flow = getattr(self.model.nodes, self.flow_name)
        return

    def start_step(self, tick: int) -> None:
        """No-op start-of-step hook for the transmission component.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    def step(self, tick: int) -> None:
        """Apply stochastic transmission, moving individuals from S to the sink state.

        Computes the per-node force of infection as ``beta * I / N``, converts
        to per-tick infection probability, draws newly infected individuals from
        a binomial distribution, and moves them from S into the sink compartment.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        S = self.model.states.S[tick + 1]
        I = self.model.states.I[tick + 1]  # noqa: E741
        N = self.model.states[tick + 1].sum(axis=self.model.states.state_axis - 1)
        rates = self.beta[tick] * I / N
        probabilities = -np.expm1(-rates)
        newly_infected = np.random.binomial(S, probabilities).astype(self._flow.dtype)
        self._flow[tick] = newly_infected
        S -= newly_infected
        self._sink[tick + 1] += newly_infected
        return

    def end_step(self, tick: int) -> None:
        """No-op end-of-step hook for the transmission component.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    @property
    def properties(self) -> list[PropertyType]:
        """Return node properties required by this component.

        Returns:
            list[PropertyType]: Empty list; subclasses add the flow property.
        """
        return []

    @property
    def states(self) -> list[str]:
        """Return the compartment state names required by this component.

        Returns:
            list[str]: ``["S"]``; subclasses add the sink state name.
        """
        return ["S"]


class TransmissionSI(TransmissionCommon):
    """Transmission component that moves individuals directly from S to I.

    Specialises `TransmissionCommon` for models without an exposed period (SI,
    SIR, SIS, SIRS).  Newly infected individuals enter the I compartment
    immediately.
    """

    def __init__(self, model: Model, beta: ValuesMap, validating: bool = False) -> None:
        """Initialize the S → I transmission component.

        Args:
            model (Model): The parent model instance.
            beta (ValuesMap): Per-tick, per-node transmission rate.
            validating (bool): Enable validation checks during simulation.
        """
        super().__init__(model, beta, "I", "newly_infectious", validating)
        return

    # def setup(self) -> None:
    #     super().setup()
    #     return

    # def start_step(self, tick: int) -> None:
    #     super().start_step(tick)
    #     return

    # def step(self, tick: int) -> None:
    #     super().step(tick)
    #     return

    # def end_step(self, tick: int) -> None:
    #     supert().end_step(tick)
    #     return

    @property
    def properties(self) -> list[PropertyType]:
        """Return node properties required by this component.

        Returns:
            list[PropertyType]: Parent properties plus
                ``("newly_infectious", nticks, np.int32, 0)``.
        """
        return super().properties + [
            ("newly_infectious", self.model.params.nticks, np.int32, 0),
        ]

    @property
    def states(self) -> list[str]:
        """Return the compartment state names managed by this component.

        Returns:
            list[str]: Parent states plus ``["I"]``.
        """
        return super().states + ["I"]


class TransmissionSE(TransmissionCommon):
    """Transmission component that moves individuals from S to E.

    Specialises `TransmissionCommon` for models with an exposed/latent period
    (SEI, SEIR, SEIS, SEIRS).  Newly infected individuals enter the E
    compartment before becoming infectious.
    """

    def __init__(self, model: Model, beta: ValuesMap, validating: bool = False) -> None:
        """Initialize the S → E transmission component.

        Args:
            model (Model): The parent model instance.
            beta (ValuesMap): Per-tick, per-node transmission rate.
            validating (bool): Enable validation checks during simulation.
        """
        super().__init__(model, beta, "E", "newly_infected", validating)

    # def setup(self) -> None:
    #     super().setup()
    #     return

    # def start_step(self, tick: int) -> None:
    #     super().start_step(tick)
    #     return

    # def step(self, tick: int) -> None:
    #     super().step(tick)
    #     return

    # def end_step(self, tick: int) -> None:
    #     supert().end_step(tick)
    #     return

    @property
    def properties(self) -> list[PropertyType]:
        """Return node properties required by this component.

        Returns:
            list[PropertyType]: Parent properties plus
                ``("newly_infected", nticks, np.int32, 0)``.
        """
        return super().properties + [
            ("newly_infected", self.model.params.nticks, np.int32, 0),
        ]

    @property
    def states(self) -> list[str]:
        """Return the compartment state names managed by this component.

        Returns:
            list[str]: Parent states plus ``["E"]``.
        """
        return super().states + ["E"]
