"""Epidemiological compartment components for cohort-based simulation.

Each class represents one compartment or transition rule that can be composed
into a complete compartmental model (SI, SIR, SEIR, etc.).  Components are
registered on a `Model` instance and invoked each tick via `setup`,
`start_step`, `step`, and `end_step`.
"""

from typing import Optional, Type

import numpy as np
from laser.generic.utils import ValuesMap

PropertyType = tuple[str, int, Type[int] | Type[float] | np.dtype, int | float]


class Susceptible:
    """Susceptible (S) compartment component.

    Tracks the susceptible population and applies non-disease mortality each
    time step.  Initial S counts are read from the scenario at setup.
    """

    def __init__(self, model, mu: Optional[ValuesMap] = None, validating: bool = False):
        """Initialize the Susceptible component.

        Args:
            model (Model): The parent model instance.
            mu (ValuesMap | None): Per-tick, per-node non-disease mortality rate.
                Defaults to zero if not provided.
            validating (bool): Enable validation checks during simulation.
        """
        self.model = model
        self.mu = mu if mu is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))
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
        """Apply non-disease mortality to susceptible individuals.

        Draws deaths from a binomial distribution using the complement of the
        survival probability derived from `mu`, subtracts them from S, and
        accumulates them in `nodes.non_disease_mortality`.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        # Non-disease mortality
        probability = -np.expm1(-self.mu[tick])
        sus = self.model.states.S
        mortality = np.random.binomial(sus[tick + 1], probability).astype(sus.dtype)
        self.model.nodes.non_disease_mortality[tick] += mortality
        sus[tick + 1] -= mortality

        return

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
            list[PropertyType]: ``[("non_disease_mortality", nticks, np.int32, 0)]``
        """
        return [
            ("non_disease_mortality", self.model.params.nticks, np.int32, 0),
        ]

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

    def __init__(self, model, sigma: ValuesMap, mu: Optional[ValuesMap] = None, validating: bool = False):
        """Initialize the Exposed component.

        Args:
            model (Model): The parent model instance.
            sigma (ValuesMap): Per-tick, per-node rate of progression from E to I.
            mu (ValuesMap | None): Per-tick, per-node non-disease mortality rate.
                Defaults to zero if not provided.
            validating (bool): Enable validation checks during simulation.
        """
        self.model = model
        self.sigma = sigma
        self.mu = mu if mu is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))
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
        """Apply non-disease mortality and disease progression to exposed individuals.

        Draws non-disease deaths from a binomial using `mu`, then draws newly
        infectious individuals from a binomial using `sigma`, moving them from
        E to I and recording the flow in `nodes.newly_infectious`.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        # Non-disease mortality
        probability = -np.expm1(-self.mu[tick])
        exp = self.model.states.E
        inf = self.model.states.I
        mortality = np.random.binomial(exp[tick + 1], probability).astype(exp.dtype)
        self.model.nodes.non_disease_mortality[tick] += mortality
        exp[tick + 1] -= mortality

        # Disease progression - infectiousness
        probability = -np.expm1(-self.sigma[tick])
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
            list[PropertyType]: Properties for non-disease mortality and newly
                infectious flow counts.
        """
        return [
            ("non_disease_mortality", self.model.params.nticks, np.int32, 0),
            ("newly_infectious", self.model.params.nticks, np.int32, 0),
        ]

    @property
    def states(self) -> list:
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

    def __init__(self, model, mu: Optional[ValuesMap] = None, validating: bool = False):
        """Initialize the Infectious component.

        Args:
            model (Model): The parent model instance.
            mu (ValuesMap | None): Per-tick, per-node non-disease mortality rate.
                Defaults to zero if not provided.
            validating (bool): Enable validation checks during simulation.
        """
        self.model = model
        self.mu = mu if mu is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))
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
        """Apply non-disease mortality to infectious individuals.

        Draws deaths from a binomial distribution using the complement of the
        survival probability derived from `mu`, subtracts them from I, and
        accumulates them in `nodes.non_disease_mortality`.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        # Non-disease mortality
        probability = -np.expm1(-self.mu[tick])
        inf = self.model.states.I
        mortality = np.random.binomial(inf[tick + 1], probability).astype(inf.dtype)
        self.model.nodes.non_disease_mortality[tick] += mortality
        inf[tick + 1] -= mortality

        return

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
            list[PropertyType]: ``[("non_disease_mortality", nticks, np.int32, 0)]``
        """
        return [
            ("non_disease_mortality", self.model.params.nticks, np.int32, 0),
        ]

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

    def __init__(self, model, mu: Optional[ValuesMap] = None, gamma: Optional[ValuesMap] = None, validating: bool = False) -> None:
        """Initialize the InfectiousToRecovered component.

        Args:
            model (Model): The parent model instance.
            mu (ValuesMap | None): Per-tick, per-node non-disease mortality rate.
                Defaults to zero if not provided.
            gamma (ValuesMap | None): Per-tick, per-node recovery rate (I → R).
                Defaults to zero if not provided.
            validating (bool): Enable validation checks during simulation.
        """
        super().__init__(model, mu, validating)
        self.gamma = gamma if gamma is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))
        return

    # def setup(self) -> None:
    #     super().setup()
    #     return

    # def start_step(self, tick: int) -> None:
    #     return super().start_step(tick)

    def step(self, tick: int) -> None:
        """Apply non-disease mortality and recovery (I → R) to infectious individuals.

        Delegates non-disease mortality to the parent `Infectious.step`, then
        draws newly recovered individuals from a binomial using `gamma`, moving
        them from I to R and recording the flow in `nodes.newly_recovered`.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        super().step(tick)  # handles non-disease mortality

        # Disease progression - recovery
        inf = self.model.states.I
        probability = -np.expm1(-self.gamma[tick])
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

    def __init__(self, model, mu: Optional[ValuesMap] = None, gamma: Optional[ValuesMap] = None, validating: bool = False) -> None:
        """Initialize the InfectiousToSusceptible component.

        Args:
            model (Model): The parent model instance.
            mu (ValuesMap | None): Per-tick, per-node non-disease mortality rate.
                Defaults to zero if not provided.
            gamma (ValuesMap | None): Per-tick, per-node rate of recovery to S
                (I → S).  Defaults to zero if not provided.
            validating (bool): Enable validation checks during simulation.
        """
        super().__init__(model, mu, validating)
        self.gamma = gamma if gamma is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))
        return

    # def setup(self) -> None:
    #     super().setup()
    #     return

    # def start_step(self, tick: int) -> None:
    #     return super().start_step(tick)

    def step(self, tick: int) -> None:
        """Apply non-disease mortality and I → S recovery to infectious individuals.

        Delegates non-disease mortality to the parent `Infectious.step`, then
        draws newly susceptible individuals from a binomial using `gamma`,
        moving them from I to S and recording the flow in `nodes.newly_susceptible`.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        super().step(tick)  # handles non-disease mortality

        # Disease progression - recovery
        inf = self.model.states.I
        probability = -np.expm1(-self.gamma[tick])
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

    def __init__(self, model, mu: Optional[ValuesMap] = None, validating: bool = False):
        """Initialize the Recovered component.

        Args:
            model (Model): The parent model instance.
            mu (ValuesMap | None): Per-tick, per-node non-disease mortality rate.
                Defaults to zero if not provided.
            validating (bool): Enable validation checks during simulation.
        """
        self.model = model
        self.mu = mu if mu is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))
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
        """Apply non-disease mortality to recovered individuals.

        Draws deaths from a binomial distribution using the complement of the
        survival probability derived from `mu`, subtracts them from R, and
        accumulates them in `nodes.non_disease_mortality`.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        # Non-disease mortality
        probability = -np.expm1(-self.mu[tick])
        rec = self.model.states.R
        mortality = np.random.binomial(rec[tick + 1], probability).astype(rec.dtype)
        self.model.nodes.non_disease_mortality += mortality
        rec[tick + 1] -= mortality

        return

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
            list[PropertyType]: ``[("non_disease_mortality", nticks, np.int32, 0)]``
        """
        return [
            ("non_disease_mortality", self.model.params.nticks, np.int32, 0),
        ]

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

    def __init__(self, model, mu: Optional[ValuesMap] = None, gamma: Optional[ValuesMap] = None, validating: bool = False) -> None:
        """Initialize the RecoveredToSusceptible component.

        Args:
            model (Model): The parent model instance.
            mu (ValuesMap | None): Per-tick, per-node non-disease mortality rate.
                Defaults to zero if not provided.
            gamma (ValuesMap | None): Per-tick, per-node rate of waning immunity
                (R → S).  Defaults to zero if not provided.
            validating (bool): Enable validation checks during simulation.
        """
        super().__init__(model, mu, validating)

        self.gamma = gamma if gamma is not None else ValuesMap.from_scalar(0, model.params.nticks, len(model.scenario))

        return

    # def setup(self) -> None:
    #     super().setup()
    #     return

    # def start_step(self, tick: int) -> None:
    #     return super().start_step(tick)

    def step(self, tick: int) -> None:
        """Apply non-disease mortality and waning immunity (R → S) to recovered individuals.

        Delegates non-disease mortality to the parent `Recovered.step`, then
        draws individuals with waned immunity from a binomial using `gamma`,
        moving them from R to S and recording the flow in `nodes.newly_susceptible`.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        super().step(tick)  # handles non-disease mortality

        # Disease progression - waning
        rec = self.model.states.R
        probability = -np.expm1(-self.gamma[tick])
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

    def __init__(self, model, beta: ValuesMap, sink_name: str, flow_name: str, validating: bool = False) -> None:
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

    def __init__(self, model, beta: ValuesMap, validating: bool = False) -> None:
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

    def __init__(self, model, beta: ValuesMap, validating: bool = False) -> None:
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
