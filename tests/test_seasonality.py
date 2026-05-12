"""Tests for seasonality effects on disease transmission.

Seasonality is a ``np.ndarray`` of shape ``(nticks, nnodes)`` passed as the
``seasonality`` argument to ``TransmissionSI``.  When ``None``, the component
uses a constant 1.0 multiplier on beta.

Tests sweep from zero seasonality (no transmission at all) to extreme values
(instant susceptible depletion), and verify structured seasonal patterns
including sinusoidal, triangle-wave, and two-peak annual schedules.
All tests use a single-node SIR model so network effects are absent.
"""

import numpy as np

import laser.core.random
from laser.core import PropertySet
from laser.core.utils import grid
from laser.cohorts import Model
from laser.generic.utils import ValuesMap
import laser.cohorts.SIR as SIR


def _build_sir(
    nticks: int,
    pop: int,
    n_infected: int,
    beta: float,
    r_recovery: float,
    seasonality=None,
) -> Model:
    """Build a single-node SIR model ready to run.

    Args:
        nticks (int): Number of simulation ticks.
        pop (int): Initial total population.
        n_infected (int): Initial infectious count.
        beta (float): Transmission rate.
        r_recovery (float): Per-tick recovery rate.
        seasonality: Optional ndarray of shape ``(nticks, 1)`` or ``ValuesMap``;
            passed directly to ``TransmissionSI``.

    Returns:
        Model: Constructed but not yet run model.
    """
    scenario = grid(M=1, N=1)
    scenario["S"] = pop - n_infected
    scenario["I"] = n_infected
    scenario["R"] = 0

    p = PropertySet({"nticks": nticks, "beta": beta, "r_recovery": r_recovery})
    model = Model(scenario, p)
    betas = ValuesMap.from_scalar(beta, nticks, len(scenario))
    r_recoveries = ValuesMap.from_scalar(r_recovery, nticks, len(scenario))
    model.components = [
        SIR.Susceptible(model),
        SIR.Infectious(model, r_recovery=r_recoveries),
        SIR.Recovered(model),
        SIR.Transmission(model, beta=betas, seasonality=seasonality),
    ]
    return model


def test_zero_seasonality_prevents_all_transmission() -> None:
    """Given a single-node SIR model with zero seasonality across all ticks and
    100 initially infectious individuals, when the model runs for 30 ticks,
    then no new infections occur at any tick.

    Zero seasonality forces the force of infection to zero regardless of S and I
    counts (foi = beta × 0 × I/N = 0), so binomial(S, 0) = 0 always.  This is
    a deterministic test — no seed required.  Failure means the seasonality factor
    is ignored or bypassed in the FoI computation.
    """
    nticks = 30
    seasonality = np.zeros((nticks, 1), dtype=np.float32)
    model = _build_sir(
        nticks=nticks,
        pop=1000,
        n_infected=100,
        beta=0.3,
        r_recovery=1.0 / 7,
        seasonality=seasonality,
    )
    model.run()

    assert model.nodes.newly_infectious.sum() == 0


def test_unit_constant_seasonality_matches_no_seasonality() -> None:
    """Given two identical single-node SIR models run from the same RNG seed —
    one with seasonality=None and one with a constant 1.0 array — when both run
    for 60 ticks, then their per-tick incidence time series are identical.

    Confirms that None defaults to 1.0 and that the constant-array path uses the
    same code as the default path.  Failure means None substitutes a value other
    than 1.0, or the two paths produce different binomial draws.
    """
    nticks = 60
    seasonality_ones = np.ones((nticks, 1), dtype=np.float32)

    laser.core.random.seed(0)
    model_none = _build_sir(
        nticks=nticks,
        pop=10000,
        n_infected=100,
        beta=0.3,
        r_recovery=1.0 / 7,
        seasonality=None,
    )
    model_none.run()

    laser.core.random.seed(0)
    model_ones = _build_sir(
        nticks=nticks,
        pop=10000,
        n_infected=100,
        beta=0.3,
        r_recovery=1.0 / 7,
        seasonality=seasonality_ones,
    )
    model_ones.run()

    np.testing.assert_array_equal(
        model_none.nodes.newly_infectious,
        model_ones.nodes.newly_infectious,
    )


def test_high_constant_seasonality_increases_total_infections() -> None:
    """Given two single-node SIR models run from the same seed — one with
    seasonality=1.0 and one with seasonality=2.0 — when both run for 60 ticks,
    then cumulative infections are strictly higher for the high-seasonality model.

    The FoI is multiplied linearly by seasonality; doubling seasonality roughly
    doubles the early-epidemic infection rate.  Failure means the seasonality
    multiplier has no effect or is incorrectly normalised (e.g., treated as 1).
    """
    nticks = 60

    laser.core.random.seed(0)
    model_low = _build_sir(
        nticks=nticks,
        pop=10000,
        n_infected=100,
        beta=0.3,
        r_recovery=1.0 / 7,
        seasonality=np.ones((nticks, 1), dtype=np.float32),
    )
    model_low.run()

    laser.core.random.seed(0)
    model_high = _build_sir(
        nticks=nticks,
        pop=10000,
        n_infected=100,
        beta=0.3,
        r_recovery=1.0 / 7,
        seasonality=np.full((nticks, 1), 2.0, dtype=np.float32),
    )
    model_high.run()

    assert model_high.nodes.newly_infectious.sum() > model_low.nodes.newly_infectious.sum()


def test_step_function_seasonality_zeros_first_half() -> None:
    """Given a single-node SIR model with step-function seasonality (0.0 for
    the first 60 ticks, 2.0 for the last 60) and 1000 initial infectious in a
    population of 10000, when the model runs for 120 ticks, then no infections
    occur in the first 60 ticks and at least some occur in the second 60.

    The zero-first-half check is deterministic (foi = 0 → binomial(S, 0) = 0
    always).  The positive-second-half check is reliable because with recovery
    rate 1/60 approximately 366 infectious individuals remain at tick 60, and
    seasonality=2 and beta=0.5 produce an expected incidence well above zero.
    Failure in the first assertion means seasonality is not applied per tick.
    Failure in the second means the elevated seasonality or remaining I pool is
    insufficient.
    """
    nticks = 120
    half = nticks // 2
    seasonality = np.zeros((nticks, 1), dtype=np.float32)
    seasonality[half:] = 2.0

    model = _build_sir(
        nticks=nticks,
        pop=10000,
        n_infected=1000,
        beta=0.5,
        r_recovery=1.0 / 60,
        seasonality=seasonality,
    )
    model.run()

    assert model.nodes.newly_infectious[:half].sum() == 0
    assert model.nodes.newly_infectious[half:].sum() > 0


def test_sinusoidal_seasonality_zero_at_trough_tick() -> None:
    """Given a single-node SIR model with annual sinusoidal seasonality
    1 + sin(2π*t/364) (period T=364, divisible by 4) and 500 initial
    infectious, when the model runs for one full period, then no infections
    occur at tick 273 (= 3T/4) where the sinusoid evaluates to exactly -1.

    T=364 is chosen so that 3*364//4 = 273 gives sin(2π × 0.75) = sin(3π/2) = -1
    exactly in IEEE 754 float64, making seasonality[273] = 0.0.  Because
    binomial(S, 0) = 0 always, this is a deterministic test regardless of how
    many susceptibles remain at tick 273.  Failure means the seasonality array is
    not indexed by the current tick or the trough value is not exactly zero.
    """
    T = 364  # divisible by 4 so 3*T/4 = 273 is an exact integer trough
    nticks = T
    t = np.arange(nticks)
    seasonality = (1 + np.sin(2 * np.pi * t / T)).astype(np.float32).reshape(nticks, 1)
    trough_tick = 3 * T // 4  # sin(3π/2) = -1 → seasonality = 0

    model = _build_sir(
        nticks=nticks,
        pop=10000,
        n_infected=500,
        beta=0.3,
        r_recovery=1.0 / 7,
        seasonality=seasonality,
    )
    model.run()

    assert model.nodes.newly_infectious[trough_tick, 0] == 0


def test_triangle_wave_seasonality_zero_at_trough_tick() -> None:
    """Given a single-node SIR model with a triangle-wave seasonality of period
    T=60 that is exactly 0 at t=0 and t=T (and every multiple of T), when the
    model runs for 70 ticks, then no infections occur at tick T=60 (the second
    trough, mid-simulation).

    The triangle rises from 0 at t=0 to a peak of 2.0 at t=30 and falls back to
    0 at t=60.  Tick 60 tests that the trough is reproduced correctly in the
    second period (not just at t=0).  Because binomial(S, 0) = 0 always, this is
    deterministic.  Failure indicates the period wraps incorrectly or the trough
    value is not exactly zero.
    """
    T = 60
    nticks = T + 10  # run past one full period to reach the second trough at t=T
    t = np.arange(nticks)
    phase = (t % T) / T  # repeating [0, 1) with period T
    triangle = np.where(phase < 0.5, 4 * phase, 4 * (1 - phase)).astype(np.float32)
    seasonality = triangle.reshape(nticks, 1)

    model = _build_sir(
        nticks=nticks,
        pop=10000,
        n_infected=500,
        beta=0.3,
        r_recovery=1.0 / 7,
        seasonality=seasonality,
    )
    model.run()

    # At t=T=60: phase = (60 % 60)/60 = 0 → triangle = 4*0 = 0.0
    assert model.nodes.newly_infectious[T, 0] == 0


def test_two_peak_annual_seasonality_zeros_at_all_four_troughs() -> None:
    """Given a single-node SIR model with two-peak seasonality 1+sin(4π*t/T)
    where T=8 (two peaks and two troughs per period), run for two full periods
    (16 ticks) with 500 initial infectious, when the model runs, then no
    infections occur at any of the four trough ticks (t=3, 7, 11, 15).

    A two-peaks-per-period waveform has troughs at t = 3T/8 + kT/2 for integer k;
    with T=8 these fall at exact integers 3, 7, 11, 15 where the seasonality is
    exactly 0.  Checking all four troughs over two full periods verifies both that
    the multi-peak waveform is applied tick-by-tick and that it repeats correctly
    beyond the first period.  Failure at any trough means the seasonality array
    is truncated, phase-shifted, or not indexed correctly.
    """
    T = 8  # short period; peaks at t≈1 and t≈5, troughs at t=3 and t=7
    nticks = 2 * T  # two full periods = 16 ticks
    t = np.arange(nticks)
    seasonality = (1 + np.sin(4 * np.pi * t / T)).astype(np.float32).reshape(nticks, 1)
    trough_ticks = [3, 7, 11, 15]  # 3T/8=3, 7T/8=7, 11T/8=11, 15T/8=15

    model = _build_sir(
        nticks=nticks,
        pop=10000,
        n_infected=500,
        beta=0.3,
        r_recovery=1.0 / 7,
        seasonality=seasonality,
    )
    model.run()

    for tick in trough_ticks:
        assert model.nodes.newly_infectious[tick, 0] == 0, (
            f"Expected 0 infections at trough tick {tick}, got {model.nodes.newly_infectious[tick, 0]}"
        )


def test_extreme_high_seasonality_infects_all_susceptibles_in_first_tick() -> None:
    """Given a single-node SIR model with extreme seasonality (20000×) applied
    every tick, 990 susceptibles, and 10 initially infectious, when the model
    runs, then all 990 susceptibles are infected in the very first tick.

    foi = 20000 × 0.2 × 10/1000 = 40.  In float32, −expm1(−40) = 1.0 exactly
    (since exp(−40) ≈ 4×10⁻¹⁸ rounds to 0 within float32 precision near 1.0).
    binomial(990, 1.0) = 990 deterministically.  Failure means extreme seasonality
    is capped, clipped, or stored/retrieved incorrectly.
    """
    nticks = 10
    pop = 1000
    n_infected = 10
    seasonality = np.full((nticks, 1), 20_000.0, dtype=np.float32)

    model = _build_sir(
        nticks=nticks,
        pop=pop,
        n_infected=n_infected,
        beta=0.2,
        r_recovery=1.0 / 7,
        seasonality=seasonality,
    )
    model.run()

    assert model.nodes.newly_infectious[0, 0] == pop - n_infected
