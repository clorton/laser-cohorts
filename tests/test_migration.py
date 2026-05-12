"""Tests for the Migration component.

``Migration`` physically moves individuals between nodes each tick.  It is
distinct from the FoI-mixing ``model.network``: migration changes actual
compartment counts rather than blending forces of infection.

All tests set ``model.network = np.zeros(...)`` so that any cross-node disease
dynamics observed are caused solely by the physical movement of infected
individuals, not by FoI leakage.

``routing[t, i, j]`` is the fraction of emigrants from node i going to node j
on tick t.  Row-normalised internally; nodes with an all-zero routing row on a
given tick do not emigrate that tick.

For static (time-invariant) connectivity use ``static_routing()`` which wraps
``np.broadcast_to`` to produce a 3-D view without copying data.
"""

import numpy as np
import pytest

import laser.core.random
from laser.core import PropertySet
from laser.core.utils import grid
from laser.cohorts import Migration, Model
from laser.generic.utils import ValuesMap
import laser.cohorts.SIR as SIR
from laser.cohorts.utils import static_routing


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_sir(
    n_nodes: int,
    pop_per_node,
    infected_per_node,
    nticks: int,
    beta: float,
    r_recovery: float,
    r_migration=0.0,
    routing: np.ndarray | None = None,
) -> Model:
    """Build a multi-node SIR model with optional migration and zero FoI network.

    Args:
        n_nodes (int): Number of nodes.
        pop_per_node: Scalar or 1-D array-like; total population per node.
        infected_per_node: Scalar or 1-D array-like; initial infectious per node.
        nticks (int): Number of simulation ticks.
        beta (float): Transmission rate (uniform).
        r_recovery (float): Per-tick recovery rate (uniform).
        r_migration (float): Scalar emigration rate applied to all nodes and
            ticks.  Ignored when routing is None.
        routing (np.ndarray | None): Shape ``(nticks, n_nodes, n_nodes)``
            routing tensor.  When None, no Migration component is added.

    Returns:
        Model: Constructed but not yet run model.
    """
    scenario = grid(M=n_nodes, N=1)
    scenario["S"] = np.asarray(pop_per_node) - np.asarray(infected_per_node)
    scenario["I"] = np.asarray(infected_per_node)
    scenario["R"] = 0

    p = PropertySet({"nticks": nticks, "beta": beta, "r_recovery": r_recovery})
    model = Model(scenario, p)
    # Explicit zero network: disease spread is via migration only
    model.network = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    betas = ValuesMap.from_scalar(beta, nticks, n_nodes)
    r_recoveries = ValuesMap.from_scalar(r_recovery, nticks, n_nodes)

    components = [
        SIR.Susceptible(model),
        SIR.Infectious(model, r_recovery=r_recoveries),
        SIR.Recovered(model),
        SIR.Transmission(model, beta=betas),
    ]
    if routing is not None:
        components.append(Migration(model, r_migration=r_migration, routing=routing))

    model.components = components
    return model


# ---------------------------------------------------------------------------
# Static-routing tests (use static_routing() to create 3-D view from 2-D)
# ---------------------------------------------------------------------------


def test_zero_migration_rate_confines_infection_to_seeded_node() -> None:
    """Given a two-node SIR model with a zero migration rate (r_migration=0) and
    node 1 pristine (I=0), when the model runs for 365 ticks, then no infections
    ever occur in node 1.

    With r_migration=0 the binomial probability is 0, so no individuals leave
    node 0.  Combined with a zero network, node 1 has no exposure to infection.
    This is a deterministic test.  Failure means migration is occurring even when
    the rate is zero.
    """
    n = 2
    nticks = 365
    routing_2d = np.array([[0, 1], [1, 0]], dtype=np.float64)
    model = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=[100, 0],
        nticks=nticks,
        beta=0.3,
        r_recovery=1.0 / 7,
        r_migration=0.0,
        routing=static_routing(routing_2d, nticks),
    )
    model.run()

    assert model.nodes.newly_infectious[:, 1].sum() == 0


def test_full_migration_transfers_all_population_in_one_tick() -> None:
    """Given a two-node SIR model where node 0 sends all emigrants to node 1
    (routing[t,0,1]=1) with r_migration=1e6 (probability ≈ 1.0), when the model
    runs for one tick, then node 0 is completely empty after the tick.

    With p=1.0 and binomial(n, 1.0) = n always, every individual in node 0
    leaves in the first tick.  Because r_recovery and beta are both zero, the
    epidemic does not alter compartment counts before migration fires.  This
    is a deterministic test.  Failure means probability=1.0 does not empty
    the source node.
    """
    n = 2
    nticks = 1
    routing_2d = np.array([[0, 1], [0, 0]], dtype=np.float64)
    model = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=[100, 0],
        nticks=nticks,
        beta=0.0,
        r_recovery=0.0,
        r_migration=1e6,
        routing=static_routing(routing_2d, nticks),
    )
    model.run()

    total_node0 = model.states[-1, :, 0].sum()
    total_node1 = model.states[-1, :, 1].sum()
    assert total_node0 == 0
    assert total_node1 == 20000


def test_migration_conserves_total_population() -> None:
    """Given a four-node SIR model with moderate migration on a ring topology
    (routing[t,i,(i+1)%4]=1, r_migration=0.1), when the model runs for 100 ticks,
    then the total population summed across all nodes equals the initial total at
    every tick.

    Migration moves individuals between nodes but does not create or destroy
    them.  Failure at any tick indicates a leak or duplication in the step logic.
    """
    n = 4
    nticks = 100
    routing_2d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        routing_2d[i, (i + 1) % n] = 1.0

    laser.core.random.seed(0)
    model = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=[100, 0, 0, 0],
        nticks=nticks,
        beta=0.3,
        r_recovery=1.0 / 7,
        r_migration=0.1,
        routing=static_routing(routing_2d, nticks),
    )
    model.run()

    initial_total = model.states[0].sum()
    for tick in range(model.params.nticks + 1):
        tick_total = model.states[tick].sum()
        assert tick_total == initial_total, f"Population not conserved at tick {tick}: expected {initial_total}, got {tick_total}"


def test_directed_migration_only_moves_in_one_direction() -> None:
    """Given a two-node SIR model with asymmetric routing (routing[t,0,1]=1,
    routing[t,1,:]=0) and r_migration=1.0, when the model runs for 30 ticks
    with beta=0 and r_recovery=0, then node 1 gains population and node 0 loses
    population, while no population ever moves from node 1 to node 0.

    Directional routing means only node 0's individuals migrate (to node 1);
    node 1's individuals never leave.  Failure means either the routing direction
    is reversed or node 1 is also losing population.
    """
    n = 2
    nticks = 30
    routing_2d = np.array([[0, 1], [0, 0]], dtype=np.float64)

    laser.core.random.seed(0)
    model = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=[0, 0],
        nticks=nticks,
        beta=0.0,
        r_recovery=0.0,
        r_migration=1.0,
        routing=static_routing(routing_2d, nticks),
    )
    model.run()

    initial_node0 = model.states[0, :, 0].sum()
    initial_node1 = model.states[0, :, 1].sum()
    final_node0 = model.states[-1, :, 0].sum()
    final_node1 = model.states[-1, :, 1].sum()

    assert final_node0 < initial_node0, "Node 0 should have lost population"
    assert final_node1 > initial_node1, "Node 1 should have gained population"
    assert final_node0 + final_node1 == initial_node0 + initial_node1


def test_migration_carries_infected_into_pristine_node() -> None:
    """Given a two-node SIR model where node 0 is seeded (I=100) and node 1 is
    pristine (I=0), with routing[t,0,1]=1 and r_migration=0.1, when the model
    runs for 365 ticks, then node 1 accumulates recovered individuals.

    With zero network, node 1 can only receive infected individuals via physical
    migration from node 0.  Once infected individuals arrive, they can transmit
    locally.  Failure means migrated infected individuals do not participate in
    transmission at their destination.
    """
    n = 2
    nticks = 365
    routing_2d = np.array([[0, 1], [0, 0]], dtype=np.float64)

    laser.core.random.seed(0)
    model = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=[100, 0],
        nticks=nticks,
        beta=0.5,
        r_recovery=1.0 / 7,
        r_migration=0.1,
        routing=static_routing(routing_2d, nticks),
    )
    model.run()

    assert model.states.R[-1, 1] > 0, "Node 1 should have recovered individuals from migrated infection"


def test_chain_migration_spreads_epidemic_along_path() -> None:
    """Given a four-node SIR model with one-directional chain migration
    (routing[t,i,i+1]=1 for i=0,1,2) and only node 0 seeded, when the model runs
    for 365 ticks with r_migration=0.1, then all four nodes accumulate recovered
    individuals.

    With zero network, disease hops node-by-node (0->1->2->3) via physical
    movement of infected individuals.  Checking ``nodes.newly_infectious``
    (which accumulates as infections happen and is not affected by subsequent
    migration) is the correct approach because one-directional routing drains
    early nodes of their recovered individuals.  Failure at any downstream node
    means the multi-hop migration transfer is broken.
    """
    n = 4
    nticks = 365
    routing_2d = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        routing_2d[i, i + 1] = 1.0

    laser.core.random.seed(0)
    model = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=[100, 0, 0, 0],
        nticks=nticks,
        beta=0.5,
        r_recovery=1.0 / 7,
        r_migration=0.1,
        routing=static_routing(routing_2d, nticks),
    )
    model.run()

    for node in range(n):
        assert model.nodes.newly_infectious[:, node].sum() > 0, f"Node {node} never had any infections"


def test_hub_and_spoke_migration_spreads_epidemic() -> None:
    """Given a five-node hub-and-spoke SIR model (hub=0, spokes=1-4) where spokes
    send all migrants to the hub and the hub distributes equally to all spokes,
    with only spoke 1 seeded, when the model runs for 365 ticks, then all five
    nodes accumulate recovered individuals.

    Infected individuals from spoke 1 migrate to the hub, and from the hub to
    other spokes, enabling a two-hop epidemic spread.  Failure means the
    spoke->hub->spoke migration route is broken.
    """
    n = 5
    nticks = 365
    routing_2d = np.zeros((n, n), dtype=np.float64)
    for spoke in range(1, n):
        routing_2d[spoke, 0] = 1.0
        routing_2d[0, spoke] = 1.0 / (n - 1)

    infected_per_node = [0] * n
    infected_per_node[1] = 100

    laser.core.random.seed(0)
    model = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=infected_per_node,
        nticks=nticks,
        beta=0.5,
        r_recovery=1.0 / 7,
        r_migration=0.1,
        routing=static_routing(routing_2d, nticks),
    )
    model.run()

    for node in range(n):
        assert model.states.R[-1, node] > 0, f"Node {node} has no recovered individuals"


def test_ring_migration_spreads_epidemic() -> None:
    """Given a four-node SIR model with one-directional ring migration
    (routing[t,i,(i+1)%4]=1) and only node 0 seeded, when the model runs for
    365 ticks with r_migration=0.1, then all four nodes accumulate recovered
    individuals.

    Infected individuals circulate around the ring via physical migration,
    seeding each successive node.  Failure suggests the ring routing is not
    applied consistently across all nodes.
    """
    n = 4
    nticks = 365
    routing_2d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        routing_2d[i, (i + 1) % n] = 1.0

    laser.core.random.seed(0)
    model = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=[100, 0, 0, 0],
        nticks=nticks,
        beta=0.5,
        r_recovery=1.0 / 7,
        r_migration=0.1,
        routing=static_routing(routing_2d, nticks),
    )
    model.run()

    for node in range(n):
        assert model.states.R[-1, node] > 0, f"Node {node} has no recovered individuals"


def test_isolated_node_stays_pristine() -> None:
    """Given a three-node SIR model where nodes 0 and 1 exchange migrants
    and node 2 has an all-zero routing row (no emigration and no immigration),
    when nodes 0 and 1 are both seeded and the model runs for 365 ticks, then
    no infections ever occur in node 2.

    With zero network and zero routing into node 2, node 2 receives no infected
    individuals and its force of infection is zero every tick.  This is a
    deterministic test.  Failure means individuals are migrating into node 2
    despite its all-zero routing column.
    """
    n = 3
    nticks = 365
    routing_2d = np.zeros((n, n), dtype=np.float64)
    routing_2d[0, 1] = 1.0
    routing_2d[1, 0] = 1.0

    model = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=[100, 100, 0],
        nticks=nticks,
        beta=0.5,
        r_recovery=1.0 / 7,
        r_migration=0.1,
        routing=static_routing(routing_2d, nticks),
    )
    model.run()

    assert model.nodes.newly_infectious[:, 2].sum() == 0


def test_migration_raises_on_wrong_routing_shape() -> None:
    """Given a two-node model with nticks=10, when Migration is constructed
    with a routing tensor of the wrong shape (3×3×3 instead of 10×2×2), then a
    ValueError is raised.

    Routing tensors with the wrong number of nodes or ticks are silently wrong
    and must be caught early.  Failure means invalid routing shapes are accepted
    without error, leading to silent incorrect results.

    Also verifies that a plain 2-D routing matrix (without using static_routing)
    is correctly rejected, nudging users toward the 3-D API.
    """
    scenario = grid(M=2, N=1)
    scenario["S"] = 1000
    scenario["I"] = 0
    scenario["R"] = 0
    p = PropertySet({"nticks": 10, "beta": 0.3, "r_recovery": 1.0 / 7})
    model = Model(scenario, p)

    # Wrong 3-D shape
    wrong_3d = np.zeros((3, 3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="routing must be shape"):
        Migration(model, r_migration=0.1, routing=wrong_3d)

    # 2-D routing (missing time dimension) must also be rejected
    wrong_2d = np.eye(2, dtype=np.float64)
    with pytest.raises(ValueError, match="routing must be shape"):
        Migration(model, r_migration=0.1, routing=wrong_2d)


# ---------------------------------------------------------------------------
# Time-varying routing tests
# ---------------------------------------------------------------------------


def test_routing_inactive_first_half_then_active_second_half() -> None:
    """Given a two-node SIR model where routing is all-zero for ticks 0..N/2-1
    (no migration) and sends all emigrants from node 0 to node 1 for ticks
    N/2..N-1, when node 1 is pristine and beta=0, then node 1 receives no
    population during the inactive phase and gains population only in the
    active phase.

    With beta=0 there is no transmission, so the only way node 1 gains
    individuals is via migration.  The time-varying routing guarantees that
    migration is strictly off for the first half.  This is a deterministic
    test (r_migration=1e6 ensures probability=1.0 each active tick, no
    stochasticity needed).

    Failure means time-varying routing is not indexed by tick, or the inactive
    ticks accidentally allow emigration.
    """
    n = 2
    nticks = 20
    half = nticks // 2

    # Build time-varying routing: zeros for first half, one-way 0->1 for second
    routing_3d = np.zeros((nticks, n, n), dtype=np.float64)
    routing_3d[half:, 0, 1] = 1.0

    laser.core.random.seed(0)
    model = _build_sir(
        n_nodes=n,
        pop_per_node=1000,
        infected_per_node=[0, 0],
        nticks=nticks,
        beta=0.0,
        r_recovery=0.0,
        r_migration=1e6,
        routing=routing_3d,
    )
    model.run()

    # During the inactive phase (ticks 0..half-1) node 1 must not gain anyone.
    # states[tick] is the state entering that tick; after tick t is executed the
    # result is stored in states[t+1].  So states[half, :, 1] holds the state
    # after tick half-1 (all inactive ticks have fired, no active ones yet).
    node1_after_inactive = model.states[half, :, 1].sum()
    initial_node1 = model.states[0, :, 1].sum()
    assert node1_after_inactive == initial_node1, (
        f"Node 1 changed during inactive routing phase: initial={initial_node1}, after inactive={node1_after_inactive}"
    )

    # After the active phase node 1 must have gained population from node 0.
    final_node1 = model.states[-1, :, 1].sum()
    assert final_node1 > initial_node1, "Node 1 should have gained population during active phase"


def test_alternating_direction_routing_moves_population_both_ways() -> None:
    """Given a two-node model with no transmission, where on even ticks node 0
    sends all emigrants to node 1 and on odd ticks node 1 sends all emigrants
    to node 0, when r_migration=1.0, then after many ticks both nodes retain
    non-zero population and the total is conserved.

    Alternating routing bounces population back and forth between the nodes
    each tick.  Because r_migration=1.0 gives a moderate probability (not
    exactly 1), both nodes keep some people.  Population conservation must
    hold at every tick.  Failure means the alternating pattern is not being
    read from the routing tensor, or emigration and immigration accounting
    is wrong.
    """
    n = 2
    nticks = 40

    routing_3d = np.zeros((nticks, n, n), dtype=np.float64)
    for t in range(nticks):
        if t % 2 == 0:
            routing_3d[t, 0, 1] = 1.0  # even: 0 -> 1
        else:
            routing_3d[t, 1, 0] = 1.0  # odd:  1 -> 0

    laser.core.random.seed(0)
    model = _build_sir(
        n_nodes=n,
        pop_per_node=5000,
        infected_per_node=[0, 0],
        nticks=nticks,
        beta=0.0,
        r_recovery=0.0,
        r_migration=1.0,
        routing=routing_3d,
    )
    model.run()

    initial_total = model.states[0].sum()
    for tick in range(nticks + 1):
        tick_total = model.states[tick].sum()
        assert tick_total == initial_total, f"Population not conserved at tick {tick}: expected {initial_total}, got {tick_total}"

    # Both nodes must end up with non-trivial population (not all drained to one side).
    assert model.states[-1, :, 0].sum() > 0, "Node 0 should retain some population"
    assert model.states[-1, :, 1].sum() > 0, "Node 1 should retain some population"


def test_time_varying_routing_conserves_population() -> None:
    """Given a three-node SIR model where the routing changes every 30 ticks
    (cycling through three different directed patterns), when the model runs for
    90 ticks with r_migration=0.15, then the total population is exactly
    conserved at every tick.

    This tests conservation under a non-trivial time-varying routing scheme
    with disease dynamics (beta>0) running simultaneously.  Failure at any tick
    indicates a leak or duplication in the sequential-binomial decomposition.
    """
    n = 3
    nticks = 90
    period = 30

    # Three rotating directed patterns: 0->1->2->0, 1->2->0->1, 2->0->1->2
    patterns = [
        np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64),
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64),
    ]

    routing_3d = np.zeros((nticks, n, n), dtype=np.float64)
    for t in range(nticks):
        routing_3d[t] = patterns[t // period]

    laser.core.random.seed(0)
    model = _build_sir(
        n_nodes=n,
        pop_per_node=8000,
        infected_per_node=[100, 0, 0],
        nticks=nticks,
        beta=0.3,
        r_recovery=1.0 / 7,
        r_migration=0.15,
        routing=routing_3d,
    )
    model.run()

    initial_total = model.states[0].sum()
    for tick in range(nticks + 1):
        tick_total = model.states[tick].sum()
        assert tick_total == initial_total, f"Population not conserved at tick {tick}: expected {initial_total}, got {tick_total}"
