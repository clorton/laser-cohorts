"""Tests for spatial disease spread via inter-node network connectivity.

``model.network`` is a 2-D ``np.ndarray`` of shape ``(nnodes, nnodes)``.
Entry ``network[i, j]`` is the fraction of node i's force of infection (FoI)
that leaks into node j each tick.  The FoI computation each tick is:

    foi[:] = beta * seasonality * I / N          # local FoI
    transfer = foi[:, None] * model.network       # what each node sends
    foi += transfer.sum(axis=0)                   # receive from others
    foi -= transfer.sum(axis=1)                   # subtract what was sent
    foi = -expm1(-foi)                            # rate → probability

Row sums must be ≤ 1 to keep post-transfer FoI non-negative; all tests below
respect this constraint.

Network topologies covered: all-zeros (isolation), full all-to-all, one-
directional linear chain, hub-and-spoke, one-directional ring, asymmetric
one-way, strong symmetric (equalisation), and isolated node.
"""

import numpy as np

import laser.core.random
from laser.core import PropertySet
from laser.core.utils import grid
from laser.cohorts import Model
from laser.generic.utils import ValuesMap
import laser.cohorts.SIR as SIR


def _build_sir(
    n_nodes: int,
    pop_per_node,
    infected_per_node,
    nticks: int,
    beta: float,
    r_recovery: float,
    network=None,
) -> Model:
    """Build a multi-node SIR model ready to run.

    Args:
        n_nodes (int): Number of nodes (rows in the scenario).
        pop_per_node: Scalar or 1-D array-like; total population in each node.
        infected_per_node: Scalar or 1-D array-like; initial infectious count
            per node.
        nticks (int): Number of simulation ticks.
        beta (float): Uniform transmission rate (applied to all nodes and ticks).
        r_recovery (float): Per-tick recovery rate.
        network: Optional ndarray of shape ``(n_nodes, n_nodes)``.  When None
            the model default (all-zeros) is used.

    Returns:
        Model: Constructed but not yet run model.
    """
    scenario = grid(M=n_nodes, N=1)
    scenario["S"] = np.asarray(pop_per_node) - np.asarray(infected_per_node)
    scenario["I"] = np.asarray(infected_per_node)
    scenario["R"] = 0

    p = PropertySet({"nticks": nticks, "beta": beta, "r_recovery": r_recovery})
    model = Model(scenario, p)
    if network is not None:
        model.network = network
    betas = ValuesMap.from_scalar(beta, nticks, n_nodes)
    r_recoveries = ValuesMap.from_scalar(r_recovery, nticks, n_nodes)
    model.components = [
        SIR.Susceptible(model),
        SIR.Infectious(model, r_recovery=r_recoveries),
        SIR.Recovered(model),
        SIR.Transmission(model, beta=betas),
    ]
    return model


def test_zero_network_infection_confined_to_seeded_node() -> None:
    """Given a two-node SIR model where node 0 is seeded (100 infectious) and
    node 1 is pristine (I=0), when the model runs for 365 ticks with the default
    all-zeros network, then no infections ever occur in node 1.

    With I₁=0 always and no incoming FoI from the network, foi[1] = 0 every
    tick → binomial(S₁, 0) = 0 always.  This is a deterministic test.  Failure
    means FoI is leaking between nodes even when the network matrix is zero.
    """
    model = _build_sir(
        n_nodes=2,
        pop_per_node=10000,
        infected_per_node=[100, 0],
        nticks=365,
        beta=0.3,
        r_recovery=1.0 / 7,
    )
    model.run()

    assert model.nodes.newly_infectious[:, 1].sum() == 0


def test_full_connectivity_spreads_disease_to_all_nodes() -> None:
    """Given a four-node SIR model with full all-to-all connectivity
    (off-diagonal network weights = 0.2, row sums = 0.6) and only node 0
    seeded, when the model runs for 365 ticks, then every node accumulates at
    least one recovered individual.

    Every node receives FoI from every other node each tick, so disease cannot
    remain isolated in the seed node.  Failure at any unseeded node indicates
    that off-diagonal network weights do not propagate FoI to uninfected nodes.
    """
    n = 4
    network = (1 - np.eye(n)).astype(np.float32) * 0.2

    laser.core.random.seed(0)
    model = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=[100, 0, 0, 0],
        nticks=365,
        beta=0.3,
        r_recovery=1.0 / 7,
        network=network,
    )
    model.run()

    for node in range(n):
        assert model.states.R[-1, node] > 0, f"Node {node} has no recovered individuals"


def test_linear_chain_propagates_disease_along_path() -> None:
    """Given a four-node SIR model with a one-directional chain (network[i, i+1]
    = 0.5 for i=0,1,2; all other entries = 0) and only node 0 seeded, when the
    model runs for 365 ticks, then all four nodes accumulate recovered individuals.

    Disease must hop node-by-node (0→1→2→3), so all four acquiring R>0 confirms
    that chained propagation through directed paths works correctly.  Failure at
    any downstream node means the multi-hop spread is broken or the weight 0.5 is
    insufficient over 365 ticks.
    """
    n = 4
    network = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        network[i, i + 1] = 0.5

    laser.core.random.seed(0)
    model = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=[100, 0, 0, 0],
        nticks=365,
        beta=0.3,
        r_recovery=1.0 / 7,
        network=network,
    )
    model.run()

    for node in range(n):
        assert model.states.R[-1, node] > 0, f"Node {node} has no recovered individuals"


def test_hub_and_spoke_spreads_from_seeded_spoke_to_all_nodes() -> None:
    """Given a five-node hub-and-spoke SIR model (hub=node 0, spokes=nodes 1–4;
    network[i, 0] = network[0, i] = 0.2 for i=1–4) with only spoke 1 seeded,
    when the model runs for 365 ticks, then all five nodes accumulate recovered
    individuals.

    Disease must travel spoke 1 → hub → remaining spokes via a two-hop path.
    All five nodes acquiring R>0 confirms that hub-mediated indirect spread works
    correctly.  Failure means the FoI from spoke 1 does not reach the hub, or
    the hub does not forward it to the other spokes.
    """
    n = 5  # node 0 = hub, nodes 1–4 = spokes
    network = np.zeros((n, n), dtype=np.float32)
    for spoke in range(1, n):
        network[spoke, 0] = 0.2  # spoke sends to hub
        network[0, spoke] = 0.2  # hub sends to spoke

    infected_per_node = [0] * n
    infected_per_node[1] = 100  # seed only spoke 1

    laser.core.random.seed(0)
    model = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=infected_per_node,
        nticks=365,
        beta=0.3,
        r_recovery=1.0 / 7,
        network=network,
    )
    model.run()

    for node in range(n):
        assert model.states.R[-1, node] > 0, f"Node {node} has no recovered individuals"


def test_ring_topology_all_nodes_eventually_infected() -> None:
    """Given a four-node SIR model in a one-directional ring (network[i, (i+1)%4]
    = 0.5) with only node 0 seeded, when the model runs for 365 ticks, then all
    four nodes accumulate recovered individuals.

    Each node receives FoI only from its predecessor; disease circulates around
    the ring until all nodes are infected.  Failure at any node suggests directed
    cycles do not propagate FoI correctly, or that 365 ticks with weight 0.5 is
    insufficient for the wave to complete the ring.
    """
    n = 4
    network = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        network[i, (i + 1) % n] = 0.5

    laser.core.random.seed(0)
    model = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=[100, 0, 0, 0],
        nticks=365,
        beta=0.3,
        r_recovery=1.0 / 7,
        network=network,
    )
    model.run()

    for node in range(n):
        assert model.states.R[-1, node] > 0, f"Node {node} has no recovered individuals"


def test_asymmetric_network_one_way_spread_and_blocked_reverse() -> None:
    """Given a two-node SIR model with an asymmetric network (network[0,1]=0.5,
    network[1,0]=0), two scenarios are verified:

    Forward: node 0 seeded, node 1 pristine → node 1 accumulates infections
    because FoI leaks from node 0 to node 1 via the non-zero weight.

    Reverse: node 1 seeded, node 0 pristine → node 0 never receives any FoI
    because network[1,0]=0 and I₀=0 throughout, so binomial(S₀, 0) = 0 always.

    The reverse case is deterministic.  The forward case uses a seeded RNG.
    Failure in the reverse case means the asymmetry is not respected; failure
    in the forward case means weight 0.5 is insufficient to transmit FoI across
    the directed edge.
    """
    n = 2
    network = np.zeros((n, n), dtype=np.float32)
    network[0, 1] = 0.5  # only node 0 sends to node 1

    # Forward: node 0 seeded → node 1 should pick up infections
    laser.core.random.seed(0)
    model_fwd = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=[100, 0],
        nticks=365,
        beta=0.3,
        r_recovery=1.0 / 7,
        network=network,
    )
    model_fwd.run()
    assert model_fwd.nodes.newly_infectious[:, 1].sum() > 0

    # Reverse: node 1 seeded, same network → node 0 must stay pristine (deterministic)
    model_rev = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=[0, 100],
        nticks=365,
        beta=0.3,
        r_recovery=1.0 / 7,
        network=network,
    )
    model_rev.run()
    assert model_rev.nodes.newly_infectious[:, 0].sum() == 0


def test_strong_symmetric_connectivity_equalizes_epidemic_across_nodes() -> None:
    """Given a two-node SIR model with symmetric connectivity (network[0,1] =
    network[1,0] = 0.5) and only node 0 seeded (100 infectious), when the model
    runs for 365 ticks, then both nodes reach similar recovered fractions
    (within 5 percentage points of each other).

    With w=0.5, each tick the post-transfer FoI of both nodes equals their
    average, so both nodes face the same infection pressure from the very first
    tick and their epidemics track each other closely.  Compare to zero network
    where only node 0 gets infected.  Failure suggests symmetric weights are
    not correctly mixing the FoI between nodes.
    """
    n = 2
    network = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32)

    laser.core.random.seed(0)
    model = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=[100, 0],
        nticks=365,
        beta=0.3,
        r_recovery=1.0 / 7,
        network=network,
    )
    model.run()

    N = model.states[-1].sum(axis=model.states.state_axis - 1)
    r_frac = model.states.R[-1] / N
    assert abs(r_frac[0] - r_frac[1]) < 0.05, f"R fractions diverged: node 0 = {r_frac[0]:.3f}, node 1 = {r_frac[1]:.3f}"


def test_isolated_node_stays_pristine() -> None:
    """Given a three-node SIR model where nodes 0 and 1 are connected
    (network[0,1] = network[1,0] = 0.5) and node 2 is fully isolated (row 2
    and column 2 of the network matrix are all zeros), when nodes 0 and 1 are
    both seeded and the model runs for 365 ticks, then no infections ever
    occur in node 2.

    With I₂=0 always and no incoming FoI (column 2 of network = 0), foi[2] = 0
    every tick → binomial(S₂, 0) = 0 always.  This is a deterministic test.
    Failure means FoI bleeds into a structurally isolated node, indicating a
    matrix indexing error (e.g., rows and columns transposed).
    """
    n = 3
    network = np.zeros((n, n), dtype=np.float32)
    network[0, 1] = 0.5  # nodes 0 and 1 connected; node 2 fully isolated
    network[1, 0] = 0.5

    model = _build_sir(
        n_nodes=n,
        pop_per_node=10000,
        infected_per_node=[100, 100, 0],
        nticks=365,
        beta=0.3,
        r_recovery=1.0 / 7,
        network=network,
    )
    model.run()

    assert model.nodes.newly_infectious[:, 2].sum() == 0
