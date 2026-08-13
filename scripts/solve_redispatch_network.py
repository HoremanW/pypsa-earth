# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Solves the frozen, UC-enabled redispatch network (see redispatch_with_uc.py)
using PyPSA's built-in rolling-horizon optimizer instead of one monolithic
full-year solve.

Why: a full 8760-snapshot UC MILP for this network (~1.9M binary variables
once every conventional generator/link is committable) did not finish its
root LP relaxation within a 30-minute Gurobi time limit -- Gurobi never even
reached the branch-and-bound phase. Chunking the year into overlapping
windows via n.optimize.optimize_with_rolling_horizon() and solving each
window as a much smaller MILP resolved this: manually validated at
horizon=168 (1 week), overlap=24 (1 day) -- all 61 windows solved to
optimality, ~53 minutes total wall time.

pypsa's rolling-horizon optimizer carries state between windows natively:
- Store.e_initial / StorageUnit.state_of_charge_initial are set from the
  previous window's final values before each subsequent solve.
- Committable status carryover is handled a level deeper, inside pypsa's
  own define_operational_constraints_for_committables(): when the snapshots
  being solved don't start at n.snapshots[0], it looks backward into
  n.pnl(c).status (already populated by the prior window's solve, since
  windowed solves write into the same network object) to infer each unit's
  up_time_before/down_time_before automatically. This is the same class of
  initial-condition mechanism used to fix the hour-0 infeasibility in
  redispatch_with_uc.py, generalized by pypsa to every window boundary for
  free.

`overlap` is not optional polish: with overlap=0 each window is solved once
and never revisited, which risks end-of-window myopia (e.g. a unit shutting
down right before a boundary because it can't see the demand just after it).
With overlap>0, consecutive windows share their tail region, so the next
window re-solves and can revise those hours with better foresight.

Deliberately NOT wired into stage-1 capacity expansion: rolling-horizon
would re-decide p_nom/e_nom independently in every window, which doesn't
make sense for a single annual investment decision that needs to see the
whole year to size capacity correctly. This script is only ever used on
the stage-2 redispatch network, where every component is already
non-extendable (see fix_capacities_for_redispatch.py) and only dispatch/
commitment decisions remain -- exactly what rolling-horizon is for.

Known simplification carried over from the validated manual run: unlike
scripts/solve_network.py's solve_network(), this does not add any of the
extra_functionality policy constraints (CCL, EQ, BAU, RES share, CHP, H2
network cap, battery charger/discharger symmetry). Most of those are
capacity constraints that are moot once nothing is extendable; if any
operational ones (CHP, battery symmetry) turn out to matter here, they can
be passed to optimize_with_rolling_horizon via its **kwargs (it forwards
directly to n.optimize each window) -- not done yet since the manually
validated run didn't include them either.

Relevant Settings
------------------
    solving:
        redispatch_rolling_horizon:
            horizon: snapshots per window
            overlap: snapshots shared between consecutive windows
        solver:
            name
        solver_options

Inputs
------
- ``redispatch/prenetworks/...nc``: fixed-capacity, UC-enabled network from
  fix_capacities_for_redispatch.py.

Outputs
-------
- ``redispatch/postnetworks/...nc``: solved redispatch network.
"""
import pypsa
from _helpers import configure_logging, create_logger

logger = create_logger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_redispatch_network",
            simpl="",
            clusters="24",
            ll="v1.0",
            opts="1h",
            sopts="336seg",
            planning_horizons="2030",
            discountrate="0.071",
            demand="AB",
            h2export="0",
        )

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)

    solving = snakemake.params.solving
    rh_params = solving.get("redispatch_rolling_horizon", {})
    horizon = rh_params.get("horizon", 168)
    overlap = rh_params.get("overlap", 24)

    solver_name = solving["solver"]["name"]
    set_of_options = solving["solver"]["options"]
    solver_options = solving["solver_options"][set_of_options] if set_of_options else {}

    logger.info(
        f"Solving redispatch network with rolling horizon: "
        f"{len(n.snapshots)} snapshots, horizon={horizon}, overlap={overlap}."
    )
    n.optimize.optimize_with_rolling_horizon(
        horizon=horizon,
        overlap=overlap,
        solver_name=solver_name,
        solver_options=solver_options,
    )

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
    logger.info(f"Objective (last window only, not full-year total): {n.objective}")
