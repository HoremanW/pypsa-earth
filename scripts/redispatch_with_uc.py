# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Freezes a solved capacity-expansion network at its optimal capacities and
switches on unit commitment for the thermal fleet, so a subsequent
:mod:`solve_network` run re-dispatches the *optimal* system -- including
new-build capacity -- under UC constraints.

Why this exists
----------------
PyPSA forbids ``committable=True`` together with ``p_nom_extendable=True``:
the dispatch bounds ``p_min_pu * p_nom * status <= p <= p_max_pu * p_nom *
status`` become a bilinear (nonconvex) term if ``p_nom`` is also a decision
variable. So a single-stage joint capacity-expansion-with-UC run can only
apply UC to the non-extendable (existing/legacy) fleet.

The standard literature workaround for this (see e.g. the "GEP-UC" /
generation-expansion-with-unit-commitment tension discussed in Sisternes &
Webster, and the two-stage practice common in capacity expansion studies) is
to decouple planning from operations:

1. Solve capacity expansion without UC (convex, tractable) -- already what
   the overnight ``solve_sector_network`` rule does.
2. Fix all resulting capacities at their optimum and re-enable
   ``committable`` (this script).
3. Re-solve as a pure dispatch problem over the full snapshot set to get
   UC-consistent dispatch, startup costs, and ramping for the *optimal*
   fleet, not just whatever legacy plants survived into stage 1.

The alternative in the literature -- unit-clustered expansion+UC (Palmintier
& Webster 2014), where candidate capacity is built in fixed-size integer
blocks so both the build and commitment decisions stay integer/linear in a
single stage -- avoids the two-stage split but requires discretizing every
candidate carrier into fixed unit sizes, which PyPSA does not support
natively. Not attempted here.

Caveats
-------
- This is a re-dispatch, not a re-optimization of capacity: if the stage-1
  optimal mix cannot actually meet demand once UC constraints (min up/down
  time, ramping, minimum stable load) bite, the stage-2 solve can become
  infeasible or lean heavily on load shedding. ``solving.options.load_shedding``
  is already enabled in this repo's configs, so infeasibility should show up
  as (possibly large) load-shedding costs rather than a hard failure -- check
  for that in the redispatch results before trusting them.
- Zero-capacity generators (carriers the stage-1 solve chose not to build)
  are still frozen and made committable here; they are harmless (p_nom=0
  forces status=0) but add unnecessary binary variables. Consider dropping
  them if solve time becomes an issue.

Relevant Settings
------------------
    costs:
        committable, p_min_pu, ramp_limit_up, ramp_limit_down,
        min_up_time, min_down_time, start_up_cost_per_mw

Inputs
------
- ``postnetworks/...nc``: solved network from the overnight capacity
  expansion (``solve_sector_network``).
- ``costs_{planning_horizons}_sec.csv``: the resolved cost table used by
  that same run, carrying the UC columns applied via the ``costs:`` config
  overrides (see ``process_cost_data.py``).

Outputs
-------
- ``redispatch/prenetworks/...nc``: same network, with every previously
  extendable component fixed at its solved optimum and UC attributes
  applied to generators of the configured committable carriers, ready for
  :mod:`solve_network` to re-dispatch.
"""
import pandas as pd
import pypsa
from _helpers import configure_logging, create_logger

logger = create_logger(__name__)

# component dataframe -> (nominal attr, solved-optimal attr, extendable flag)
EXTENDABLE_COMPONENTS = {
    "generators": ("p_nom", "p_nom_opt", "p_nom_extendable"),
    "links": ("p_nom", "p_nom_opt", "p_nom_extendable"),
    "stores": ("e_nom", "e_nom_opt", "e_nom_extendable"),
    "storage_units": ("p_nom", "p_nom_opt", "p_nom_extendable"),
    "lines": ("s_nom", "s_nom_opt", "s_nom_extendable"),
}


def fix_optimal_capacities(n: pypsa.Network) -> None:
    """Freeze every extendable component at its solved optimum."""
    for df_name, (nom, nom_opt, ext_attr) in EXTENDABLE_COMPONENTS.items():
        df = getattr(n, df_name)
        if df.empty or ext_attr not in df.columns:
            continue
        ext = df.index[df[ext_attr]]
        if ext.empty:
            continue
        df.loc[ext, nom] = df.loc[ext, nom_opt]
        df.loc[ext, ext_attr] = False
        logger.info(f"Fixed {len(ext)} extendable '{df_name}' at their solved optimum.")


# Schema defaults to backfill when a UC attribute column doesn't exist yet on a
# component dataframe (e.g. a Link that has never had `committable` touched).
# ramp_limit_up/down are deliberately left at pypsa's own NaN default
# ("unconstrained"), not 0 ("cannot ramp") -- see add_electricity.py's identical
# fillna choice for the same attributes.
_UC_ATTR_DEFAULTS = {
    "committable": False,
    "p_min_pu": 0.0,
    "ramp_limit_up": float("nan"),
    "ramp_limit_down": float("nan"),
    "min_up_time": 0,
    "min_down_time": 0,
    "start_up_cost": 0.0,
    "up_time_before": 1,
    "down_time_before": 0,
}


def _enable_uc_on_component(
    df: pd.DataFrame, costs: pd.DataFrame, uc_carriers: pd.Index
) -> int:
    idx = df.index[df["carrier"].isin(uc_carriers)]
    if idx.empty:
        return 0

    for attr, default in _UC_ATTR_DEFAULTS.items():
        if attr not in df.columns:
            df[attr] = default

    carriers = df.loc[idx, "carrier"]
    df.loc[idx, "committable"] = True
    df.loc[idx, "p_min_pu"] = carriers.map(costs["p_min_pu"])
    df.loc[idx, "ramp_limit_up"] = carriers.map(costs["ramp_limit_up"])
    df.loc[idx, "ramp_limit_down"] = carriers.map(costs["ramp_limit_down"])
    df.loc[idx, "min_up_time"] = carriers.map(costs["min_up_time"]).fillna(0).astype(int)
    df.loc[idx, "min_down_time"] = (
        carriers.map(costs["min_down_time"]).fillna(0).astype(int)
    )
    df.loc[idx, "start_up_cost"] = (
        carriers.map(costs["start_up_cost_per_mw"]).fillna(0) * df.loc[idx, "p_nom"]
    )
    # Neutralize PyPSA's initial-condition carryover: pypsa's own default
    # (up_time_before=1, down_time_before=0) means "was on for exactly 1h
    # before the horizon started". If min_up_time > 1 that silently forces
    # a must-stay-up obligation for the first (min_up_time - 1) snapshots
    # regardless of economics or of what this unit's capacity/frozen
    # neighbours can actually absorb -- exactly the mechanism that made the
    # first redispatch solve infeasible (see IIS: Link-com-status-min_up_
    # time_must_stay_up forcing status=1 on a frozen-capacity CCGT link at
    # snapshot 0, with no curtailment/export slack left to absorb it).
    # Setting up_time_before = min_up_time already satisfies the minimum-up
    # requirement as of snapshot 0 (pypsa: must_stay_up length is
    # max(0, min_up_time - up_time_before), so this clips it to 0), freeing
    # the optimizer to choose on/off from the first snapshot instead of
    # inheriting a phantom commitment from before the modelled horizon.
    # down_time_before is deliberately left at pypsa's own default (0,
    # falsy) rather than also set to min_down_time: pypsa treats
    # up_time_before>0 and down_time_before>0 as independent flags (own
    # consistency check: "both up and down before the simulation"), and
    # setting both nonzero simultaneously -- even to values that each
    # individually clip their own forced-window to zero -- still trips that
    # warning and is not a state a real generator could ever be in.
    df.loc[idx, "up_time_before"] = df.loc[idx, "min_up_time"]
    return len(idx)


def enable_unit_commitment(n: pypsa.Network, costs: pd.DataFrame) -> None:
    """
    Apply UC attributes to every Generator *and* Link of a committable
    carrier, regardless of whether it was extendable in stage 1.

    Links matter in practice: prepare_sector_network.py's
    convert_conventional_generators_to_links() rebuilds every conventional
    Generator (OCGT/CCGT/coal/oil/biomass/lignite, i.e. anything listed under
    sector.conventional_generation) as a fuel-bus -> AC-bus Link, and that
    conversion does not carry committable/p_min_pu/ramp/min_up_time/
    min_down_time/start_up_cost across. So in a solved sector-coupled
    network, a carrier's UC-eligible capacity may live on Generators, Links,
    or (if some legacy capacity wasn't eligible for conversion) both -- this
    reapplies UC attributes to whichever component type actually holds it.
    """
    uc_carriers = costs.index[costs["committable"].fillna(False).astype(bool)]

    n_gens = _enable_uc_on_component(n.generators, costs, uc_carriers)
    n_links = (
        _enable_uc_on_component(n.links, costs, uc_carriers) if not n.links.empty else 0
    )

    if n_gens == 0 and n_links == 0:
        logger.warning(
            f"No generators or links found for the configured UC carriers {list(uc_carriers)}."
        )
        return

    logger.info(
        f"Enabled unit commitment on {n_gens} generators and {n_links} links "
        f"across carriers {list(uc_carriers)} (previously extendable and existing alike)."
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "fix_capacities_for_redispatch",
            simpl="",
            clusters="4",
            ll="c1",
            opts="Co2L",
            sopts="",
            planning_horizons="2030",
            discountrate="0.071",
            demand="AB",
            h2export="0",
        )

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    costs = pd.read_csv(snakemake.input.costs, index_col=0)

    fix_optimal_capacities(n)
    enable_unit_commitment(n, costs)

    n.export_to_netcdf(snakemake.output[0])
