"""
MPPI hyperparameter grid search over sigma, rho, and lambd.

For each parameter we sweep its values while holding the other two at their
defaults.  Every (param_value, scenario) combination runs one full simulation
and records the mean weighted lambda-ladder cost from controller.cost_log.

Output: one figure per parameter (3 total), each with five scenario lines.
"""

from __future__ import annotations

import math
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── package imports ────────────────────────────────────────────────────────────
from mpc_ship_nav.mpc.mpc_controller import SimplifiedMPCController
from mpc_ship_nav.sim.engine import SimConfig, Simulator
from mpc_ship_nav.dynamics.traffic import Scenario, TrafficGenerator
from mpc_ship_nav.dynamics.vessel import Vessel, VesselState, VesselParams
from scenario_utils import build_open_water_env, setup_standard_scenario


# ══════════════════════════════════════════════════════════════════════════════
# 1.  GRID DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

# Default (baseline) values — match the original hardcoded values in the controller
DEFAULT_SIGMA = None          # None → max_yaw_rate / 2  inside the controller
DEFAULT_RHO   = 0.8
DEFAULT_LAMBD = 0.01

SIGMA_GRID = [
    math.radians(2),   # very narrow noise
    math.radians(5),
    math.radians(10),
    math.radians(15),
    math.radians(20),  # ≈ max_yaw_rate / 2 at 20°/s vessel
    math.radians(30),
    math.radians(40),
]

RHO_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]

LAMBD_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

# ══════════════════════════════════════════════════════════════════════════════
# 2.  SCENARIO FACTORY
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO_NAMES = [
    "head_on",
    "crossing_starboard",
    "crossing_port",
    "overtaking",
    "overtaken",
]

# Shared simulation settings (kept short so the sweep runs quickly)
SIM_DT          = 1.0     # s
SIM_T_FINAL     = 600.0   # s  – enough for vessels to interact and separate
SIM_LOG_INTERVAL = 5


def _make_scenario_and_route(name: str):
    """
    Build (env, scenario, route) for one of the five COLREG encounter types.
    All scenarios use an open-water environment and an Eastward route for the
    own ship so the geometry is deterministic and comparable across runs.
    """
    env = build_open_water_env()

    # Common own-ship setup: start at origin, head East, target 20 km ahead
    own_x, own_y   = 0.0, 0.0
    own_psi        = 0.0          # East
    own_speed      = 8.0          # m/s
    route          = [(0.0, 0.0), (10_000.0, 0.0), (20_000.0, 0.0)]

    own_state  = VesselState(lat=43.5, lon=16.4, psi=own_psi, v=own_speed,
                             x=own_x, y=own_y)
    own_params = VesselParams(max_yaw_rate=math.radians(20))
    own_ship   = Vessel(own_state, own_params)

    # Traffic vessel geometry — positions chosen so the encounter is clear-cut
    NM = 1852.0  # metres per nautical mile
    sep = 2.5 * NM  # initial separation ≈ 2.5 nm

    if name == "head_on":
        # Other ship heading West, directly in front
        other_state = VesselState(lat=43.5, lon=16.48, psi=math.pi,
                                  v=8.0, x=sep, y=0.0)

    elif name == "crossing_starboard":
        # Other ship approaching from starboard (South), heading North
        other_state = VesselState(lat=43.46, lon=16.42, psi=math.pi / 2,
                                  v=8.0, x=sep * 0.5, y=-sep)

    elif name == "crossing_port":
        # Other ship approaching from port (North), heading South
        other_state = VesselState(lat=43.54, lon=16.42, psi=-math.pi / 2,
                                  v=8.0, x=sep * 0.5, y=sep)

    elif name == "overtaking":
        # Own ship (faster) overtakes slower vessel directly ahead
        other_state = VesselState(lat=43.5, lon=16.42, psi=0.0,
                                  v=4.0, x=sep * 0.5, y=0.0)

    elif name == "overtaken":
        # Own ship (slower) being overtaken from behind
        own_state  = VesselState(lat=43.5, lon=16.4, psi=0.0, v=4.0,
                                 x=0.0, y=0.0)
        own_ship   = Vessel(own_state, own_params)
        other_state = VesselState(lat=43.5, lon=16.36, psi=0.0,
                                  v=8.0, x=-sep * 0.5, y=0.0)
    else:
        raise ValueError(f"Unknown scenario: {name}")

    other_params = VesselParams(max_yaw_rate=math.radians(20))
    other_vessel = Vessel(other_state, other_params)

    # Make sure local (x,y) coords are set on every vessel
    for v in [own_ship, other_vessel]:
        if v.state.x is None or v.state.y is None:
            v.state.x, v.state.y = env.to_local(v.state.lat, v.state.lon)

    scenario = Scenario(own_ship=own_ship, other_vessels=[other_vessel])
    return env, scenario, route, own_ship


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SINGLE-RUN HELPER
# ══════════════════════════════════════════════════════════════════════════════

def run_one(
    scenario_name: str,
    sigma: float | None,
    rho: float,
    lambd: float,
) -> float:
    """
    Run one simulation and return the **mean weighted lambda-ladder cost**
    recorded by the controller across all time steps.

    Returns np.nan if the simulation produced no cost entries (e.g. route
    finished immediately).
    """
    env, scenario, route, own_ship = _make_scenario_and_route(scenario_name)

    controller = SimplifiedMPCController(
        dt=SIM_DT,
        waypoints_xy=route,
        sigma=sigma,
        rho=rho,
        lambd=lambd,
    )

    sim_cfg = SimConfig(dt=SIM_DT, t_final=SIM_T_FINAL, log_interval=SIM_LOG_INTERVAL)
    sim = Simulator(env, scenario, controller, sim_cfg)
    sim.run()

    costs = controller.cost_log
    if not costs:
        return np.nan
    return float(np.mean(costs))


# ══════════════════════════════════════════════════════════════════════════════
# 4.  GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def sweep_parameter(
    param_name: str,
    param_grid: List[float | None],
    fixed_sigma: float | None,
    fixed_rho: float,
    fixed_lambd: float,
) -> Dict[str, List[float]]:
    """
    Sweep one parameter, holding the other two fixed.

    Returns {scenario_name: [mean_cost_per_grid_point]}.
    """
    results: Dict[str, List[float]] = {s: [] for s in SCENARIO_NAMES}

    total = len(param_grid) * len(SCENARIO_NAMES)
    done  = 0

    for val in param_grid:
        if param_name == "sigma":
            sigma, rho, lambd = val, fixed_rho, fixed_lambd
        elif param_name == "rho":
            sigma, rho, lambd = fixed_sigma, val, fixed_lambd
        elif param_name == "lambd":
            sigma, rho, lambd = fixed_sigma, fixed_rho, val
        else:
            raise ValueError(param_name)

        for scen in SCENARIO_NAMES:
            cost = run_one(scen, sigma=sigma, rho=rho, lambd=lambd)
            results[scen].append(cost)
            done += 1
            label = f"{val:.4f}" if val is not None else "default"
            print(f"  [{done:3d}/{total}] {param_name}={label:8s}  {scen:22s}  cost={cost:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO_COLORS = {
    "head_on":            "#e41a1c",
    "crossing_starboard": "#ff7f00",
    "crossing_port":      "#4daf4a",
    "overtaking":         "#377eb8",
    "overtaken":          "#984ea3",
}

SCENARIO_LABELS = {
    "head_on":            "Head-on",
    "crossing_starboard": "Crossing (starboard)",
    "crossing_port":      "Crossing (port)",
    "overtaking":         "Overtaking",
    "overtaken":          "Overtaken",
}

SCENARIO_STYLES = {
    "head_on":            dict(linestyle="-",  marker="o", linewidth=2.5, markersize=7),
    "crossing_starboard": dict(linestyle="--", marker="s", linewidth=2.0, markersize=6),
    "crossing_port":      dict(linestyle="-.", marker="^", linewidth=2.0, markersize=6),
    "overtaking":         dict(linestyle=":",  marker="D", linewidth=2.5, markersize=6),
    "overtaken":          dict(linestyle=(0,(5,1)), marker="x", linewidth=2.0, markersize=7),
}


def plot_sweep(ax, x_values, results, x_label, title, x_formatter=None):
    for scen in SCENARIO_NAMES:
        y = results[scen]
        ax.plot(
            x_values, y,
            color=SCENARIO_COLORS[scen],
            label=SCENARIO_LABELS[scen],
            **SCENARIO_STYLES[scen],
        )

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Mean weighted λ-ladder cost", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    if x_formatter is not None:
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(x_formatter))


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import matplotlib.ticker

    print("=" * 60)
    print("MPPI grid search — sigma sweep")
    print("=" * 60)
    sigma_results = sweep_parameter(
        "sigma", SIGMA_GRID,
        fixed_sigma=DEFAULT_SIGMA, fixed_rho=DEFAULT_RHO, fixed_lambd=DEFAULT_LAMBD,
    )

    print()
    print("=" * 60)
    print("MPPI grid search — rho sweep")
    print("=" * 60)
    rho_results = sweep_parameter(
        "rho", RHO_GRID,
        fixed_sigma=DEFAULT_SIGMA, fixed_rho=DEFAULT_RHO, fixed_lambd=DEFAULT_LAMBD,
    )

    print()
    print("=" * 60)
    print("MPPI grid search — lambd sweep")
    print("=" * 60)
    lambd_results = sweep_parameter(
        "lambd", LAMBD_GRID,
        fixed_sigma=DEFAULT_SIGMA, fixed_rho=DEFAULT_RHO, fixed_lambd=DEFAULT_LAMBD,
    )

    # ── build figures ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "MPPI hyperparameter sensitivity — mean weighted λ-ladder cost per COLREG scenario",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # σ plot — x axis in degrees for readability
    sigma_deg = [math.degrees(s) for s in SIGMA_GRID]
    plot_sweep(
        axes[0], sigma_deg, sigma_results,
        x_label="σ  (control noise std dev, deg/s)",
        title="σ sweep\n(ρ={}, λ={})".format(DEFAULT_RHO, DEFAULT_LAMBD),
    )

    # ρ plot
    plot_sweep(
        axes[1], RHO_GRID, rho_results,
        x_label="ρ  (AR(1) noise correlation)",
        title="ρ sweep\n(σ=default, λ={})".format(DEFAULT_LAMBD),
    )

    # λ plot — log scale
    plot_sweep(
        axes[2], LAMBD_GRID, lambd_results,
        x_label="λ  (MPPI temperature)",
        title="λ sweep\n(σ=default, ρ={})".format(DEFAULT_RHO),
    )
    axes[2].set_xscale("log")

    plt.tight_layout()
    out_path = "mppi_grid_search.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
