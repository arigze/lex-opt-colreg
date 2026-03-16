"""
Utility functions for COLREG scenario test files.
Common functions to reduce code duplication across test scenarios.
"""

import argparse
import math
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from mpc_ship_nav.charts.config import RegionConfig
from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.dynamics.colreg import D_COLREG, D_COLLISION
from mpc_ship_nav.dynamics.traffic import Scenario
from mpc_ship_nav.dynamics.vessel import Vessel, VesselParams, VesselState
from mpc_ship_nav.mpc.mpc_controller import SimplifiedMPCController
from mpc_ship_nav.sim.engine import SimConfig, Simulator
from mpc_ship_nav.sim.visualize import animate_trajectories, plot_trajectories


def make_vessel_at_xy(
    env: ChartEnvironment,
    *,
    x: float,
    y: float,
    psi: float,
    v: float,
    max_yaw_rate_deg: float,
) -> Vessel:
    """Create a vessel at specified local (x, y) coordinates."""
    lat, lon = env.to_geo(x, y)
    state = VesselState(lat=lat, lon=lon, psi=float(psi), v=float(v), x=float(x), y=float(y))
    params = VesselParams(max_yaw_rate=math.radians(max_yaw_rate_deg))
    return Vessel(state, params)


def build_open_water_env(lat_center: float = 43.5, lon_center: float = 16.4) -> ChartEnvironment:
    """Build an open water environment (no land) for testing."""
    cfg = RegionConfig(
        lat_min=lat_center - 0.25,
        lat_max=lat_center + 0.25,
        lon_min=lon_center - 0.35,
        lon_max=lon_center + 0.35,
        grid_resolution=300.0,
        coastal_buffer=0.0,
        coastline_file=None,
    )
    env = ChartEnvironment(cfg)
    env.land_geometry = Polygon()  # Fully navigable water
    env.set_origin(lat_center, lon_center)
    return env

def build_coastal_env() -> ChartEnvironment:
    """Build a coastal environment for testing."""
    lat_center = 43.5
    lon_center = 16.4
    cfg = RegionConfig(
        lat_min=lat_center - 0.25,
        lat_max=lat_center + 0.25,
        lon_min=lon_center - 0.35,
        lon_max=lon_center + 0.35,
        grid_resolution=300.0,
        coastal_buffer=0.0,
        coastline_file=None,
    )
    env = ChartEnvironment(cfg)
    x_c, y_c = env.to_local(lat_center, lon_center)
    size = 10_000.0  # 10 km half-size
    env.land_geometry = Polygon([
                (x_c - size, y_c - size),
                (x_c + size, y_c - size),
                (x_c + size, y_c + size),
                (x_c - size, y_c + size),
            ])  # Coastal land 
    env.set_origin(lat_center, lon_center)
    return env

def check_collisions(log, threshold: float = D_COLLISION) -> Tuple[bool, Optional[float], float]:
    """
    Check if any collision occurred during simulation.
    
    Returns:
        (collision_occurred, collision_time, min_distance)
        If no collision, collision_time is None.
    """
    times = log.times
    own_states = log.own_states
    traffic_states = log.traffic_states
    if not traffic_states:
        return False, None, float("inf")

    min_dist_overall = float("inf")
    collision_time = None

    for k in range(len(times)):
        own = own_states[k]
        for tr in traffic_states[k]:
            d = math.hypot(tr.x - own.x, tr.y - own.y)
            if d < min_dist_overall:
                min_dist_overall = d
            if d < threshold:
                if collision_time is None or times[k] < collision_time:
                    collision_time = times[k]

    if collision_time is not None:
        return True, collision_time, min_dist_overall
    return False, None, min_dist_overall


def create_standard_parser() -> argparse.ArgumentParser:
    """Create a standard argument parser for COLREG test scenarios."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-final", type=float, default=100000.0, help="Simulation duration (seconds)")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step (seconds)")
    parser.add_argument("--log-interval", type=int, default=5, help="Logging interval (steps)")
    parser.add_argument("--animate", action="store_true", help="Save animation GIF")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot")
    parser.add_argument("--no-traffic-colreg", action="store_true", help="Disable COLREG for traffic ships")
    return parser


def setup_standard_scenario(
    env: ChartEnvironment,
    own_x: float = 0.0,
    own_y: float = 0.0,
    own_heading: float = 0.0,
    own_speed: float = 8.0,
    route: Optional[List[Tuple[float, float]]] = None,
    dt: float = 1.0,
    t_final: float = 3000.0,
    log_interval: int = 5,
) -> Tuple[Vessel, SimplifiedMPCController, SimConfig]:
    """
    Set up standard own ship, route, controller, and simulation config.
    
    Args:
        env: Chart environment
        own_x, own_y: Own ship starting position
        own_heading: Own ship heading in radians (0 = East)
        own_speed: Own ship speed in m/s
        route: Waypoint route (default: straight East)
        dt: Time step
        t_final: Simulation duration
        log_interval: Logging interval
    
    Returns:
        (own_ship, controller, sim_cfg)
    """
    if route is None:
        route = [(0.0, 0.0), (10_000.0, 0.0), (20_000.0, 0.0)]
    
    own_ship = make_vessel_at_xy(
        env, x=own_x, y=own_y, psi=own_heading, v=own_speed, max_yaw_rate_deg=20.0
    )
    
    sim_cfg = SimConfig(dt=dt, t_final=t_final, log_interval=log_interval)
    controller = SimplifiedMPCController(dt=sim_cfg.dt, waypoints_xy=route)
    
    return own_ship, controller, sim_cfg


def run_scenario(
    env: ChartEnvironment,
    own_ship: Vessel,
    traffic_vessels: List[Vessel],
    controller: SimplifiedMPCController,
    sim_cfg: SimConfig,
    fix_traffic_positions: Optional[List[Tuple[float, float, float]]] = None,
) -> Tuple:
    """
    Run a simulation scenario.
    
    Args:
        env: Chart environment
        own_ship: Own ship vessel
        traffic_vessels: List of traffic vessels
        controller: MPC controller
        sim_cfg: Simulation configuration
        fix_traffic_positions: Optional list of (x, y, psi) tuples to fix traffic positions
                             after simulator initialization. Useful to ensure correct spawn positions.
    
    Returns:
        (log, sim) - simulation log and simulator instance
    """
    scenario = Scenario(own_ship=own_ship, other_vessels=traffic_vessels)
    sim = Simulator(env, scenario, controller, sim_cfg)
    
    # Fix traffic positions if specified (useful for head-on scenarios)
    if fix_traffic_positions is not None:
        for i, (x, y, psi) in enumerate(fix_traffic_positions):
            if i < len(sim.other_vessels):
                v = sim.other_vessels[i]
                v.state.x = x
                v.state.y = y
                v.state.psi = psi
                lat, lon = env.to_geo(x, y)
                v.state.lat = lat
                v.state.lon = lon
    
    log = sim.run()
    return log, sim


def plot_results(
    env: ChartEnvironment,
    log,
    route: List[Tuple[float, float]],
    show_plot: bool = True,
) -> None:
    """Plot simulation trajectories."""
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_trajectories(env, log, ax=ax)
    wx, wy = zip(*route)
    ax.scatter(wx, wy, c="orange", s=15, zorder=4, label="waypoints")
    ax.legend()
    if show_plot:
        plt.show()


def save_animation(
    env: ChartEnvironment,
    log,
    route: List[Tuple[float, float]],
    output_path: Path,
    fps: int = 10,
) -> None:
    """Save simulation animation as GIF."""
    animate_trajectories(env, log, route, fps=fps, save_path=output_path)
    print(f"Saved: {output_path}")


def spawn_random_background_vessels(
    env: ChartEnvironment,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    n_vessels: int,
    speed_range: Tuple[float, float] = (3.0, 8.0),
    max_yaw_rate_deg: float = 20.0,
) -> List[Vessel]:
    """Spawn random background vessels in a given geographic region."""
    vessels: List[Vessel] = []

    x1, y1 = env.to_local(lat_min, lon_min)
    x2, y2 = env.to_local(lat_max, lon_max)
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    for _ in range(n_vessels):
        for _ in range(200):
            x = float(np.random.uniform(x_min, x_max))
            y = float(np.random.uniform(y_min, y_max))

            if not env.is_navigable(x, y):
                continue

            lat, lon = env.to_geo(x, y)
            psi = float(np.random.uniform(-np.pi, np.pi))
            v = float(np.random.uniform(*speed_range))

            state = VesselState(lat=lat, lon=lon, psi=psi, v=v, x=x, y=y)
            params = VesselParams(max_yaw_rate=math.radians(max_yaw_rate_deg))
            vessels.append(Vessel(state, params))
            break

    return vessels


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def auto_bounds(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float, float, float]:
    """
    Compute automatic bounding box for a route.
    
    Paper-faithful auto-bounds:
    - small padding for coastal (~<150 km)
    - medium padding for regional (150–300 km)
    - reject trips that are too long for the paper's scale (>300 km)
    
    Returns:
        (lat_min, lat_max, lon_min, lon_max)
    """
    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    print(f"[auto_bounds] trip distance ~{dist_km:.1f} km")

    if dist_km < 150.0:
        pad = 0.3
        print("[auto_bounds] Using small coastal bounding box")
    elif dist_km < 300.0:
        pad = 0.5
        print("[auto_bounds] Using medium regional bounding box")
    else:
        raise RuntimeError(
            f"[auto_bounds] Trip distance {dist_km:.1f} km exceeds the "
            "intended paper-scale navigation range. Split the route into "
            "smaller legs or use a different planner."
        )

    lat_min = min(lat1, lat2) - pad
    lat_max = max(lat1, lat2) + pad
    lon_min = min(lon1, lon2) - pad
    lon_max = max(lon1, lon2) + pad

    print("Computed auto-bounds:")
    print(" lat:", lat_min, "→", lat_max)
    print(" lon:", lon_min, "→", lon_max)

    return lat_min, lat_max, lon_min, lon_max

