"""
Test that the COLREG rule following is prioritized over path following.
"""

import math
from pathlib import Path
from typing import List, Tuple

from scenario_utils import (
    build_open_water_env,
    check_collisions,
    create_standard_parser,
    make_vessel_at_xy,
    run_scenario,
    setup_standard_scenario,
    plot_results,
    save_animation,
)


def main():
    parser = create_standard_parser()
    args = parser.parse_args()

    env = build_open_water_env()
    route: List[Tuple[float, float]] = [(0.0, 0.0), (10_000.0, 0.0), (20_000.0, 0.0)]
    own_ship, controller, sim_cfg = setup_standard_scenario(
        env, route=route, dt=args.dt, t_final=args.t_final, log_interval=args.log_interval
    )

    # Fix traffic position after simulator init (ensures correct spawn position)
    log, _ = run_scenario(
        env=env, own_ship=own_ship, controller=controller, sim_cfg=sim_cfg,
        traffic_vessels=[]
    )

    collision_occurred, collision_time, min_dist = check_collisions(log)
    final_time = log.times[-1] if log.times else 0.0

    if not args.no_plot:
        plot_results(env, log, route)

    if args.animate:
        out_path = Path(__file__).resolve().parent / "test_colreg_headon.gif"
        save_animation(env, log, route, out_path)

    # Check collision and return appropriate exit code
    if collision_occurred:
        print(f"ERROR: Collision at t={collision_time:.1f}s, min_d={min_dist:.1f}m")
        return 1
    else:
        print(f"OK: No collision, min_d={min_dist:.1f}m, simulation ended at t={final_time:.1f}s")
        return 0


if __name__ == "__main__":
    exit(main())

