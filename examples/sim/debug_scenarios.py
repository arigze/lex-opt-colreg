# debug_scenarios.py
from grid_search_mppi import run_one, SCENARIO_NAMES, _make_scenario_and_route
from mpc_ship_nav.mpc.mpc_controller import SimplifiedMPCController
from mpc_ship_nav.sim.engine import SimConfig, Simulator
from mpc_ship_nav.dynamics.traffic import Scenario

for s in SCENARIO_NAMES:
    env, scenario, route, own_ship = _make_scenario_and_route(s)
    ctrl = SimplifiedMPCController(dt=1.0, waypoints_xy=route, sigma=None, rho=0.8, lambd=0.01)
    sim = Simulator(env, scenario, ctrl, SimConfig(dt=1.0, t_final=60.0, log_interval=5))
    sim.run()
    mean = sum(ctrl.cost_log)/len(ctrl.cost_log) if ctrl.cost_log else "EMPTY"
    print(f"{s:25s}  cost_log len={len(ctrl.cost_log)}  mean={mean}")