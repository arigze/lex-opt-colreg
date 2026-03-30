from __future__ import annotations

import math
import torch
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from mpc_ship_nav.charts.environment import ChartEnvironment
from mpc_ship_nav.dynamics.vessel import Vessel, VesselState
from mpc_ship_nav.sim.engine import Controller


@dataclass
class MPCConfig:
    """Configuration for the simplified MPC controller."""
    dt: float
    horizon: int = 16                  # H (paper: Section 2.6.3)
    n_candidates: int = 45             # M (paper: Section 2.6.3)
    max_delta_heading_deg: float = 20  # Δθ_max per step (paper: ±20°)
    collision_radius_nm: float = 0.5   # d_collision
    colreg_radius_nm: float = 3.0      # d_COLREG

    @property
    def max_delta_heading(self) -> float:
        """Maximum heading increment per step [rad]."""
        return math.radians(self.max_delta_heading_deg)

    @property
    def max_yaw_rate(self) -> float:
        """Corresponding yaw-rate limit [rad/s]. u * dt = Δθ."""
        return self.max_delta_heading / self.dt

    @property
    def collision_radius(self) -> float:
        """Collision radius [m]."""
        return self.collision_radius_nm * 1852.0

    @property
    def colreg_radius(self) -> float:
        """COLREG zone radius [m]."""
        return self.colreg_radius_nm * 1852.0


class WaypointRoute:
    """Waypoint manager in LOCAL (x,y) coordinates."""
    def __init__(self, waypoints_xy: np.ndarray, transition_radius: float):
        self.waypoints_xy = waypoints_xy  # List of waypoints in (x, y)
        self.transition_radius = transition_radius  # When within this radius, move to next waypoint
        self.idx = 0  

    def current_waypoint(self, own: VesselState) -> Tuple[float, float]:
        """Return the current waypoint and advance when inside transition radius."""
        n = len(self.waypoints_xy)
        if n == 0:
            # If no waypoints, return current position
            return own.x, own.y

        if self.idx >= n:
            self.idx = n - 1

        wp_x, wp_y = self.waypoints_xy[self.idx]
        dx = own.x - wp_x
        dy = own.y - wp_y
        dist = math.hypot(dx, dy)

        if dist <= self.transition_radius and self.idx < n - 1:
            # If close enough, advance to next waypoint
            self.idx += 1
            wp_x, wp_y = self.waypoints_xy[self.idx]

        return float(wp_x), float(wp_y)

    def is_finished(self) -> bool:
        return self.idx >= len(self.waypoints_xy) - 1

class StaticControlSeqGenerator:
    """Generates a set of static trajectories based max yaw rate, horizon and number of trajectories."""
    
    def __init__(self, max_yaw_rate: float=np.radians(20), horizon: int=20, num_trajectories: int=45, decay_factor: float=0.95):
        """
        Args:
            max_yaw_rate (float, optional): the maximum yaw rate in radians per second. Defaults to np.radians(20).
            horizon (int, optional): the number of time steps in the trajectory horizon. Defaults to 20.
            num_trajectories (int, optional): the number of trajectories to generate. Defaults to 45.
            decay_factor (float, optional): the factor by which the yaw rate decays at each time step. Defaults to 0.95.
        """
        self.max_yaw_rate = max_yaw_rate
        self.horizon = horizon
        self.num_trajectories = num_trajectories
        self.decay_factor = decay_factor
        self.control_sequences = self.generate_controls()
        
        
    def generate_controls(self) -> np.ndarray[np.ndarray[np.float64]]:
        initial_angles = np.linspace(-self.max_yaw_rate, self.max_yaw_rate, self.num_trajectories)
        controls = np.zeros((self.num_trajectories, self.horizon), dtype=np.float64)
        print("Generating trajectories with control shape:")
        print(controls.shape)

        # 2. Compute the sequence for each trajectory
        for i, start_angle in enumerate(initial_angles):
            dtheta = start_angle

            for h in range(self.horizon):
                controls[i, h] = dtheta

                # Apply decay for the next step (smoothing)
                dtheta *= self.decay_factor
        return controls 


class SimplifiedMPCController(Controller):
    """
    Simplified fan-based MPC controller as in Sec. 2.6.3:
    - Generate M constant-turn trajectories over H steps.
    - Discard those that violate d_collision to static/dynamic obstacles.
    - Select trajectory based on heading/endpoint relative to next waypoint.
    - Apply only first control input (receding horizon).
    """

    def __init__(self, dt: float, waypoints_xy: np.ndarray, vis_scale: int = 100) -> None:
        self.cfg = MPCConfig(dt=dt)
        self.route = WaypointRoute(
            waypoints_xy=np.asarray(waypoints_xy, dtype=float),
            transition_radius=self.cfg.collision_radius,
        )
        self.control_sequences = StaticControlSeqGenerator(self.cfg.max_yaw_rate, self.cfg.horizon, self.cfg.n_candidates).control_sequences
        self.u_candidates = self.control_sequences[:, 0]
        self.vis_scale = vis_scale  # for trajectory visualization
        self.active_encounters = {}  # {target_id: (encounter_type, timestamp)}

    def compute_control(
        self,
        t: float,
        own_ship: Vessel,
        other_vessels: List[Vessel],
        env: ChartEnvironment,
    ) -> Tuple[Tuple[float, int], Tuple[np.ndarray, np.ndarray]]:
        own = own_ship.state  

        if own.x is None or own.y is None:
            own.x, own.y = env.to_local(own.lat, own.lon)

        # 1) Get the current waypoint in local coords, possibly advance
        wp_x, wp_y = self.route.current_waypoint(own)

        if self.route.is_finished():
            # Return 0.0 yaw rate, index 0, and empty debug info to satisfy unpacking in engine.py
            return ((0.0, 0), (np.array([]), np.array([])))

        # 2) Calculate bearing to waypoint (target heading)
        dx = wp_x - own.x
        dy = wp_y - own.y
        theta_target = math.atan2(dy, dx)

        # 3) Collect dynamic obstacles within COLREG zone
        dyn_states: List[VesselState] = []
        for v in other_vessels:
            s = v.state
            if s.x is None or s.y is None:
                s.x, s.y = env.to_local(s.lat, s.lon)

            ddx = s.x - own.x
            ddy = s.y - own.y
            if ddx * ddx + ddy * ddy <= self.cfg.colreg_radius ** 2:
                dyn_states.append(s)

        # 4) Precompute predicted dynamic trajectories (constant velocity)
        dyn_trajs = self._predict_dynamic(dyn_states)

        # 5) Generate candidate yaw-rates and simulate own trajectories
        # Paper: M candidate trajectories with yaw rates in [-max_yaw_rate, max_yaw_rate]
        u_candidates = np.linspace(
            -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate, self.cfg.n_candidates
        )
        feasible_mask, own_trajs, own_trajs_vis = self._simulate_and_filter(
            own, dyn_trajs, env
        )

        if not np.any(feasible_mask):
            # fallback: choose u closest to 0 (maintain current heading as safest option)
            # This matches the paper's approach when no feasible trajectory exists
            idx = np.argmin(np.abs(u_candidates))
            return ((float(u_candidates[idx]), idx), (feasible_mask, own_trajs_vis))

        # 6) Calculate lambda-ladder
        collisions = self._is_colliding(feasible_mask)
        print("Collis:", collisions)
        colreg_violations = self._respects_colreg_rules(own_ship, other_vessels, u_candidates)
        print("COLREG:", colreg_violations)
        path_following_scores = self._path_following_scores((wp_x, wp_y), own_trajs)
        normalized_path_scores = self._normalize_scores(path_following_scores)
        print("Path scores:", normalized_path_scores)
        lambda_ladder = self._lambda_ladder(collisions, colreg_violations, normalized_path_scores)
        print("Lambda-ladder values:", lambda_ladder)
        idx = np.argmin(lambda_ladder)
        print(f"Selected trajectory index: {idx} and value: {lambda_ladder[idx]}")

        return (float(u_candidates[idx]), idx), (feasible_mask, own_trajs_vis)

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi


    # ------------------------------------------------------------------
    # MPC internals
    # ------------------------------------------------------------------

    def _simulate_and_filter(
        self,
        own: VesselState,
        dyn_trajs: List[np.ndarray],
        env: ChartEnvironment,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate own-ship trajectories for each yaw-rate candidate and
        return:
            feasible_mask: shape (M,), bool
            own_trajs:     shape (M, H, 2) (points over horizon)
        """
        H = self.cfg.horizon
        dt = self.cfg.dt
        v = own.v
        M = self.cfg.n_candidates
        
        sequence_by_traj = self.control_sequences 
        own_trajs_vis = np.zeros((M, H, 2), dtype=float)
        own_trajs = np.zeros((M, H, 2), dtype=float)
        feasible = np.ones(M, dtype=bool)

        for m in range(M):
            px = own.x
            py = own.y
            psi = own.psi
            px_vis = own.x
            py_vis = own.y
            psi_vis = own.psi
            sequence = sequence_by_traj[m]
            for h in range(H):
                u = sequence[h]
                psi_vis = self._wrap_angle(psi_vis + u * dt)
                px_vis += v * math.cos(psi_vis) * dt * self.vis_scale
                py_vis += v * math.sin(psi_vis) * dt * self.vis_scale
                own_trajs_vis[m, h, 0] = px_vis
                own_trajs_vis[m, h, 1] = py_vis
                
            for h in range(H):
                u = sequence[h]
                psi = self._wrap_angle(psi + u * dt)
                px += v * math.cos(psi) * dt
                py += v * math.sin(psi) * dt
                own_trajs[m, h, 0] = px
                own_trajs[m, h, 1] = py

                # --- static obstacle check (land) ---
                if not env.is_navigable(px, py):
                    feasible[m] = False
                    break

                # --- dynamic collision check ---
                for traj_other in dyn_trajs:
                    if h >= traj_other.shape[0]:
                        continue
                    ox, oy = traj_other[h]
                    dist = math.hypot(px - ox, py - oy)
                    if dist < self.cfg.collision_radius:
                        feasible[m] = False
                        break

                if not feasible[m]:
                    break

        return feasible, own_trajs, own_trajs_vis

    def _predict_dynamic(self, dyn_states: List[VesselState]) -> List[np.ndarray]:
        """Predict constant-velocity paths for dynamic obstacles over the horizon."""
        H = self.cfg.horizon
        dt = self.cfg.dt

        dyn_trajs: List[np.ndarray] = []

        for s in dyn_states:
            px = s.x
            py = s.y
            psi = s.psi
            v = s.v

            traj = np.zeros((H, 2), dtype=float)
            for h in range(H):
                px += v * math.cos(psi) * dt
                py += v * math.sin(psi) * dt
                traj[h, 0] = px
                traj[h, 1] = py

            dyn_trajs.append(traj)

        return dyn_trajs

    def _select_best(
        self,
        own: VesselState,
        waypoint_xy: Tuple[float, float],
        theta_target: float,
        u_candidates: np.ndarray,
        own_trajs: np.ndarray,        # shape (M, H, 2)
        feasible_mask: np.ndarray,    # shape (M,)
        dyn_states: List[VesselState] = None,
    ) -> int:
        """
        Select optimal trajectory according to paper equations (15-17):
        - If aligned (|θ(t) - θ_Target| ≤ 90°): minimize |θ_m(1) - θ_Target|
        - Otherwise: minimize ||x_m(H) - w_i+1||
        """
        H = self.cfg.horizon
        dt = self.cfg.dt

        idxs = np.where(feasible_mask)[0]
        if idxs.size == 0:
            return int(np.argmin(np.isfinite(feasible_mask)))

        # -----------------------------
        # Helper
        # -----------------------------
        def ang_diff(a, b):
            return abs(self._wrap_angle(a - b))

        # -----------------------------
        # Alignment rule (Equation 15)
        # -----------------------------
        angle_err_now = self._wrap_angle(theta_target - own.psi)
        aligned = abs(angle_err_now) <= math.radians(90.0)

        best_idx = int(idxs[0])
        best_score = float("inf")

        if aligned:
            # Equation (16): minimize |θ_m(1) - θ_Target|
            for m in idxs:
                u = float(u_candidates[m])
                psi1 = self._wrap_angle(own.psi + u * dt)
                score = ang_diff(theta_target, psi1)

                if score < best_score:
                    best_score = score
                    best_idx = int(m)

            return best_idx

        else:
            # Equation (17): minimize ||x_m(H) - w_i+1||
            wp_x, wp_y = waypoint_xy

            for m in idxs:
                end_x, end_y = own_trajs[m, H - 1]
                score = math.hypot(end_x - wp_x, end_y - wp_y)

                if score < best_score:
                    best_score = score
                    best_idx = int(m)

            return best_idx


    # ------------------------------------------------------------------
    # Reward/cost functions
    # ------------------------------------------------------------------

    def _is_colliding(self, feasible_mask) -> np.ndarray:
        '''
        Check if there is a collision with land or other vessel in the trajectory.
        Returns 1 if colliding (infeasible), 0 if not colliding (feasible).
        '''
        return (1 - feasible_mask).astype(int)

    def _respects_colreg_rules(self, own_ship: Vessel, other_vessels: List[Vessel], u_candidates: np.ndarray) -> np.ndarray:
        '''
        Check if the colreg rules are respected by the trajectory.
        Only considers vessels within COLREG radius.
        Returns 1 if COLREG violation, 0 if respected.
        '''
        M = u_candidates.shape[0]
        violations = np.ones(M, dtype=int)

        for m in range(M):
            has_nearby_vessels = False
            for target in other_vessels:
                distance = math.hypot(target.state.x - own_ship.state.x, target.state.y - own_ship.state.y)
                if distance <= self.cfg.colreg_radius:
                    has_nearby_vessels = True

                    # Relative bearing
                    relative_bearing = self._relative_bearing(own_ship.state, target.state)

                    # Encouter
                    encounter = self._classify_encounter(own_ship, target)

                    # Check violation
                    violations[m] = 0 if self._compute_control(encounter, relative_bearing, u_candidates[m]) else 1

                    # TODO : Do we need this patch?
                    # Special case: crossing-starboard allows the smallest candidate to be compliant in case target is too far starboard
                    # This allows to turn starboard (right) even if the relative_bearing is too small
                    if encounter == "crossing-starboard" and m == 0:
                        violations[m] = 0  # No violation for first candidate in crossing-starboard

            if not has_nearby_vessels:
                violations[m] = 0  # No violations if no vessels in COLREG radius
        
        return violations

    def _path_following_scores(self, waypoint_xy: Tuple[float, float], own_trajs: np.ndarray) -> np.ndarray:
        """
        Calculate path following scores as distance from trajectory endpoints to waypoint.
        Lower scores indicate better path following.
        """
        wp_x, wp_y = waypoint_xy
        scores = np.zeros(own_trajs.shape[0], dtype=float)
        
        for m in range(own_trajs.shape[0]):
            end_x, end_y = own_trajs[m, -1]
            scores[m] = math.hypot(end_x - wp_x, end_y - wp_y)
        
        return scores

    def _lambda_ladder(self, collisions, colreg_violations, path_following_scores) -> np.ndarray:
        '''
        Compute lambda-ladder loss function.
        Uses a hierarchical priority: collisions > COLREG violations > path following.
        '''
        # Parameters for the lambda-ladder (can be tuned)
        c = 10.0  # Scaling factor
        delta = 0.1  # Priority gap
        
        # Lambda-ladder formulation: log-sum-exp of prioritized costs
        level0 = collisions  # Highest priority: avoid collisions
        level1 = colreg_violations  # Medium priority: respect COLREG
        level2 = path_following_scores  # Lowest priority: follow path
        
        costs = np.array([level0, level1, level2])
        priorities = np.array([0, delta, 2 * delta])  # Collisions (0), COLREG (delta), Path (2*delta)
        
        # For each trajectory, compute the lambda-ladder value
        lambda_values = np.zeros(collisions.shape[0])
        for m in range(collisions.shape[0]):
            exponents = -c * (priorities + costs[:, m])
            lambda_values[m] = -torch.logsumexp(torch.tensor(exponents), dim=0) / c
        
        return lambda_values

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range for fair combination."""
        if np.all(scores == scores[0]):
            return np.zeros_like(scores)  # Avoid division by zero if all values are identical
        min_score = np.min(scores)
        max_score = np.max(scores)
        return (scores - min_score) / (max_score - min_score)

    def _classify_encounter(self, own: Vessel, target: Vessel) -> str:
        """Classify encounter type based on stored active encounters."""
        # Distance
        distance = self._distance(own, target)

        # Check if we already have an active encounter type for this target
        target_id = id(target)
        if target_id in self.active_encounters:
            # Check if still within COLREG zone
            if distance <= self.cfg.colreg_radius:
                return self.active_encounters[target_id]  # Use stored encounter type
            else:
                del self.active_encounters[target_id]  # Clear when outside zone

        # First detection - classify and store
        encounter = self._classify_new_encounter(own, target)
        self.active_encounters[target_id] = encounter
        return encounter

    def _classify_new_encounter(self, own: Vessel, target: Vessel) -> str:
        """Classify encounter type based on relative bearing and motion."""
        # Relative bearing
        relative_bearing = self._relative_bearing(own, target)

        # Heading difference
        heading_diff = self._heading_difference(own, target)

        # Classify encounter type based on relative bearing and heading difference
        if abs(relative_bearing) < math.pi/12 and abs(heading_diff) > 11*math.pi/12:
            return "head-on" # Head-on encounter
        elif abs(relative_bearing) < math.pi/6 and abs(heading_diff) < math.pi/6:
            return "overtaking" # Overtaking from behind
        elif abs(relative_bearing) > 5*math.pi/6 and abs(heading_diff) < math.pi/6:
            return "overtaken" # Being overtaken
        elif -5*math.pi/8 < relative_bearing < 0:
            return "crossing-starboard" # Crossing from starboard
        elif 0 < relative_bearing < 5*math.pi/8:
            return "crossing-port" # Crossing from port
        else:
            return "none" # No significant encounter (e.g., far away or perpendicular paths)

    def _compute_control(self, encounter_type: str, relative_bearing: float, u_candidate: np.ndarray) -> None:
        """Determine if the candidate control input respects COLREG rules based on encounter type and relative bearing."""
        # Check if candidate control respects COLREG for this target
        if encounter_type in ["head-on", "crossing-starboard"]:
            if u_candidate < relative_bearing:  # Must turn to starboard (right)
                return True  # No violation
        elif encounter_type == "overtaking":
            # TODO : This makes the vessel take a turn in both directions, but we don't need the angle to be bigger than the relative bearing on both sides
            # We need to allow the vessel smaller angles on the side where the other boat is not present
            if abs(u_candidate) > relative_bearing:  # Must turn starboard (right) or port (left), both are valid
                return True  # No violation
        elif encounter_type in ["crossing-port", "overtaken", "none"]:
            return True  # No violation

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_angle(a: float) -> float:
        # Wrap angle to [-pi, pi]
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _relative_bearing(self, own, target) -> float:
        """Calculate relative bearing from own ship to target."""
        if isinstance(own, Vessel):
            own = own.state
        if isinstance(target, Vessel):
            target = target.state

        # Own ship
        own_x = own.x
        own_y = own.y
        own_psi = own.psi

        # Target ship
        target_x = target.x
        target_y = target.y

        # Relative bearing
        dx = target_x - own_x
        dy = target_y - own_y
        relative_bearing = math.atan2(dy, dx) - own_psi
        return self._wrap_angle(relative_bearing)

    def _heading_difference(self, own, target) -> float:
        """Calculate heading difference between own ship and target."""
        if isinstance(own, Vessel):
            own = own.state
        if isinstance(target, Vessel):
            target = target.state

        # Own ship
        own_psi = own.psi

        # Target ship
        target_psi = target.psi

        # Heading difference
        return self._wrap_angle(own_psi - target_psi)

    def _distance(self, own, target) -> float:
        """Calculate distance between own ship and target."""
        if isinstance(own, Vessel):
            own = own.state
        if isinstance(target, Vessel):
            target = target.state

        # Own ship
        own_x = own.x
        own_y = own.y

        # Target ship
        target_x = target.x
        target_y = target.y

        return math.hypot(target_x - own_x, target_y - own_y)