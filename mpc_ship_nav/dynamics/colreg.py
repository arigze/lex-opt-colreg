"""
COLREG compliance logic for target ships.

Target ships follow these rules explicitly, while the own ship (MPC-controlled)
avoids collisions implicitly through its optimization constraints.
"""

import numpy as np
from .vessel import Vessel

# Paper values (jmse-13-01246): dCollision=0.5 nm, dCOLREG=3.0 nm
NM = 1852.0
D_COLLISION = 0.5 * NM  # 926 m
D_COLREG = 3.0 * NM     # 5556 m


class COLREGLogic:
    """
    COLREG compliance logic for target ships (Rule 13/14/15).

    The target ship will generate a yaw rate command when it must give way;
    otherwise it maintains course (returns 0.0).
    """

    def __init__(self, collision_threshold: float = D_COLREG):
        """
        Args:
            collision_threshold: Distance threshold for collision risk [m]
        """
        self.collision_threshold = collision_threshold
        self._overtaking_original_headings: dict = {}

    def compute_target_control(self, target: Vessel, own_ship: Vessel) -> float:
        """
        Compute control for a target ship based on COLREG rules.

        Args:
            target: The target ship to control
            own_ship: The own ship (autonomous vessel)

        Returns:
            u: yaw rate command [rad/s]
        """
        target_id = id(target)
        distance = self._distance(target, own_ship)

        # Emergency collision avoidance (< 0.5 nm)
        in_collision_zone = distance <= D_COLLISION
        if in_collision_zone:
            risk = self._risk_of_collision(target, own_ship)
            if risk:
                sign = self._emergency_turn_direction(target, own_ship)
                return sign * target.params.max_yaw_rate

        # COLREG compliance (0.5-3.0 nm) 
        in_colreg_radius = distance <= self.collision_threshold
        risk_of_collision = self._risk_of_collision(target, own_ship) if in_colreg_radius else False

        if target_id in self._overtaking_original_headings:
            if not in_colreg_radius:

                original_heading = self._overtaking_original_headings[target_id]
                current_heading = target.state.psi

                heading_error = self._wrap_angle(current_heading - original_heading)

                if abs(heading_error) < np.radians(5.0):
                    del self._overtaking_original_headings[target_id]
                    return 0.0

                if heading_error > 0:
                    return -target.params.max_yaw_rate  
                else:
                    return target.params.max_yaw_rate  
            # If still in COLREG radius, continue with normal COLREG logic below

        if not risk_of_collision:
            return 0.0

        encounter = self._classify_encounter(target, own_ship)

        if self._target_must_give_way(encounter, target, own_ship):
            # For overtaking encounters, store original heading if not already stored
            if encounter == "overtaking" and target_id not in self._overtaking_original_headings:
                self._overtaking_original_headings[target_id] = target.state.psi

            sign = self._turn_direction(encounter, target, own_ship)
            return sign * target.params.max_yaw_rate

        return 0.0

    def _distance(self, ship1: Vessel, ship2: Vessel) -> float:
        """Euclidean distance between two vessels in meters."""
        dx = ship2.state.x - ship1.state.x
        dy = ship2.state.y - ship1.state.y
        return np.hypot(dx, dy)

    def _risk_of_collision(self, target: Vessel, own_ship: Vessel) -> bool:
        """
        Determine if there is a risk of collision using BOTH closing and crossing speed.

        Logic:
        1. High crossing speed → Ships will miss → No risk
        2. Low crossing speed + approaching → Collision course → Risk
        3. Low crossing speed + separating → No risk

        Returns True if there is risk, False otherwise.
        """
        distance = self._distance(target, own_ship)
        if distance > self.collision_threshold:
            return False

        dx = own_ship.state.x - target.state.x
        dy = own_ship.state.y - target.state.y

        vx_t = target.state.v * np.cos(target.state.psi)
        vy_t = target.state.v * np.sin(target.state.psi)
        vx_o = own_ship.state.v * np.cos(own_ship.state.psi)
        vy_o = own_ship.state.v * np.sin(own_ship.state.psi)

        dvx = vx_o - vx_t
        dvy = vy_o - vy_t

        # Closing speed (negative = approaching)
        closing_speed = (dvx * dx + dvy * dy) / distance

        cross_product = dvx * dy - dvy * dx
        crossing_speed = abs(cross_product) / distance

        # Thresholds
        CROSSING_SPEED_THRESHOLD = 2.0  # m/s (high = will miss)
        CLOSING_SPEED_THRESHOLD = -0.5  # m/s (negative = approaching)

        # High crossing speed → ships will miss each other
        if crossing_speed > CROSSING_SPEED_THRESHOLD:
            return False

        if closing_speed < CLOSING_SPEED_THRESHOLD:
            return True 

        return False

    def _classify_encounter(self, target: Vessel, own_ship: Vessel) -> str:
        """
        Classify the type of encounter according to COLREG.

        Returns:
            'head_on', 'crossing', 'overtaking', or 'none'
        """
        rel_bearing = self._relative_bearing(target, own_ship)

        heading_diff = abs(own_ship.state.psi - target.state.psi)
        heading_diff = min(heading_diff, 2 * np.pi - heading_diff)

        # Head-on: nearly reciprocal courses and own ship in target's forward cone
        if heading_diff > np.radians(170) and abs(rel_bearing) < np.radians(20):
            return "head_on"

        # Overtaking: target is faster and own ship is in target's stern sector
        rel_bearing_own = self._relative_bearing(own_ship, target)
        if target.state.v > own_ship.state.v and abs(rel_bearing_own) > np.radians(112.5):
            return "overtaking"

        # Crossing: intersecting paths with appreciable heading difference
        if np.radians(5) < heading_diff < np.radians(170):
            return "crossing"

        return "none"

    def _target_must_give_way(self, encounter: str, target: Vessel, own_ship: Vessel) -> bool:
        """Return True if the target ship must maneuver."""
        if encounter == "head_on":
            return True

        if encounter == "crossing":
            rel_bearing = self._relative_bearing(target, own_ship)
            return -np.radians(112.5) < rel_bearing <= 1e-6

        if encounter == "overtaking":
            rel_bearing_own = self._relative_bearing(own_ship, target)
            return target.state.v > own_ship.state.v and abs(rel_bearing_own) > np.radians(112.5)

        return False

    def _turn_direction(self, encounter: str, target: Vessel, own_ship: Vessel) -> int:
        """
        Determine turn direction respecting COLREG while using velocity awareness.

        COLREG rules take priority. Velocity vectors used for:
        - Validation that COLREG direction is safe

        Returns +1 for port (left) turn, -1 for starboard (right) turn.
        """
        # TODO : This function always returns -1, something seems wrong
        dx = own_ship.state.x - target.state.x
        dy = own_ship.state.y - target.state.y
        d_magnitude = np.hypot(dx, dy)

        vx_target = target.state.v * np.cos(target.state.psi)
        vy_target = target.state.v * np.sin(target.state.psi)
        vx_own = own_ship.state.v * np.cos(own_ship.state.psi)
        vy_own = own_ship.state.v * np.sin(own_ship.state.psi)

        dvx = vx_own - vx_target
        dvy = vy_own - vy_target

        closing_speed = (dvx * dx + dvy * dy) / d_magnitude

        cross_product = dvx * dy - dvy * dx

        velocity_preference = -1 if cross_product > 0 else 1 if cross_product < 0 else -1

        # Apply COLREG constraints (STRICT PRIORITY)

        if encounter == "head_on":
            # COLREG Rule 14: Both vessels must turn starboard
            return -1  # Always starboard

        if encounter == "overtaking":
            # COLREG Rule 13: Overtaking vessel keeps clear
            return -1  

        # Crossing encounter (COLREG Rule 15)
        # Give-way vessel (determined by _target_must_give_way) turns starboard

        if abs(closing_speed) < 0.5:
            if velocity_preference == -1:
                return -1 
            else:
                return -1

        # COLREG Rule 15: Give-way vessel turns starboard
        return -1  

    def _emergency_turn_direction(self, target: Vessel, own_ship: Vessel) -> int:
        """
        Choose turn direction based on closing speed and relative velocity.
        No COLREG priority - pure collision avoidance.

        Returns +1 for port (left) turn, -1 for starboard (right) turn.
        """
        dx = own_ship.state.x - target.state.x
        dy = own_ship.state.y - target.state.y
        d_magnitude = np.hypot(dx, dy)

        vx_target = target.state.v * np.cos(target.state.psi)
        vy_target = target.state.v * np.sin(target.state.psi)
        vx_own = own_ship.state.v * np.cos(own_ship.state.psi)
        vy_own = own_ship.state.v * np.sin(own_ship.state.psi)

        dvx = vx_own - vx_target
        dvy = vy_own - vy_target

        closing_speed = (dvx * dx + dvy * dy) / d_magnitude

        cross_product = dvx * dy - dvy * dx

        CLOSING_SPEED_THRESHOLD = 0.5  # m/s

        # Edge case 1: Low closing speed (parallel/perpendicular courses)
        if abs(closing_speed) < CLOSING_SPEED_THRESHOLD:
            rel_bearing = self._relative_bearing(target, own_ship)
            return 1 if rel_bearing < 0 else -1

        # Edge case 2: Already separating
        if closing_speed > 0:
            rel_bearing = self._relative_bearing(target, own_ship)
            return 1 if rel_bearing < 0 else -1

        # Main case: Approaching - use velocity-based decision
        if cross_product > 0:
            return -1  # Turn starboard (right)
        elif cross_product < 0:
            return 1   # Turn port (left)
        else:
            # Exact head-on (rare)
            rel_bearing = self._relative_bearing(target, own_ship)
            return 1 if rel_bearing < 0 else -1

    def _relative_bearing(self, observer: Vessel, target: Vessel) -> float:
        """
        Relative bearing from observer to target in radians, normalized to [-π, π].
        0 = dead ahead of observer; +π/2 = starboard; -π/2 = port; ±π = astern.
        """
        dx = target.state.x - observer.state.x
        dy = target.state.y - observer.state.y

        bearing_absolute = np.arctan2(dy, dx)
        relative = bearing_absolute - observer.state.psi
        return np.arctan2(np.sin(relative), np.cos(relative))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        return np.arctan2(np.sin(angle), np.cos(angle))
