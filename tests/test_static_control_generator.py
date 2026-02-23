"""
Unit tests for StaticControlSeqGenerator.
"""

import math
import numpy as np
import pytest

from mpc_ship_nav.mpc.mpc_controller import StaticControlSeqGenerator


class TestStaticControlSeqGenerator:
    """Tests for StaticControlSeqGenerator class."""

    def test_generate_controls_shape(self):
        """Test that control sequences have correct shape."""
        generator = StaticControlSeqGenerator(
            max_yaw_rate=math.radians(20.0),
            horizon=16,
            num_trajectories=45,
            decay_factor=0.95
        )
        
        controls = generator.control_sequences
        assert controls.shape == (45, 16)

    def test_generate_controls_range(self):
        """Test that initial yaw rates span the correct range."""
        max_yaw_rate = math.radians(20.0)
        generator = StaticControlSeqGenerator(
            max_yaw_rate=max_yaw_rate,
            horizon=16,
            num_trajectories=45,
            decay_factor=0.95
        )
        
        controls = generator.control_sequences
        
        first_column = controls[:, 0]
        assert abs(first_column[0] - (-max_yaw_rate)) < 1e-6
        assert abs(first_column[-1] - max_yaw_rate) < 1e-6

    def test_decay_factor(self):
        """Test that yaw rate decays by decay_factor each step."""
        decay_factor = 0.95
        generator = StaticControlSeqGenerator(
            max_yaw_rate=math.radians(20.0),
            horizon=16,
            num_trajectories=5,
            decay_factor=decay_factor
        )
        
        controls = generator.control_sequences
        
        for traj_idx in range(5):
            for step in range(15):
                current = controls[traj_idx, step]
                next_val = controls[traj_idx, step + 1]
                
                if abs(current) > 1e-6:
                    actual_decay = next_val / current
                    assert abs(actual_decay - decay_factor) < 1e-6

    def test_zero_decay_factor(self):
        """Test behavior with zero decay factor."""
        generator = StaticControlSeqGenerator(
            max_yaw_rate=math.radians(20.0),
            horizon=5,
            num_trajectories=3,
            decay_factor=0.0
        )
        
        controls = generator.control_sequences
        
        for traj_idx in range(3):
            for step in range(1, 5):
                assert abs(controls[traj_idx, step]) < 1e-6

    def test_single_trajectory(self):
        """Test with single trajectory."""
        generator = StaticControlSeqGenerator(
            max_yaw_rate=math.radians(20.0),
            horizon=10,
            num_trajectories=1,
            decay_factor=0.95
        )
        
        controls = generator.control_sequences
        assert controls.shape == (1, 10)
        assert abs(controls[0, 0] - (-math.radians(20.0))) < 1e-6

    def test_horizon_one(self):
        """Test with horizon of 1."""
        generator = StaticControlSeqGenerator(
            max_yaw_rate=math.radians(20.0),
            horizon=1,
            num_trajectories=5,
            decay_factor=0.95
        )
        
        controls = generator.control_sequences
        assert controls.shape == (5, 1)
        
        first_column = controls[:, 0]
        max_yaw = math.radians(20.0)
        assert abs(first_column[0] - (-max_yaw)) < 1e-6
        assert abs(first_column[-1] - max_yaw) < 1e-6

