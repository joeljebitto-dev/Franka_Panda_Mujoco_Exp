from __future__ import annotations

import pytest

from franka_mujoco.trajectory import JointTrajectory


def test_cubic_trajectory_hits_endpoints() -> None:
    start = [0.0] * 7
    goal = [0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    trajectory = JointTrajectory(start, goal, duration_s=2.0, method="cubic")

    assert trajectory.sample(0.0).q == start
    assert trajectory.sample(2.0).q == goal
    assert trajectory.sample(0.0).qd == [0.0] * 7
    assert trajectory.sample(2.0).qd == [0.0] * 7


def test_quintic_midpoint_is_halfway_for_symmetric_move() -> None:
    trajectory = JointTrajectory([0.0] * 7, [1.0] * 7, duration_s=4.0, method="quintic")

    midpoint = trajectory.sample(2.0)

    assert midpoint.q == pytest.approx([0.5] * 7)


def test_trajectory_requires_seven_joints() -> None:
    with pytest.raises(ValueError):
        JointTrajectory([0.0] * 6, [1.0] * 6, duration_s=1.0)
