from __future__ import annotations

from pathlib import Path

import pytest

from franka_mujoco.env import FrankaPandaEnv
from franka_mujoco.kinematics import DampedLeastSquaresIK, MuJoCoKinematics


MODEL_PATH = Path("assets/mujoco_menagerie/franka_emika_panda/scene.xml")


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="MuJoCo Menagerie Panda assets are not installed")
def test_mujoco_kinematics_returns_7dof_shapes() -> None:
    pytest.importorskip("mujoco")
    env = FrankaPandaEnv()
    kinematics = MuJoCoKinematics(env.model, env.data, env.mujoco, env.arm)

    pose = kinematics.forward()
    jacobian = kinematics.jacobian()

    assert len(pose.position) == 3
    assert len(pose.rotation) == 3
    assert len(jacobian) == 6
    assert all(len(row) == 7 for row in jacobian)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="MuJoCo Menagerie Panda assets are not installed")
def test_position_ik_solves_small_reachable_offset() -> None:
    pytest.importorskip("mujoco")
    env = FrankaPandaEnv()
    kinematics = MuJoCoKinematics(env.model, env.data, env.mujoco, env.arm)
    ik = DampedLeastSquaresIK(kinematics, env.arm)
    pose = kinematics.forward()
    target = [pose.position[0] + 0.01, pose.position[1], pose.position[2] + 0.01]

    result = ik.solve_position(target, tolerance=5e-4)

    assert result.success
    assert len(result.q) == 7
    assert result.position_error_norm < 5e-4


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="MuJoCo Menagerie Panda assets are not installed")
def test_pose_ik_solves_reachable_pose() -> None:
    pytest.importorskip("mujoco")
    env = FrankaPandaEnv()
    kinematics = MuJoCoKinematics(env.model, env.data, env.mujoco, env.arm)
    ik = DampedLeastSquaresIK(kinematics, env.arm)
    start_q = env.arm.joint_positions()
    target_q = list(start_q)
    target_q[0] += 0.12
    target_q[1] += 0.05
    target_q[2] -= 0.08
    target_q[4] += 0.06
    target_pose = kinematics.forward(env.arm.clip_to_joint_limits(target_q))

    result = ik.solve_pose(
        target_position=target_pose.position,
        target_rotation=target_pose.rotation,
        initial_q=start_q,
        position_tolerance=1e-4,
        orientation_tolerance=1e-3,
        orientation_weight=0.4,
    )

    assert result.success
    assert len(result.q) == 7
    assert result.position_error_norm < 1e-4
    assert result.orientation_error_norm < 1e-3
