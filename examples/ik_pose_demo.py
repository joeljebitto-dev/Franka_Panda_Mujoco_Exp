from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from franka_mujoco.env import FrankaPandaEnv  # noqa: E402
from franka_mujoco.kinematics import DampedLeastSquaresIK, MuJoCoKinematics  # noqa: E402


DEFAULT_TARGET_OFFSETS = [1, 0.05, -0.08, 0.0, 0.06, 0.0, 0.0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve full 7-DOF end-effector pose IK and command the result."
    )
    parser.add_argument(
        "--offsets",
        type=float,
        nargs=7,
        default=DEFAULT_TARGET_OFFSETS,
        metavar=("DQ1", "DQ2", "DQ3", "DQ4", "DQ5", "DQ6", "DQ7"),
        help="Joint offsets used only to generate a reachable target pose.",
    )
    parser.add_argument(
        "--duration", type=float, default=2.0, help="Viewer run duration after solving."
    )
    parser.add_argument("--position-tolerance", type=float, default=1e-4)
    parser.add_argument("--orientation-tolerance", type=float, default=1e-3)
    parser.add_argument("--orientation-weight", type=float, default=0.4)
    parser.add_argument(
        "--headless", action="store_true", help="Run without opening the viewer."
    )
    args = parser.parse_args()

    env = FrankaPandaEnv()
    kinematics = MuJoCoKinematics(env.model, env.data, env.mujoco, env.arm)
    ik = DampedLeastSquaresIK(kinematics, env.arm)

    start_q = env.arm.joint_positions()
    target_q = env.arm.clip_to_joint_limits(
        [q + dq for q, dq in zip(start_q, args.offsets)]
    )
    target_pose = kinematics.forward(target_q)
    result = ik.solve_pose(
        target_position=target_pose.position,
        target_rotation=target_pose.rotation,
        initial_q=start_q,
        position_tolerance=args.position_tolerance,
        orientation_tolerance=args.orientation_tolerance,
        orientation_weight=args.orientation_weight,
    )

    print("generated target q [rad]:", [round(value, 5) for value in target_q])
    print("target position [m]:", [round(value, 5) for value in target_pose.position])
    print("success:", result.success)
    print("iterations:", result.iterations)
    print("position error norm [m]:", f"{result.position_error_norm:.6f}")
    print("orientation error norm [rad]:", f"{result.orientation_error_norm:.6f}")
    print("weighted error norm:", f"{result.weighted_error_norm:.6f}")
    print("q solution [rad]:", [round(value, 5) for value in result.q])

    if not result.success:
        return

    env.arm.command_joint_positions(result.q)
    if args.headless:
        run(env, duration=args.duration, viewer=None)
        return

    from mujoco import viewer as mujoco_viewer  # type: ignore

    with mujoco_viewer.launch_passive(env.model, env.data) as viewer:
        run(env, duration=args.duration, viewer=viewer)


def run(env: FrankaPandaEnv, duration: float, viewer) -> None:
    start_time = env.time
    next_wall_time = time.time()
    while env.time - start_time < duration:
        if viewer is not None and not viewer.is_running():
            break

        env.step()
        if viewer is not None:
            viewer.sync()

        next_wall_time += env.timestep
        sleep_s = next_wall_time - time.time()
        if sleep_s > 0.0:
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()
