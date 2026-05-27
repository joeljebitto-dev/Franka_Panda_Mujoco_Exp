from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from franka_mujoco.env import FrankaPandaEnv  # noqa: E402
from franka_mujoco.kinematics import DampedLeastSquaresIK, MuJoCoKinematics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a small position-only IK target and command it.")
    parser.add_argument("--dx", type=float, default=0.03, help="Target x offset from the current hand pose.")
    parser.add_argument("--dy", type=float, default=-0.02, help="Target y offset from the current hand pose.")
    parser.add_argument("--dz", type=float, default=0.03, help="Target z offset from the current hand pose.")
    parser.add_argument("--duration", type=float, default=2.0, help="Viewer run duration after solving.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the viewer.")
    args = parser.parse_args()

    env = FrankaPandaEnv()
    kinematics = MuJoCoKinematics(env.model, env.data, env.mujoco, env.arm)
    ik = DampedLeastSquaresIK(kinematics, env.arm)

    current_pose = kinematics.forward()
    target = [
        current_pose.position[0] + args.dx,
        current_pose.position[1] + args.dy,
        current_pose.position[2] + args.dz,
    ]
    result = ik.solve_position(target)

    print("target position [m]:", [round(value, 5) for value in target])
    print("success:", result.success)
    print("iterations:", result.iterations)
    print("position error norm [m]:", f"{result.position_error_norm:.6f}")
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
