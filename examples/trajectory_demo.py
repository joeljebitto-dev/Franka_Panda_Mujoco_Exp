from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from franka_mujoco.env import FrankaPandaEnv  # noqa: E402
from franka_mujoco.trajectory import JointTrajectory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a smooth 7-DOF joint-space trajectory.")
    parser.add_argument("--duration", type=float, default=4.0, help="Trajectory duration in seconds.")
    parser.add_argument("--method", choices=("linear", "cubic", "quintic"), default="cubic")
    parser.add_argument("--headless", action="store_true", help="Run without opening the viewer.")
    args = parser.parse_args()

    env = FrankaPandaEnv()
    start_q = env.arm.joint_positions()
    goal_q = list(start_q)
    goal_q[0] += 0.35
    goal_q[2] -= 0.25
    goal_q[4] += 0.20
    goal_q = env.arm.clip_to_actuator_ranges(goal_q)
    trajectory = JointTrajectory(start_q=start_q, goal_q=goal_q, duration_s=args.duration, method=args.method)

    print("start q:", [round(value, 5) for value in start_q])
    print("goal q:", [round(value, 5) for value in goal_q])

    if args.headless:
        run(env, trajectory, viewer=None)
        return

    from mujoco import viewer as mujoco_viewer  # type: ignore

    with mujoco_viewer.launch_passive(env.model, env.data) as viewer:
        run(env, trajectory, viewer=viewer)


def run(env: FrankaPandaEnv, trajectory: JointTrajectory, viewer) -> None:
    start_time = env.time
    next_wall_time = time.time()

    while True:
        elapsed = env.time - start_time
        if elapsed > trajectory.duration_s:
            break
        if viewer is not None and not viewer.is_running():
            break

        point = trajectory.sample(elapsed)
        env.arm.command_joint_positions(point.q)
        env.step()

        if viewer is not None:
            viewer.sync()

        next_wall_time += env.timestep
        sleep_s = next_wall_time - time.time()
        if sleep_s > 0.0:
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()
