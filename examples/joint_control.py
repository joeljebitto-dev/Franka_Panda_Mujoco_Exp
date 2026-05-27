from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from franka_mujoco.env import FrankaPandaEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Move one Panda joint with a sine command.")
    parser.add_argument("--joint", type=int, default=1, choices=range(1, 8), help="Joint number, 1 through 7.")
    parser.add_argument("--amplitude", type=float, default=0.25, help="Sine amplitude in radians.")
    parser.add_argument("--frequency", type=float, default=0.2, help="Sine frequency in Hz.")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration in seconds.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the viewer.")
    args = parser.parse_args()

    env = FrankaPandaEnv()
    joint_index = args.joint - 1
    actuator_id = env.arm.arm_actuator_ids[joint_index]

    home_ctrl = env.data.ctrl.copy()
    home_target = float(home_ctrl[actuator_id])

    if args.headless:
        run(env, actuator_id, home_ctrl, home_target, args)
        return

    from mujoco import viewer as mujoco_viewer  # type: ignore

    with mujoco_viewer.launch_passive(env.model, env.data) as viewer:
        run(env, actuator_id, home_ctrl, home_target, args, viewer)


def run(env: FrankaPandaEnv, actuator_id: int, home_ctrl, home_target: float, args, viewer=None) -> None:
    start_sim_time = env.time
    next_wall_time = time.time()

    while True:
        elapsed = env.time - start_sim_time
        if args.duration is not None and elapsed >= args.duration:
            break
        if viewer is not None and not viewer.is_running():
            break

        env.data.ctrl[:] = home_ctrl
        target = home_target + args.amplitude * math.sin(2.0 * math.pi * args.frequency * elapsed)
        env.data.ctrl[actuator_id] = clip_to_actuator_range(env, actuator_id, target)

        env.step()

        if viewer is not None:
            viewer.sync()

        next_wall_time += env.timestep
        sleep_s = next_wall_time - time.time()
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def clip_to_actuator_range(env: FrankaPandaEnv, actuator_id: int, value: float) -> float:
    if not env.model.actuator_ctrllimited[actuator_id]:
        return value

    low, high = env.model.actuator_ctrlrange[actuator_id]
    return min(max(value, float(low)), float(high))


if __name__ == "__main__":
    main()
