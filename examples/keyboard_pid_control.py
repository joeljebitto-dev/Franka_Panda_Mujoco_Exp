from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from franka_mujoco.control import JointPIDController  # noqa: E402
from franka_mujoco.env import FrankaPandaEnv  # noqa: E402
from franka_mujoco.math_utils import clamp  # noqa: E402


class KeyboardPIDControl:
    def __init__(self, env: FrankaPandaEnv, pid: JointPIDController, step_size: float):
        self.env = env
        self.pid = pid
        self.step_size = step_size
        self.selected_joint = 0
        self.paused = False
        self.desired_q = env.arm.joint_positions()

    def on_key(self, keycode: int) -> None:
        if ord("1") <= keycode <= ord("7"):
            self.selected_joint = keycode - ord("1")
            self.print_state()
            return

        if keycode == ord("["):
            self.nudge_selected_joint(-self.step_size)
            return

        if keycode == ord("]"):
            self.nudge_selected_joint(self.step_size)
            return

        if keycode in (ord("R"), ord("r")):
            self.reset()
            return

        if keycode == ord(" "):
            self.paused = not self.paused
            print("paused" if self.paused else "running")

    def reset(self) -> None:
        self.env.reset()
        self.pid.reset()
        self.desired_q = self.env.arm.joint_positions()
        print("reset to home")
        self.print_state()

    def apply_pid(self) -> None:
        actual_q = self.env.arm.joint_positions()
        actual_qvel = self.env.arm.joint_velocities()
        pid_result = self.pid.compute_position_targets(
            self.desired_q, actual_q, actual_qvel, self.env.timestep
        )

        for command, actuator_id in zip(pid_result.position_targets, self.env.arm.arm_actuator_ids):
            self.env.data.ctrl[actuator_id] = self.clip_to_actuator_range(
                actuator_id, command
            )

    def nudge_selected_joint(self, delta: float) -> None:
        self.desired_q[self.selected_joint] += delta
        actuator_id = self.env.arm.arm_actuator_ids[self.selected_joint]
        self.desired_q[self.selected_joint] = self.clip_to_actuator_range(
            actuator_id,
            self.desired_q[self.selected_joint],
        )
        self.print_state()

    def clip_to_actuator_range(self, actuator_id: int, value: float) -> float:
        if not self.env.model.actuator_ctrllimited[actuator_id]:
            return value

        low, high = self.env.model.actuator_ctrlrange[actuator_id]
        return clamp(value, float(low), float(high))

    def print_state(self) -> None:
        target = self.desired_q[self.selected_joint]
        print(f"joint {self.selected_joint + 1}: desired {target:.3f} rad")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keyboard joint control through a simple PID loop."
    )
    parser.add_argument("--kp", type=float, default=1.0)
    parser.add_argument("--ki", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.05)
    parser.add_argument("--integral-limit", type=float, default=0.5)
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.05,
        help="Keyboard target nudge in radians.",
    )
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    env = FrankaPandaEnv()
    pid = JointPIDController(
        kp=args.kp, ki=args.ki, kd=args.kd, integral_limit=args.integral_limit
    )
    controller = KeyboardPIDControl(env, pid, step_size=args.step_size)

    print(
        "keys: 1-7 select joint, [ decrease desired, ] increase desired, R reset, Space pause"
    )
    controller.print_state()

    if args.headless:
        run(env, controller, viewer=None, duration=args.duration)
        return

    from mujoco import viewer as mujoco_viewer  # type: ignore

    with mujoco_viewer.launch_passive(
        env.model, env.data, key_callback=controller.on_key
    ) as viewer:
        run(env, controller, viewer=viewer, duration=args.duration)


def run(
    env: FrankaPandaEnv, controller: KeyboardPIDControl, viewer, duration: float | None
) -> None:
    start_sim_time = env.time
    next_wall_time = time.time()

    while True:
        if duration is not None and env.time - start_sim_time >= duration:
            break
        if viewer is not None and not viewer.is_running():
            break

        controller.apply_pid()

        if not controller.paused:
            env.step()

        if viewer is not None:
            viewer.sync()

        next_wall_time += env.timestep
        sleep_s = next_wall_time - time.time()
        if sleep_s > 0.0:
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()
