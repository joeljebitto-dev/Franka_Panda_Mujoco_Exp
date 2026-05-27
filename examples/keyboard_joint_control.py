from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from franka_mujoco.env import FrankaPandaEnv  # noqa: E402


class KeyboardJointControl:
    def __init__(self, env: FrankaPandaEnv, step_size: float):
        self.env = env
        self.step_size = step_size
        self.selected_joint = 0
        self.paused = False
        self.home_ctrl = env.data.ctrl.copy()
        self.target_ctrl = env.data.ctrl.copy()

    def on_key(self, keycode: int) -> None:
        if ord("1") <= keycode <= ord("7"):
            self.selected_joint = keycode - ord("1")
            self._print_state()
            return

        if keycode == ord("["):
            self._nudge_selected_joint(-self.step_size)
            return

        if keycode == ord("]"):
            self._nudge_selected_joint(self.step_size)
            return

        if keycode in (ord("R"), ord("r")):
            self.reset()
            return

        if keycode == ord(" "):
            self.paused = not self.paused
            print("paused" if self.paused else "running")

    def reset(self) -> None:
        self.env.reset()
        self.home_ctrl = self.env.data.ctrl.copy()
        self.target_ctrl = self.env.data.ctrl.copy()
        print("reset to home")

    def apply(self) -> None:
        self.env.data.ctrl[:] = self.target_ctrl

    def _nudge_selected_joint(self, delta: float) -> None:
        actuator_id = self.env.arm.arm_actuator_ids[self.selected_joint]
        target = float(self.target_ctrl[actuator_id]) + delta
        self.target_ctrl[actuator_id] = self._clip_to_actuator_range(actuator_id, target)
        self._print_state()

    def _clip_to_actuator_range(self, actuator_id: int, value: float) -> float:
        if not self.env.model.actuator_ctrllimited[actuator_id]:
            return value

        low, high = self.env.model.actuator_ctrlrange[actuator_id]
        return min(max(value, float(low)), float(high))

    def _print_state(self) -> None:
        actuator_id = self.env.arm.arm_actuator_ids[self.selected_joint]
        target = self.target_ctrl[actuator_id]
        print(f"joint {self.selected_joint + 1}: target {target:.3f} rad")


def main() -> None:
    parser = argparse.ArgumentParser(description="Control Panda joint targets from the MuJoCo viewer.")
    parser.add_argument("--step-size", type=float, default=0.05, help="Joint target nudge size in radians.")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration in seconds.")
    args = parser.parse_args()

    env = FrankaPandaEnv()
    controller = KeyboardJointControl(env, step_size=args.step_size)

    print("keys: 1-7 select joint, [ decrease, ] increase, R reset, Space pause")
    controller._print_state()

    from mujoco import viewer as mujoco_viewer  # type: ignore

    with mujoco_viewer.launch_passive(env.model, env.data, key_callback=controller.on_key) as viewer:
        run(env, controller, viewer, duration=args.duration)


def run(env: FrankaPandaEnv, controller: KeyboardJointControl, viewer, duration: float | None) -> None:
    start_sim_time = env.time
    next_wall_time = time.time()

    while viewer.is_running():
        elapsed = env.time - start_sim_time
        if duration is not None and elapsed >= duration:
            break

        controller.apply()

        if not controller.paused:
            env.step()

        viewer.sync()

        next_wall_time += env.timestep
        sleep_s = next_wall_time - time.time()
        if sleep_s > 0.0:
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()
