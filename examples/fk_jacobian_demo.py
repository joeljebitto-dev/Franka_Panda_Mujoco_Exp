from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from franka_mujoco.env import FrankaPandaEnv  # noqa: E402
from franka_mujoco.kinematics import MuJoCoKinematics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Print MuJoCo FK and 6x7 Jacobian data for the Panda.")
    parser.add_argument(
        "--q",
        type=float,
        nargs=7,
        default=None,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
        help="Optional 7-DOF joint configuration in radians.",
    )
    args = parser.parse_args()

    env = FrankaPandaEnv()
    kinematics = MuJoCoKinematics(env.model, env.data, env.mujoco, env.arm)
    q = args.q or env.arm.joint_positions()

    pose = kinematics.forward(q)
    jacobian = kinematics.jacobian(q)
    condition = kinematics.condition_number(q)

    print("q:", [round(value, 5) for value in q])
    print("end-effector position [m]:", [round(value, 5) for value in pose.position])
    print("end-effector rotation matrix:")
    for row in pose.rotation:
        print(" ", [round(value, 5) for value in row])
    print("Jacobian shape: 6 x 7")
    for row in jacobian:
        print(" ", [round(value, 5) for value in row])
    print("condition number:", round(condition, 5))


if __name__ == "__main__":
    main()
