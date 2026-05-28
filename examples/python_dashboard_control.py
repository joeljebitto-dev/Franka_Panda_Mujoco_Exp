from __future__ import annotations

from franka_mujoco.client import DEFAULT_DASHBOARD_URL, FrankaDashboardClient

ROBOT_WEB_ADDRESS = DEFAULT_DASHBOARD_URL


def main() -> None:
    robot = FrankaDashboardClient(ROBOT_WEB_ADDRESS)

    print(robot.health())
    robot.use_python_mode()
    robot.set_pid(kp=1.2, ki=0.0, kd=0.06)

    state = robot.state()
    q = state["q"]
    print("current fk:", robot.fk(q)["position"])

    target_q = list(q)
    target_q[1] += 0.05
    robot.move_joints(target_q)

    target_position = robot.current_pose()["position"]
    target_position[2] += 0.03
    ik = robot.apply_ik(target_position)
    print("ik:", ik)


if __name__ == "__main__":
    main()
