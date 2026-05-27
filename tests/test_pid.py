from __future__ import annotations

from franka_mujoco.control import JointPIDController


def test_pid_zero_error_holds_current_position() -> None:
    pid = JointPIDController(kp=1.0, ki=0.1, kd=0.05)
    q = [0.0, 0.1, 0.2, -1.0, 0.0, 1.2, -0.5]
    qd = [0.0] * 7

    result = pid.compute_position_targets(q, q, qd, dt=0.002)

    assert result.position_targets == q
    assert result.error == [0.0] * 7


def test_pid_integral_is_clamped() -> None:
    pid = JointPIDController(kp=0.0, ki=1.0, kd=0.0, integral_limit=0.01)
    desired_q = [1.0] * 7
    actual_q = [0.0] * 7
    actual_qd = [0.0] * 7

    for _ in range(10):
        result = pid.compute_position_targets(desired_q, actual_q, actual_qd, dt=0.01)

    assert result.integral == [0.01] * 7
