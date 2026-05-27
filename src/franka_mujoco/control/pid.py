from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from franka_mujoco.constants import PANDA_DOF
from franka_mujoco.math_utils import clamp

NumberOrVector = float | Sequence[float]


@dataclass(frozen=True)
class PIDResult:
    position_targets: list[float]
    error: list[float]
    integral: list[float]
    correction: list[float]


class JointPIDController:
    """7-DOF joint-space PID helper for position-actuated MuJoCo models.

    The Panda Menagerie model exposes position targets as actuator controls.
    This controller computes corrected position targets from desired joint
    positions, measured positions, and measured velocities.
    """

    def __init__(
        self,
        kp: NumberOrVector,
        ki: NumberOrVector,
        kd: NumberOrVector,
        integral_limit: float = 0.5,
        correction_limit: float | None = None,
        dof: int = PANDA_DOF,
    ):
        self.dof = dof
        self.kp = _expand(kp, dof, "kp")
        self.ki = _expand(ki, dof, "ki")
        self.kd = _expand(kd, dof, "kd")
        self.integral_limit = float(integral_limit)
        self.correction_limit = correction_limit
        self.integral = [0.0] * dof

    def reset(self) -> None:
        self.integral = [0.0] * self.dof

    def compute_position_targets(
        self,
        desired_q: Sequence[float],
        actual_q: Sequence[float],
        actual_qd: Sequence[float],
        dt: float,
        desired_qd: Sequence[float] | None = None,
    ) -> PIDResult:
        _require_length(desired_q, self.dof, "desired_q")
        _require_length(actual_q, self.dof, "actual_q")
        _require_length(actual_qd, self.dof, "actual_qd")
        if desired_qd is None:
            desired_qd = [0.0] * self.dof
        _require_length(desired_qd, self.dof, "desired_qd")

        targets: list[float] = []
        errors: list[float] = []
        corrections: list[float] = []

        for i in range(self.dof):
            error = float(desired_q[i]) - float(actual_q[i])
            velocity_error = float(desired_qd[i]) - float(actual_qd[i])
            self.integral[i] += error * dt
            self.integral[i] = clamp(self.integral[i], -self.integral_limit, self.integral_limit)

            correction = self.kp[i] * error + self.ki[i] * self.integral[i] + self.kd[i] * velocity_error
            if self.correction_limit is not None:
                correction = clamp(correction, -self.correction_limit, self.correction_limit)

            errors.append(error)
            corrections.append(correction)
            targets.append(float(actual_q[i]) + correction)

        return PIDResult(
            position_targets=targets,
            error=errors,
            integral=list(self.integral),
            correction=corrections,
        )


def _expand(value: NumberOrVector, dof: int, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * dof
    _require_length(value, dof, name)
    return [float(item) for item in value]


def _require_length(values: Sequence[float], expected: int, name: str) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} must contain {expected} values, got {len(values)}")
