from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from franka_mujoco.constants import PANDA_DOF
from franka_mujoco.math_utils import clamp


@dataclass(frozen=True)
class JointTrajectoryPoint:
    time_s: float
    q: list[float]
    qd: list[float]


@dataclass(frozen=True)
class JointTrajectory:
    """Smooth 7-DOF joint trajectory between two configurations."""

    start_q: Sequence[float]
    goal_q: Sequence[float]
    duration_s: float
    method: str = "cubic"

    def __post_init__(self) -> None:
        _require_dof(self.start_q, "start_q")
        _require_dof(self.goal_q, "goal_q")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if self.method not in {"linear", "cubic", "quintic"}:
            raise ValueError("method must be one of: linear, cubic, quintic")

    def sample(self, time_s: float) -> JointTrajectoryPoint:
        elapsed = clamp(float(time_s), 0.0, self.duration_s)
        r = elapsed / self.duration_s
        scale, scale_dot = self._time_scaling(r)

        q = []
        qd = []
        for start, goal in zip(self.start_q, self.goal_q):
            delta = float(goal) - float(start)
            q.append(float(start) + delta * scale)
            qd.append(delta * scale_dot / self.duration_s)

        return JointTrajectoryPoint(time_s=elapsed, q=q, qd=qd)

    def _time_scaling(self, r: float) -> tuple[float, float]:
        if self.method == "linear":
            return r, 1.0 if 0.0 < r < 1.0 else 0.0
        if self.method == "cubic":
            return 3.0 * r**2 - 2.0 * r**3, 6.0 * r - 6.0 * r**2
        return 10.0 * r**3 - 15.0 * r**4 + 6.0 * r**5, 30.0 * r**2 - 60.0 * r**3 + 30.0 * r**4


def _require_dof(values: Sequence[float], name: str) -> None:
    if len(values) != PANDA_DOF:
        raise ValueError(f"{name} must contain {PANDA_DOF} values, got {len(values)}")
