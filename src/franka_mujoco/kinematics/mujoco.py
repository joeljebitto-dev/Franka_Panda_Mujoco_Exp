from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from franka_mujoco.constants import DEFAULT_END_EFFECTOR_BODY, PANDA_DOF

T = TypeVar("T")


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "position": list(self.position),
            "rotation": [list(row) for row in self.rotation],
        }


class MuJoCoKinematics:
    """Ground-truth FK and Jacobian helpers backed by MuJoCo.

    This is intentionally separate from AnalyticalPandaKinematics. Use this
    module to validate the FK, Jacobian, and IK implementations you build later.
    """

    def __init__(self, model, data, mujoco_module, arm, end_effector_body: str = DEFAULT_END_EFFECTOR_BODY):
        self.model = model
        self.data = data
        self.mujoco = mujoco_module
        self.arm = arm
        self.end_effector_body_id = self.mujoco.mj_name2id(
            model,
            self.mujoco.mjtObj.mjOBJ_BODY,
            end_effector_body,
        )
        if self.end_effector_body_id < 0:
            raise ValueError(f"Model is missing end-effector body: {end_effector_body}")

    def forward(self, q: Sequence[float] | None = None) -> Pose:
        if q is None:
            self.mujoco.mj_forward(self.model, self.data)
            return self._current_pose()
        return self._evaluate_at_q(q, self._current_pose)

    def jacobian(self, q: Sequence[float] | None = None) -> list[list[float]]:
        if q is None:
            self.mujoco.mj_forward(self.model, self.data)
            return self._current_jacobian()
        return self._evaluate_at_q(q, self._current_jacobian)

    def condition_number(self, q: Sequence[float] | None = None) -> float:
        jacobian = np.array(self.jacobian(q), dtype=float)
        return float(np.linalg.cond(jacobian))

    def _current_pose(self) -> Pose:
        position = self.data.xpos[self.end_effector_body_id]
        flat_rotation = self.data.xmat[self.end_effector_body_id]
        return Pose(
            position=(float(position[0]), float(position[1]), float(position[2])),
            rotation=(
                (float(flat_rotation[0]), float(flat_rotation[1]), float(flat_rotation[2])),
                (float(flat_rotation[3]), float(flat_rotation[4]), float(flat_rotation[5])),
                (float(flat_rotation[6]), float(flat_rotation[7]), float(flat_rotation[8])),
            ),
        )

    def _current_jacobian(self) -> list[list[float]]:
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        self.mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.end_effector_body_id)
        full = np.vstack([jacp, jacr])
        arm_columns = full[:, self.arm.arm_qvel_indices]
        if arm_columns.shape != (6, PANDA_DOF):
            raise ValueError(f"Expected a 6x{PANDA_DOF} Jacobian, got {arm_columns.shape}")
        return arm_columns.tolist()

    def _evaluate_at_q(self, q: Sequence[float], evaluator: Callable[[], T]) -> T:
        if len(q) != PANDA_DOF:
            raise ValueError(f"q must contain {PANDA_DOF} values, got {len(q)}")

        old_qpos = np.array(self.data.qpos, copy=True)
        old_qvel = np.array(self.data.qvel, copy=True)
        old_ctrl = np.array(self.data.ctrl, copy=True)
        try:
            self.arm.set_joint_positions(q)
            self.data.qvel[:] = 0.0
            self.mujoco.mj_forward(self.model, self.data)
            return evaluator()
        finally:
            self.data.qpos[:] = old_qpos
            self.data.qvel[:] = old_qvel
            self.data.ctrl[:] = old_ctrl
            self.mujoco.mj_forward(self.model, self.data)
