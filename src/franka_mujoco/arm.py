from __future__ import annotations

from collections.abc import Sequence

from franka_mujoco.constants import (
    DEFAULT_END_EFFECTOR_BODY,
    PANDA_ACTUATOR_NAMES,
    PANDA_DOF,
    PANDA_FINGER_JOINT_NAMES,
    PANDA_JOINT_NAMES,
)
from franka_mujoco.math_utils import clamp_vector
from franka_mujoco.state import RobotState


class FrankaPandaArm:
    """Helper around the Panda 7-DOF arm joints, actuators, and end-effector."""

    dof = PANDA_DOF
    arm_joint_names = PANDA_JOINT_NAMES
    finger_joint_names = PANDA_FINGER_JOINT_NAMES
    arm_actuator_names = PANDA_ACTUATOR_NAMES

    def __init__(self, model, data, mujoco_module, end_effector_body: str = DEFAULT_END_EFFECTOR_BODY):
        self.model = model
        self.data = data
        self.mujoco = mujoco_module
        self.end_effector_body_name = end_effector_body

        self.arm_joint_ids = self._ids(self.arm_joint_names, self.mujoco.mjtObj.mjOBJ_JOINT)
        self.finger_joint_ids = self._ids(self.finger_joint_names, self.mujoco.mjtObj.mjOBJ_JOINT)
        self.arm_actuator_ids = self._ids(self.arm_actuator_names, self.mujoco.mjtObj.mjOBJ_ACTUATOR)
        self.end_effector_body_id = self._id(end_effector_body, self.mujoco.mjtObj.mjOBJ_BODY)
        self.arm_qpos_indices = tuple(int(self.model.jnt_qposadr[joint_id]) for joint_id in self.arm_joint_ids)
        self.arm_qvel_indices = tuple(int(self.model.jnt_dofadr[joint_id]) for joint_id in self.arm_joint_ids)

    def hold_current_arm_pose(self) -> None:
        """Command the current arm joint positions into the model's position actuators."""

        for qpos_index, actuator_id in zip(self.arm_qpos_indices, self.arm_actuator_ids):
            self.data.ctrl[actuator_id] = self.data.qpos[qpos_index]

    def joint_positions(self) -> list[float]:
        return [float(self.data.qpos[index]) for index in self.arm_qpos_indices]

    def joint_velocities(self) -> list[float]:
        return [float(self.data.qvel[index]) for index in self.arm_qvel_indices]

    def joint_position_targets(self) -> list[float]:
        return [float(self.data.ctrl[actuator_id]) for actuator_id in self.arm_actuator_ids]

    def set_joint_positions(self, q: Sequence[float]) -> None:
        """Write a full 7-joint configuration into qpos.

        This changes simulator state directly. Use it for FK/IK evaluation,
        reset logic, and examples. Use command_joint_positions for normal
        control through MuJoCo actuators.
        """

        self._require_dof(q, "q")
        for qpos_index, value in zip(self.arm_qpos_indices, q):
            self.data.qpos[qpos_index] = float(value)

    def command_joint_positions(self, q: Sequence[float]) -> None:
        """Command 7 joint-position targets through the Panda position actuators."""

        targets = self.clip_to_actuator_ranges(q)
        for actuator_id, target in zip(self.arm_actuator_ids, targets):
            self.data.ctrl[actuator_id] = target

    def joint_limits(self) -> list[tuple[float, float]]:
        limits = []
        for joint_id in self.arm_joint_ids:
            if self.model.jnt_limited[joint_id]:
                low, high = self.model.jnt_range[joint_id]
                limits.append((float(low), float(high)))
            else:
                limits.append((float("-inf"), float("inf")))
        return limits

    def actuator_ctrl_ranges(self) -> list[tuple[float, float]]:
        limits = []
        for actuator_id in self.arm_actuator_ids:
            if self.model.actuator_ctrllimited[actuator_id]:
                low, high = self.model.actuator_ctrlrange[actuator_id]
                limits.append((float(low), float(high)))
            else:
                limits.append((float("-inf"), float("inf")))
        return limits

    def clip_to_joint_limits(self, q: Sequence[float]) -> list[float]:
        return clamp_vector(q, self.joint_limits())

    def clip_to_actuator_ranges(self, q: Sequence[float]) -> list[float]:
        return clamp_vector(q, self.actuator_ctrl_ranges())

    def end_effector_position(self) -> tuple[float, float, float]:
        position = self.data.xpos[self.end_effector_body_id]
        return (float(position[0]), float(position[1]), float(position[2]))

    def end_effector_rotation_matrix(self) -> tuple[tuple[float, float, float], ...]:
        flat = self.data.xmat[self.end_effector_body_id]
        return (
            (float(flat[0]), float(flat[1]), float(flat[2])),
            (float(flat[3]), float(flat[4]), float(flat[5])),
            (float(flat[6]), float(flat[7]), float(flat[8])),
        )

    def state(self, time_s: float) -> RobotState:
        return RobotState(
            time_s=float(time_s),
            q=tuple(self.joint_positions()),
            qd=tuple(self.joint_velocities()),
            ctrl=tuple(self.joint_position_targets()),
            ee_position=self.end_effector_position(),
            ee_rotation=self.end_effector_rotation_matrix(),
        )

    def _ids(self, names: tuple[str, ...], object_type) -> tuple[int, ...]:
        return tuple(self._id(name, object_type) for name in names)

    def _id(self, name: str, object_type) -> int:
        object_id = self.mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Model is missing expected Panda item: {name}")
        return int(object_id)

    def _require_dof(self, values: Sequence[float], name: str) -> None:
        if len(values) != self.dof:
            raise ValueError(f"{name} must contain {self.dof} values, got {len(values)}")
