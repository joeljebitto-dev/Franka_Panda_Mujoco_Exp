from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from franka_mujoco.constants import PANDA_DOF


@dataclass(frozen=True)
class IKResult:
    success: bool
    q: list[float]
    iterations: int
    position_error_norm: float
    orientation_error_norm: float
    weighted_error_norm: float
    message: str


class DampedLeastSquaresIK:
    """7-DOF damped least-squares inverse kinematics solver.

    The solver supports both position-only IK and full pose IK. It uses the
    MuJoCo-backed FK/Jacobian helper as the kinematic model, which makes it a
    good validation target while you build your own analytical FK/Jacobian.
    """

    def __init__(self, kinematics, arm):
        self.kinematics = kinematics
        self.arm = arm

    def solve_position(
        self,
        target_position: Sequence[float],
        initial_q: Sequence[float] | None = None,
        tolerance: float = 1e-4,
        max_iterations: int = 100,
        damping: float = 1e-2,
        step_size: float = 0.5,
        max_delta_q: float = 0.25,
    ) -> IKResult:
        return self.solve_pose(
            target_position=target_position,
            target_rotation=None,
            initial_q=initial_q,
            position_tolerance=tolerance,
            orientation_tolerance=float("inf"),
            max_iterations=max_iterations,
            damping=damping,
            step_size=step_size,
            orientation_weight=0.0,
            max_delta_q=max_delta_q,
        )

    def solve_pose(
        self,
        target_position: Sequence[float],
        target_rotation: Sequence[Sequence[float]] | None,
        initial_q: Sequence[float] | None = None,
        position_tolerance: float = 1e-4,
        orientation_tolerance: float = 1e-3,
        max_iterations: int = 200,
        damping: float = 2e-2,
        step_size: float = 0.5,
        orientation_weight: float = 0.5,
        max_delta_q: float = 0.25,
    ) -> IKResult:
        """Solve for joint angles that reach a target end-effector pose.

        Args:
            target_position: Desired world-frame end-effector position.
            target_rotation: Desired 3x3 world-frame rotation matrix. Pass None
                for position-only IK.
            initial_q: Optional 7-joint seed. Defaults to the current arm state.
            position_tolerance: Position convergence threshold in meters.
            orientation_tolerance: Orientation convergence threshold in radians.
            max_iterations: Maximum Newton-style update iterations.
            damping: DLS damping coefficient. Increase near singularities.
            step_size: Scales each solved joint update.
            orientation_weight: Scales orientation error relative to position.
            max_delta_q: Per-iteration joint update clamp in radians.
        """

        target_pos = _as_position(target_position)
        target_rot = _as_rotation(target_rotation) if target_rotation is not None else None
        if damping < 0.0:
            raise ValueError("damping must be non-negative")
        if step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if max_delta_q <= 0.0:
            raise ValueError("max_delta_q must be positive")
        if target_rot is not None and orientation_weight <= 0.0:
            raise ValueError("orientation_weight must be positive for full pose IK")
        q = self._initial_q(initial_q)

        last_position_norm = float("inf")
        last_orientation_norm = 0.0
        last_weighted_norm = float("inf")

        for iteration in range(max_iterations + 1):
            pose = self.kinematics.forward(q.tolist())
            current_pos = np.array(pose.position, dtype=float)
            position_error = target_pos - current_pos
            position_norm = float(np.linalg.norm(position_error))

            if target_rot is None:
                orientation_error = np.zeros(3, dtype=float)
                orientation_norm = 0.0
            else:
                current_rot = np.array(pose.rotation, dtype=float)
                orientation_error = rotation_error(target_rot, current_rot)
                orientation_norm = float(np.linalg.norm(orientation_error))

            task_error = self._weighted_error(position_error, orientation_error, orientation_weight, target_rot)
            weighted_norm = float(np.linalg.norm(task_error))

            last_position_norm = position_norm
            last_orientation_norm = orientation_norm
            last_weighted_norm = weighted_norm

            if position_norm <= position_tolerance and orientation_norm <= orientation_tolerance:
                return IKResult(
                    success=True,
                    q=q.tolist(),
                    iterations=iteration,
                    position_error_norm=position_norm,
                    orientation_error_norm=orientation_norm,
                    weighted_error_norm=weighted_norm,
                    message="converged",
                )

            jacobian = np.array(self.kinematics.jacobian(q.tolist()), dtype=float)
            task_jacobian = self._weighted_jacobian(jacobian, orientation_weight, target_rot)
            delta_q = damped_least_squares(task_jacobian, task_error, damping)
            delta_q = np.clip(delta_q, -max_delta_q, max_delta_q)
            q = q + step_size * delta_q
            q = np.array(self.arm.clip_to_joint_limits(q.tolist()), dtype=float)

        return IKResult(
            success=False,
            q=q.tolist(),
            iterations=max_iterations,
            position_error_norm=last_position_norm,
            orientation_error_norm=last_orientation_norm,
            weighted_error_norm=last_weighted_norm,
            message="maximum iterations reached",
        )

    def _initial_q(self, initial_q: Sequence[float] | None) -> np.ndarray:
        if initial_q is None:
            return np.array(self.arm.joint_positions(), dtype=float)
        if len(initial_q) != PANDA_DOF:
            raise ValueError(f"initial_q must contain {PANDA_DOF} values, got {len(initial_q)}")
        return np.array(initial_q, dtype=float)

    def _weighted_error(
        self,
        position_error: np.ndarray,
        orientation_error: np.ndarray,
        orientation_weight: float,
        target_rotation: np.ndarray | None,
    ) -> np.ndarray:
        if target_rotation is None:
            return position_error
        return np.concatenate([position_error, orientation_weight * orientation_error])

    def _weighted_jacobian(
        self,
        jacobian: np.ndarray,
        orientation_weight: float,
        target_rotation: np.ndarray | None,
    ) -> np.ndarray:
        if target_rotation is None:
            return jacobian[:3, :]
        weighted = np.array(jacobian, copy=True)
        weighted[3:, :] *= orientation_weight
        return weighted


def damped_least_squares(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    """Return dq = J^T (J J^T + lambda^2 I)^-1 error."""

    rows = jacobian.shape[0]
    lhs = jacobian @ jacobian.T + (damping**2) * np.eye(rows)
    return jacobian.T @ np.linalg.solve(lhs, error)


def rotation_error(target_rotation: np.ndarray, current_rotation: np.ndarray) -> np.ndarray:
    """World-frame rotation-vector error from current orientation to target."""

    error_rotation = target_rotation @ current_rotation.T
    return rotation_matrix_log(error_rotation)


def rotation_matrix_log(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix into an axis-angle rotation vector."""

    trace_value = float(np.trace(rotation))
    cos_angle = np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    skew_values = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    )

    if angle < 1e-9:
        return 0.5 * skew_values

    if np.pi - angle < 1e-4:
        axis = np.sqrt(np.maximum(np.diag(rotation) + 1.0, 0.0) * 0.5)
        axis[0] = np.copysign(axis[0], skew_values[0])
        axis[1] = np.copysign(axis[1], skew_values[1])
        axis[2] = np.copysign(axis[2], skew_values[2])
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return np.zeros(3, dtype=float)
        return angle * axis / norm

    return angle * skew_values / (2.0 * np.sin(angle))


def _as_position(values: Sequence[float]) -> np.ndarray:
    if len(values) != 3:
        raise ValueError(f"target_position must contain 3 values, got {len(values)}")
    return np.array(values, dtype=float)


def _as_rotation(values: Sequence[Sequence[float]]) -> np.ndarray:
    rotation = np.array(values, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"target_rotation must be a 3x3 matrix, got {rotation.shape}")
    return rotation
