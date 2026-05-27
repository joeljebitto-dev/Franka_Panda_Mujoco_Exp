from __future__ import annotations

from pathlib import Path

from franka_mujoco.constants import DEFAULT_END_EFFECTOR_BODY
from franka_mujoco.arm import FrankaPandaArm
from franka_mujoco.config import ProjectConfig
from franka_mujoco.state import RobotState


class FrankaPandaEnv:
    """Minimal MuJoCo environment that loads and resets the Panda scene."""

    def __init__(self, config: ProjectConfig | None = None, end_effector_body: str = DEFAULT_END_EFFECTOR_BODY):
        self.config = config or ProjectConfig.from_yaml()
        self.mujoco = self._require_mujoco()

        model_path = Path(self.config.robot.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MuJoCo model not found: {model_path}\n"
                "Expected: assets/mujoco_menagerie/franka_emika_panda/scene.xml"
            )

        self.model = self.mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = self.config.simulation.timestep_s
        self.data = self.mujoco.MjData(self.model)
        self.arm = FrankaPandaArm(self.model, self.data, self.mujoco, end_effector_body=end_effector_body)
        self.reset()

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def reset(self) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self._apply_home_pose()
        self.arm.hold_current_arm_pose()
        self.mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        self.mujoco.mj_step(self.model, self.data)

    def state_snapshot(self) -> RobotState:
        self.mujoco.mj_forward(self.model, self.data)
        return self.arm.state(self.time)

    def _apply_home_pose(self) -> None:
        key_name = self.config.robot.home_keyframe
        if key_name:
            key_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_KEY, key_name)
            if key_id >= 0:
                self.data.qpos[:] = self.model.key_qpos[key_id]
                self.data.ctrl[:] = self.model.key_ctrl[key_id]
                return

        home_qpos = self.config.robot.home_qpos
        if home_qpos:
            if len(home_qpos) != self.model.nq:
                raise ValueError(f"home_qpos has {len(home_qpos)} values, model.nq is {self.model.nq}")
            self.data.qpos[:] = home_qpos

    @staticmethod
    def _require_mujoco():
        try:
            import mujoco  # type: ignore
        except ImportError as exc:
            raise ImportError("Install MuJoCo first: pip install -e .") from exc
        return mujoco
