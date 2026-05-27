from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def resolve_project_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


@dataclass(frozen=True)
class RobotConfig:
    name: str
    model_path: Path
    home_keyframe: str | None
    home_qpos: tuple[float, ...]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "RobotConfig":
        return cls(
            name=str(data["name"]),
            model_path=resolve_project_path(data["model_path"]),
            home_keyframe=data.get("home_keyframe"),
            home_qpos=tuple(float(value) for value in data.get("home_qpos", ())),
        )


@dataclass(frozen=True)
class SimulationConfig:
    timestep_s: float = 0.002


@dataclass(frozen=True)
class ProjectConfig:
    robot: RobotConfig
    simulation: SimulationConfig

    @classmethod
    def from_yaml(cls, path: str | Path = PROJECT_ROOT / "configs" / "robot.yaml") -> "ProjectConfig":
        data = load_yaml(path)
        simulation = data.get("simulation", {})
        return cls(
            robot=RobotConfig.from_mapping(data["robot"]),
            simulation=SimulationConfig(
                timestep_s=float(simulation.get("timestep_s", 0.002)),
            ),
        )
