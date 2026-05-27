from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RobotState:
    """JSON-friendly snapshot of the 7-DOF arm state."""

    time_s: float
    q: tuple[float, ...]
    qd: tuple[float, ...]
    ctrl: tuple[float, ...]
    ee_position: tuple[float, float, float]
    ee_rotation: tuple[tuple[float, float, float], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "q": list(self.q),
            "qd": list(self.qd),
            "ctrl": list(self.ctrl),
            "ee_position": list(self.ee_position),
            "ee_rotation": [list(row) for row in self.ee_rotation],
        }
