from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from franka_mujoco.constants import PANDA_DOF

# Edit this value or pass base_url=... to FrankaDashboardClient.
DEFAULT_DASHBOARD_URL = "http://127.0.0.1:8000"


class DashboardAPIError(RuntimeError):
    """Raised when the dashboard backend returns an error response."""


class FrankaDashboardClient:
    """Small Python client for controlling the running web dashboard robot."""

    def __init__(self, base_url: str = DEFAULT_DASHBOARD_URL, timeout_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def health(self) -> dict[str, Any]:
        return self._get("/api/health")

    def state(self) -> dict[str, Any]:
        return self._get("/api/state")

    def set_mode(self, mode: str = "python") -> dict[str, Any]:
        return self._post("/api/control/mode", {"mode": mode})

    def use_python_mode(self) -> dict[str, Any]:
        return self.set_mode("python")

    def set_running(self, running: bool) -> dict[str, Any]:
        return self._post("/api/control/running", {"running": bool(running)})

    def pause(self) -> dict[str, Any]:
        return self.set_running(False)

    def resume(self) -> dict[str, Any]:
        return self.set_running(True)

    def reset(self) -> dict[str, Any]:
        return self._post("/api/reset")

    def fk(self, q: list[float] | tuple[float, ...] | None = None) -> dict[str, Any]:
        """Return FK, transform, Jacobian, and condition number for q or current state."""

        return self._post("/api/kinematics/fk", {"q": _optional_dof(q)})

    def jacobian(self, q: list[float] | tuple[float, ...] | None = None) -> list[list[float]]:
        return self.fk(q)["jacobian"]

    def current_pose(self) -> dict[str, Any]:
        fk = self.fk()
        return {
            "position": fk["position"],
            "rotation": fk["rotation"],
            "euler_xyz": fk["euler_xyz"],
            "quaternion_wxyz": fk["quaternion_wxyz"],
            "transform": fk["transform"],
        }

    def solve_ik(
        self,
        position: list[float] | tuple[float, float, float],
        rotation: list[list[float]] | tuple[tuple[float, float, float], ...] | None = None,
        *,
        use_orientation: bool | None = None,
        initial_q: list[float] | tuple[float, ...] | None = None,
        orientation_weight: float = 0.4,
    ) -> dict[str, Any]:
        return self._post(
            "/api/kinematics/ik",
            {
                "position": _vector(position, 3, "position"),
                "rotation": rotation,
                "use_orientation": bool(rotation) if use_orientation is None else bool(use_orientation),
                "initial_q": _optional_dof(initial_q),
                "apply": False,
                "orientation_weight": float(orientation_weight),
            },
        )

    def apply_ik(
        self,
        position: list[float] | tuple[float, float, float],
        rotation: list[list[float]] | tuple[tuple[float, float, float], ...] | None = None,
        *,
        use_orientation: bool | None = None,
        initial_q: list[float] | tuple[float, ...] | None = None,
        mode_after_solve: str = "python",
        orientation_weight: float = 0.4,
    ) -> dict[str, Any]:
        return self._post(
            "/api/kinematics/ik",
            {
                "position": _vector(position, 3, "position"),
                "rotation": rotation,
                "use_orientation": bool(rotation) if use_orientation is None else bool(use_orientation),
                "initial_q": _optional_dof(initial_q),
                "apply": True,
                "mode_after_solve": mode_after_solve,
                "orientation_weight": float(orientation_weight),
            },
        )

    def set_cartesian_target(
        self,
        position: list[float] | tuple[float, float, float],
        rotation: list[list[float]] | tuple[tuple[float, float, float], ...] | None = None,
        *,
        use_orientation: bool | None = None,
        mode_after_solve: str = "python",
    ) -> dict[str, Any]:
        return self._post(
            "/api/target/cartesian",
            {
                "position": _vector(position, 3, "position"),
                "rotation": rotation,
                "use_orientation": bool(rotation) if use_orientation is None else bool(use_orientation),
                "mode_after_solve": mode_after_solve,
            },
        )

    def set_joint_target(self, q: list[float] | tuple[float, ...], mode: str = "python") -> dict[str, Any]:
        return self._post("/api/target/joint", {"q": _dof(q), "mode": mode})

    def move_joints(self, q: list[float] | tuple[float, ...], mode: str = "python") -> dict[str, Any]:
        return self.set_joint_target(q, mode=mode)

    def start_trajectory(
        self,
        goal_q: list[float] | tuple[float, ...],
        *,
        duration_s: float = 3.5,
        method: str = "quintic",
    ) -> dict[str, Any]:
        return self._post(
            "/api/trajectory/start",
            {
                "goal_q": _dof(goal_q),
                "duration_s": float(duration_s),
                "method": method,
            },
        )

    def set_pid(
        self,
        *,
        kp: float | list[float] = 1.0,
        ki: float | list[float] = 0.0,
        kd: float | list[float] = 0.05,
        integral_limit: float = 0.5,
        correction_limit: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "integral_limit": float(integral_limit),
        }
        if correction_limit is not None:
            body["correction_limit"] = float(correction_limit)
        return self._post("/api/pid", body)

    def _get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def _post(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("POST", path, body)

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"} if body is not None else {}
        request = Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise DashboardAPIError(f"{method} {path} failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise DashboardAPIError(f"Could not reach dashboard at {self.base_url}: {exc.reason}") from exc

        if not response_body:
            return {}
        return json.loads(response_body)


def _optional_dof(values: list[float] | tuple[float, ...] | None) -> list[float] | None:
    if values is None:
        return None
    return _dof(values)


def _dof(values: list[float] | tuple[float, ...]) -> list[float]:
    return _vector(values, PANDA_DOF, "q")


def _vector(values: list[float] | tuple[float, ...], expected_len: int, name: str) -> list[float]:
    if len(values) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values, got {len(values)}")
    return [float(value) for value in values]


__all__ = [
    "DEFAULT_DASHBOARD_URL",
    "DashboardAPIError",
    "FrankaDashboardClient",
]
