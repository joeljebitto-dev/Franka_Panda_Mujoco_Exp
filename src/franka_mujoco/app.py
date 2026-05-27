from __future__ import annotations

import argparse
import asyncio
import math
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from franka_mujoco.config import PROJECT_ROOT
from franka_mujoco.constants import PANDA_DOF
from franka_mujoco.control import JointPIDController
from franka_mujoco.env import FrankaPandaEnv
from franka_mujoco.kinematics import DampedLeastSquaresIK, MuJoCoKinematics
from franka_mujoco.trajectory import JointTrajectory

ControlMode = Literal["manual", "pid", "ik", "trajectory", "ai"]


class ModeRequest(BaseModel):
    mode: ControlMode


class JointTargetRequest(BaseModel):
    q: list[float] = Field(min_length=PANDA_DOF, max_length=PANDA_DOF)
    mode: ControlMode | None = None


class CartesianTargetRequest(BaseModel):
    position: list[float] = Field(min_length=3, max_length=3)
    rotation: list[list[float]] | None = None
    use_orientation: bool = False
    mode_after_solve: ControlMode = "pid"

    @field_validator("rotation")
    @classmethod
    def _validate_rotation(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return value
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("rotation must be a 3x3 matrix")
        return value


class PIDRequest(BaseModel):
    kp: float | list[float] = 1.0
    ki: float | list[float] = 0.0
    kd: float | list[float] = 0.05
    integral_limit: float = 0.5
    correction_limit: float | None = None


class TrajectoryRequest(BaseModel):
    goal_q: list[float] = Field(min_length=PANDA_DOF, max_length=PANDA_DOF)
    duration_s: float = Field(default=3.0, gt=0.0)
    method: Literal["linear", "cubic", "quintic"] = "cubic"


class StepRequest(BaseModel):
    running: bool


@dataclass
class RuntimeEvent:
    time_s: float
    level: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "level": self.level,
            "message": self.message,
        }


class DashboardRuntime:
    """Owns the MuJoCo simulation and exposes thread-safe dashboard commands."""

    def __init__(self):
        self.env = FrankaPandaEnv()
        self.kinematics = MuJoCoKinematics(self.env.model, self.env.data, self.env.mujoco, self.env.arm)
        self.ik = DampedLeastSquaresIK(self.kinematics, self.env.arm)
        self.pid = JointPIDController(kp=1.0, ki=0.0, kd=0.05)

        self.lock = threading.RLock()
        self.mode: ControlMode = "manual"
        self.running = True
        self.target_q = self.env.arm.joint_positions()
        self.cartesian_target: dict[str, Any] | None = None
        self.trajectory: JointTrajectory | None = None
        self.trajectory_start_time = 0.0
        self.events: deque[RuntimeEvent] = deque(maxlen=80)
        self.history: deque[dict[str, Any]] = deque(maxlen=300)
        self._last_history_time = -1.0
        self._last_wall_tick = time.perf_counter()
        self._sim_steps_since_tick = 0
        self._sim_fps = 0.0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._add_event("info", "Dashboard runtime initialized")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="franka-dashboard-runtime", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def set_mode(self, mode: ControlMode) -> dict[str, Any]:
        with self.lock:
            self.mode = mode
            if mode != "trajectory":
                self.trajectory = None
            self._add_event("info", f"Mode set to {mode}")
            return self.telemetry_locked()

    def set_running(self, running: bool) -> dict[str, Any]:
        with self.lock:
            self.running = running
            self._add_event("info", "Simulation running" if running else "Simulation paused")
            return self.telemetry_locked()

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.env.reset()
            self.pid.reset()
            self.target_q = self.env.arm.joint_positions()
            self.trajectory = None
            self.mode = "manual"
            self.cartesian_target = None
            self.history.clear()
            self._add_event("info", "Robot reset to home")
            return self.telemetry_locked()

    def set_joint_target(self, q: list[float], mode: ControlMode | None = None) -> dict[str, Any]:
        with self.lock:
            self.target_q = self.env.arm.clip_to_actuator_ranges(q)
            if mode is not None:
                self.mode = mode
            if self.mode == "trajectory":
                self.trajectory = None
                self.mode = "pid"
            self._add_event("info", "Joint target updated")
            return self.telemetry_locked()

    def set_pid(self, request: PIDRequest) -> dict[str, Any]:
        with self.lock:
            self.pid = JointPIDController(
                kp=request.kp,
                ki=request.ki,
                kd=request.kd,
                integral_limit=request.integral_limit,
                correction_limit=request.correction_limit,
            )
            self._add_event("info", "PID gains updated")
            return self.telemetry_locked()

    def solve_cartesian_target(self, request: CartesianTargetRequest) -> dict[str, Any]:
        with self.lock:
            rotation = request.rotation if request.use_orientation else None
            result = self.ik.solve_pose(
                target_position=request.position,
                target_rotation=rotation,
                initial_q=self.env.arm.joint_positions(),
                orientation_weight=0.4,
            )
            self.cartesian_target = {
                "position": request.position,
                "rotation": request.rotation,
                "use_orientation": request.use_orientation,
                "result": {
                    "success": result.success,
                    "iterations": result.iterations,
                    "position_error_norm": result.position_error_norm,
                    "orientation_error_norm": result.orientation_error_norm,
                    "message": result.message,
                },
            }
            if not result.success:
                self._add_event("warning", f"IK failed: {result.message}")
                raise HTTPException(status_code=422, detail=self.cartesian_target["result"])

            self.target_q = self.env.arm.clip_to_actuator_ranges(result.q)
            self.mode = request.mode_after_solve
            self._add_event("success", f"IK converged in {result.iterations} iterations")
            return self.telemetry_locked()

    def start_trajectory(self, request: TrajectoryRequest) -> dict[str, Any]:
        with self.lock:
            start_q = self.env.arm.joint_positions()
            goal_q = self.env.arm.clip_to_actuator_ranges(request.goal_q)
            self.trajectory = JointTrajectory(
                start_q=start_q,
                goal_q=goal_q,
                duration_s=request.duration_s,
                method=request.method,
            )
            self.trajectory_start_time = self.env.time
            self.mode = "trajectory"
            self.target_q = goal_q
            self._add_event("info", f"{request.method} trajectory started")
            return self.telemetry_locked()

    def telemetry(self) -> dict[str, Any]:
        with self.lock:
            return self.telemetry_locked()

    def telemetry_locked(self) -> dict[str, Any]:
        started = time.perf_counter()
        self.env.mujoco.mj_forward(self.env.model, self.env.data)
        state = self.env.arm.state(self.env.time).as_dict()
        pose = self.kinematics.forward()
        jacobian = self.kinematics.jacobian()
        transform = _homogeneous_transform(pose.position, pose.rotation)
        euler = _rotation_to_euler_xyz(np.array(pose.rotation, dtype=float))
        quaternion = _rotation_to_quaternion(np.array(pose.rotation, dtype=float))
        q_error = [target - actual for target, actual in zip(self.target_q, state["q"])]
        backend_latency_ms = (time.perf_counter() - started) * 1000.0

        return {
            "time_s": state["time_s"],
            "timestep_s": self.env.timestep,
            "mode": self.mode,
            "running": self.running,
            "q": state["q"],
            "qd": state["qd"],
            "ctrl": state["ctrl"],
            "target_q": list(self.target_q),
            "q_error": q_error,
            "ee_position": state["ee_position"],
            "ee_rotation": state["ee_rotation"],
            "ee_euler_xyz": euler,
            "ee_quaternion_wxyz": quaternion,
            "transform": transform,
            "jacobian": jacobian,
            "jacobian_condition": self._safe_condition_number(jacobian),
            "pid": {
                "kp": self.pid.kp,
                "ki": self.pid.ki,
                "kd": self.pid.kd,
                "integral": self.pid.integral,
            },
            "joint_limits": self.env.arm.joint_limits(),
            "actuator_limits": self.env.arm.actuator_ctrl_ranges(),
            "cartesian_target": self.cartesian_target,
            "trajectory": self._trajectory_status_locked(),
            "events": [event.as_dict() for event in reversed(self.events)],
            "history": list(self.history),
            "metrics": {
                "sim_fps": self._sim_fps,
                "backend_latency_ms": backend_latency_ms,
                "history_points": len(self.history),
                "mujoco_status": "ok",
            },
        }

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            step_start = time.perf_counter()
            with self.lock:
                if self.running:
                    self._apply_control_locked()
                    self.env.step()
                    self._record_history_locked()
                    self._sim_steps_since_tick += 1
                    self._update_sim_fps_locked()

            elapsed = time.perf_counter() - step_start
            sleep_s = max(0.0, self.env.timestep - elapsed)
            time.sleep(sleep_s)

    def _apply_control_locked(self) -> None:
        if self.mode == "trajectory" and self.trajectory is not None:
            elapsed = self.env.time - self.trajectory_start_time
            point = self.trajectory.sample(elapsed)
            self.env.arm.command_joint_positions(point.q)
            if elapsed >= self.trajectory.duration_s:
                self.target_q = list(self.trajectory.goal_q)
                self.trajectory = None
                self.mode = "pid"
                self._add_event("success", "Trajectory complete")
            return

        if self.mode == "pid" or self.mode == "ik":
            result = self.pid.compute_position_targets(
                desired_q=self.target_q,
                actual_q=self.env.arm.joint_positions(),
                actual_qd=self.env.arm.joint_velocities(),
                dt=self.env.timestep,
            )
            self.env.arm.command_joint_positions(result.position_targets)
            return

        if self.mode == "manual":
            self.env.arm.command_joint_positions(self.target_q)

    def _record_history_locked(self) -> None:
        if self.env.time - self._last_history_time < 0.05:
            return
        self._last_history_time = self.env.time
        pose = self.kinematics.forward()
        q = self.env.arm.joint_positions()
        q_error = [target - actual for target, actual in zip(self.target_q, q)]
        cartesian_error = 0.0
        if self.cartesian_target:
            target = np.array(self.cartesian_target["position"], dtype=float)
            cartesian_error = float(np.linalg.norm(target - np.array(pose.position, dtype=float)))
        self.history.append(
            {
                "time_s": self.env.time,
                "q": q,
                "q_error": q_error,
                "cartesian_error": cartesian_error,
            }
        )

    def _update_sim_fps_locked(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_wall_tick
        if elapsed < 1.0:
            return
        self._sim_fps = self._sim_steps_since_tick / elapsed
        self._sim_steps_since_tick = 0
        self._last_wall_tick = now

    def _trajectory_status_locked(self) -> dict[str, Any] | None:
        if self.trajectory is None:
            return None
        elapsed = max(0.0, self.env.time - self.trajectory_start_time)
        return {
            "elapsed_s": elapsed,
            "duration_s": self.trajectory.duration_s,
            "method": self.trajectory.method,
            "progress": min(1.0, elapsed / self.trajectory.duration_s),
        }

    def _safe_condition_number(self, jacobian: list[list[float]]) -> float:
        try:
            return float(np.linalg.cond(np.array(jacobian, dtype=float)))
        except np.linalg.LinAlgError:
            return float("inf")

    def _add_event(self, level: str, message: str) -> None:
        self.events.append(RuntimeEvent(time_s=self.env.time, level=level, message=message))


def create_app() -> FastAPI:
    runtime = DashboardRuntime()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            runtime.stop()

    app = FastAPI(title="Franka Panda MuJoCo Dashboard", version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "robot": "franka_panda", "dof": PANDA_DOF}

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return runtime.telemetry()

    @app.post("/api/control/mode")
    def set_mode(request: ModeRequest) -> dict[str, Any]:
        return runtime.set_mode(request.mode)

    @app.post("/api/control/running")
    def set_running(request: StepRequest) -> dict[str, Any]:
        return runtime.set_running(request.running)

    @app.post("/api/reset")
    def reset() -> dict[str, Any]:
        return runtime.reset()

    @app.post("/api/target/joint")
    def set_joint_target(request: JointTargetRequest) -> dict[str, Any]:
        return runtime.set_joint_target(request.q, request.mode)

    @app.post("/api/target/cartesian")
    def set_cartesian_target(request: CartesianTargetRequest) -> dict[str, Any]:
        return runtime.solve_cartesian_target(request)

    @app.post("/api/pid")
    def set_pid(request: PIDRequest) -> dict[str, Any]:
        return runtime.set_pid(request)

    @app.post("/api/trajectory/start")
    def start_trajectory(request: TrajectoryRequest) -> dict[str, Any]:
        return runtime.start_trajectory(request)

    @app.websocket("/ws/telemetry")
    async def websocket_telemetry(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(runtime.telemetry())
                await asyncio.sleep(1.0 / 30.0)
        except WebSocketDisconnect:
            return

    frontend_dist = PROJECT_ROOT / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    return app


def _homogeneous_transform(
    position: tuple[float, float, float],
    rotation: tuple[tuple[float, float, float], ...],
) -> list[list[float]]:
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], position[0]],
        [rotation[1][0], rotation[1][1], rotation[1][2], position[1]],
        [rotation[2][0], rotation[2][1], rotation[2][2], position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rotation_to_euler_xyz(rotation: np.ndarray) -> list[float]:
    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return [roll, pitch, yaw]


def _rotation_to_quaternion(rotation: np.ndarray) -> list[float]:
    trace_value = float(np.trace(rotation))
    if trace_value > 0.0:
        s = math.sqrt(trace_value + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
    return [float(w), float(x), float(y), float(z)]


app = create_app()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the FastAPI dashboard backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    import uvicorn

    if args.reload:
        uvicorn.run("franka_mujoco.app:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
