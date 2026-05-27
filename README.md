# Franka Panda MuJoCo 7-DOF Starter

A clean Python starter project for learning and building a 7-DOF Franka Panda simulation in MuJoCo.

The repository is intentionally small, but it now has stable places for the main robotics pieces you want to implement and extend later:

- 7-DOF robot wrapper and state snapshots
- MuJoCo viewer and headless simulation
- Ground-truth MuJoCo FK and Jacobian helpers
- Analytical FK/Jacobian placeholder for your own implementation
- Damped least-squares position and full-pose IK solver
- Joint-space PID controller module
- Joint trajectory generator with linear, cubic, and quintic time scaling
- FastAPI + React realtime dashboard
- Learning examples for state, FK/Jacobian, IK, PID, keyboard control, and trajectories

For the full learning manual, read [docs/MANUAL.md](docs/MANUAL.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The default model path is:

```text
assets/mujoco_menagerie/franka_emika_panda/scene.xml
```

If the assets are missing, fetch only the Panda model:

```bash
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git assets/mujoco_menagerie
git -C assets/mujoco_menagerie sparse-checkout set franka_emika_panda
```

## Quick Checks

Open the viewer:

```bash
python viewer.py
```

Run without opening a GUI:

```bash
python viewer.py --headless --duration 1.0
```

Print a JSON robot-state snapshot:

```bash
python examples/print_robot_state.py
```

Print end-effector FK and the 6x7 Jacobian:

```bash
python examples/fk_jacobian_demo.py
```

Solve full pose inverse kinematics:

```bash
python examples/ik_pose_demo.py
```

Run a smooth joint trajectory:

```bash
python examples/trajectory_demo.py
```

Run the tests:

```bash
pytest
```

## Realtime Web Dashboard

Build the React UI and run the FastAPI backend:

```bash
cd frontend
npm install
npm run build
cd ..
python -m franka_mujoco.app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

For frontend development with Vite:

```bash
python -m franka_mujoco.app --host 127.0.0.1 --port 8000
cd frontend
npm run dev
```

Then open `http://127.0.0.1:5173`. Vite proxies `/api` and `/ws` to the FastAPI backend.

## Example Scripts

```text
examples/joint_control.py          Move one joint with a sine target.
examples/keyboard_joint_control.py Control 7 joint targets from the keyboard.
examples/keyboard_pid_control.py   Control 7 desired joints through PID.
examples/print_robot_state.py      Print q, qd, ctrl, and end-effector pose.
examples/fk_jacobian_demo.py       Inspect MuJoCo FK and Jacobian ground truth.
examples/ik_position_demo.py       Solve a small position-only IK target.
examples/ik_pose_demo.py           Solve full position + orientation pose IK.
examples/trajectory_demo.py        Execute a smooth 7-DOF joint trajectory.
frontend/                         React realtime dashboard.
```

Keyboard controls:

- `1` to `7`: select a joint
- `[`: decrease selected joint target
- `]`: increase selected joint target
- `R`: reset to home
- `Space`: pause or resume

## Project Layout

```text
src/franka_mujoco/
  arm.py                  7-DOF joint, actuator, limit, and state helper
  env.py                  MuJoCo environment wrapper
  config.py               YAML config loading
  constants.py            Panda joint names, actuator names, DOF
  state.py                JSON-friendly robot state dataclass
  control/pid.py          Joint-space PID controller
  kinematics/mujoco.py    MuJoCo FK/Jacobian validation helper
  kinematics/ik.py        Damped least-squares position and full-pose IK
  kinematics/analytical.py Placeholder for your own FK/Jacobian
  trajectory/joint.py     Joint trajectory time scaling
  app.py                  FastAPI API and WebSocket dashboard backend
frontend/
  src/main.jsx            React dashboard application
  src/styles.css          Dashboard layout and visual system
  vite.config.js          Dev proxy to FastAPI
tests/
  test_pid.py             PID behavior checks
  test_trajectory.py      Trajectory endpoint and shape checks
  test_kinematics_smoke.py MuJoCo FK/Jacobian/IK smoke checks
```

## Future Build Path

1. Implement analytical forward kinematics in `kinematics/analytical.py`.
2. Compare your FK result against `MuJoCoKinematics.forward`.
3. Implement your own 6x7 Jacobian and compare it against `MuJoCoKinematics.jacobian`.
4. Improve `DampedLeastSquaresIK` with null-space objectives and better limits.
5. Tune and expand `JointPIDController` for trajectory tracking metrics.
6. Extend the dashboard with logging, ML panels, and richer 3D visualization.

The starter is deliberately shaped for those next steps without forcing a web app or ML stack into the first version.
