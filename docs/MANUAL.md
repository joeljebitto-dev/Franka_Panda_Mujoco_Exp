# Franka Panda MuJoCo 7-DOF Starter Manual

This manual explains what is available in the project, how the files fit together, and how to use each example while you build forward kinematics, inverse kinematics, PID control, and trajectory generation.

## 1. What This Project Is

This is a 7-DOF Franka Panda MuJoCo starter. It loads the Panda model from MuJoCo Menagerie, gives you a small Python API around the robot, and provides clean starter modules for the robotics algorithms you plan to implement.

The current code now includes the simulation/control foundation plus a FastAPI and React dashboard. RL training is still a future extension.

## 2. Install And Run

Create an environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Open the MuJoCo viewer:

```bash
python viewer.py
```

Run a headless smoke test:

```bash
python viewer.py --headless --duration 1.0
```

The robot model is configured in [configs/robot.yaml](../configs/robot.yaml). The default model path is:

```text
assets/mujoco_menagerie/franka_emika_panda/scene.xml
```

## 3. Repository Map

```text
configs/
  robot.yaml

src/franka_mujoco/
  __init__.py
  app.py
  arm.py
  config.py
  constants.py
  env.py
  math_utils.py
  state.py
  viewer.py
  control/
    pid.py
  kinematics/
    analytical.py
    ik.py
    mujoco.py
  trajectory/
    joint.py

examples/
  joint_control.py
  keyboard_joint_control.py
  keyboard_pid_control.py
  print_robot_state.py
  fk_jacobian_demo.py
  ik_position_demo.py
  ik_pose_demo.py
  trajectory_demo.py

frontend/
  package.json
  vite.config.js
  src/main.jsx
  src/styles.css

tests/
  test_pid.py
  test_trajectory.py
  test_kinematics_smoke.py

docs/
  MANUAL.md
```

## 4. Core Concepts

The Franka Panda arm has 7 revolute arm joints:

```text
q = [q1, q2, q3, q4, q5, q6, q7]
```

The MuJoCo model also includes two gripper finger joints, so the full model has 9 generalized positions. This starter deliberately separates the 7 arm joints from the gripper joints so FK, IK, PID, and trajectory work stay focused on the 7-DOF arm.

Important state variables:

```text
q      joint positions, rad
qd     joint velocities, rad/s
ctrl   7 arm actuator position targets, rad
pose   end-effector position and rotation
J      6x7 geometric Jacobian
```

The default end-effector body is `hand`, defined in the Menagerie Panda model.

## 5. Available Modules

### Environment

[src/franka_mujoco/env.py](../src/franka_mujoco/env.py) owns the MuJoCo `MjModel`, `MjData`, timestep, reset, and step logic.

Basic usage:

```python
from franka_mujoco.env import FrankaPandaEnv

env = FrankaPandaEnv()
env.step()
print(env.time)
```

Useful members:

```python
env.model
env.data
env.mujoco
env.arm
env.timestep
env.state_snapshot()
```

### Arm Helper

[src/franka_mujoco/arm.py](../src/franka_mujoco/arm.py) is the main 7-DOF robot wrapper.

```python
env = FrankaPandaEnv()

q = env.arm.joint_positions()
qd = env.arm.joint_velocities()
limits = env.arm.joint_limits()
ctrl_limits = env.arm.actuator_ctrl_ranges()
ee_position = env.arm.end_effector_position()
ee_rotation = env.arm.end_effector_rotation_matrix()
```

Command 7 joint-position targets:

```python
target_q = env.arm.joint_positions()
target_q[0] += 0.2
env.arm.command_joint_positions(target_q)
env.step()
```

The helper clips actuator targets before writing them to MuJoCo.

### Robot State Snapshot

[src/franka_mujoco/state.py](../src/franka_mujoco/state.py) provides a JSON-friendly `RobotState`.

```python
state = env.state_snapshot()
print(state.as_dict())
```

Run the included example:

```bash
python examples/print_robot_state.py
```

### MuJoCo FK And Jacobian Ground Truth

[src/franka_mujoco/kinematics/mujoco.py](../src/franka_mujoco/kinematics/mujoco.py) gives simulator-backed FK and Jacobian data.

Use this as ground truth while you implement analytical FK and your own Jacobian.

```python
from franka_mujoco.kinematics import MuJoCoKinematics

env = FrankaPandaEnv()
kin = MuJoCoKinematics(env.model, env.data, env.mujoco, env.arm)

q = env.arm.joint_positions()
pose = kin.forward(q)
J = kin.jacobian(q)
condition = kin.condition_number(q)
```

Run the example:

```bash
python examples/fk_jacobian_demo.py
```

Try a custom 7-joint configuration:

```bash
python examples/fk_jacobian_demo.py --q 0.1 0.2 0.0 -1.4 0.0 1.7 -0.5
```

### Analytical FK Placeholder

[src/franka_mujoco/kinematics/analytical.py](../src/franka_mujoco/kinematics/analytical.py) is where your own math implementation belongs.

Suggested plan:

1. Add Panda DH parameters or product-of-exponentials screw axes.
2. Implement `forward(q)`.
3. Compare position and rotation against `MuJoCoKinematics.forward(q)`.
4. Implement `jacobian(q)`.
5. Compare against `MuJoCoKinematics.jacobian(q)`.

The placeholder raises `NotImplementedError` on purpose. It marks the learning exercise clearly.

### Inverse Kinematics

[src/franka_mujoco/kinematics/ik.py](../src/franka_mujoco/kinematics/ik.py) includes damped least-squares IK for the 7-DOF arm. It supports both position-only IK and full pose IK with orientation.

```python
from franka_mujoco.kinematics import DampedLeastSquaresIK, MuJoCoKinematics

env = FrankaPandaEnv()
kin = MuJoCoKinematics(env.model, env.data, env.mujoco, env.arm)
ik = DampedLeastSquaresIK(kin, env.arm)

pose = kin.forward()
target = [pose.position[0] + 0.03, pose.position[1], pose.position[2]]
result = ik.solve_position(target)
print(result.success, result.q, result.position_error_norm)
```

Run the example:

```bash
python examples/ik_position_demo.py
```

Headless:

```bash
python examples/ik_position_demo.py --headless --duration 1.0
```

Full pose IK:

```python
start_q = env.arm.joint_positions()
target_q = list(start_q)
target_q[0] += 0.12
target_q[1] += 0.05
target_pose = kin.forward(target_q)

result = ik.solve_pose(
    target_position=target_pose.position,
    target_rotation=target_pose.rotation,
    initial_q=start_q,
)
```

Run the full pose example:

```bash
python examples/ik_pose_demo.py
```

Headless:

```bash
python examples/ik_pose_demo.py --headless --duration 1.0
```

Current IK features:

- 3x7 position-only damped least-squares solve.
- 6x7 full pose solve with rotation-vector orientation error.
- Joint-limit clipping after each update.
- Per-step joint update clipping for stability.
- Position, orientation, and weighted error reporting.

Future IK upgrades:

- Add null-space objectives for joint-limit avoidance.
- Add convergence logs for dashboard plots.

### PID Control

[src/franka_mujoco/control/pid.py](../src/franka_mujoco/control/pid.py) has a reusable joint-space PID controller.

```python
from franka_mujoco.control import JointPIDController

pid = JointPIDController(kp=1.0, ki=0.0, kd=0.05)

desired_q = env.arm.joint_positions()
desired_q[1] += 0.1

actual_q = env.arm.joint_positions()
actual_qd = env.arm.joint_velocities()
result = pid.compute_position_targets(desired_q, actual_q, actual_qd, env.timestep)
env.arm.command_joint_positions(result.position_targets)
```

Run the keyboard PID controller:

```bash
python examples/keyboard_pid_control.py
```

Headless short run:

```bash
python examples/keyboard_pid_control.py --headless --duration 1.0
```

Future PID upgrades:

- Store per-joint gains in a YAML config.
- Log error, integral, derivative, and saturation per joint.
- Add step-response metrics such as overshoot and settling time.
- Compare PID tracking against IK plus trajectory tracking.

### Trajectory Generation

[src/franka_mujoco/trajectory/joint.py](../src/franka_mujoco/trajectory/joint.py) provides smooth 7-DOF joint-space trajectories.

```python
from franka_mujoco.trajectory import JointTrajectory

start_q = env.arm.joint_positions()
goal_q = list(start_q)
goal_q[0] += 0.3

traj = JointTrajectory(start_q, goal_q, duration_s=3.0, method="quintic")
point = traj.sample(1.5)
env.arm.command_joint_positions(point.q)
```

Supported time scaling:

```text
linear
cubic
quintic
```

Run the example:

```bash
python examples/trajectory_demo.py
```

Headless short run:

```bash
python examples/trajectory_demo.py --headless --duration 1.0
```

Future trajectory upgrades:

- Add Cartesian trajectories.
- Add velocity and acceleration constraints.
- Add waypoint queues.
- Add trajectory logging and replay.
- Feed trajectory targets into PID rather than directly commanding positions.

## 6. Example Guide

### Viewer

```bash
python viewer.py
```

Use this first to confirm MuJoCo loads the Panda scene.

### Joint Sine Control

```bash
python examples/joint_control.py --joint 4 --amplitude 0.15
```

This is the simplest control example: one actuator receives a sine-wave position target.

### Keyboard Joint Control

```bash
python examples/keyboard_joint_control.py
```

Keys:

```text
1-7    select joint
[      decrease selected joint target
]      increase selected joint target
R      reset home
Space  pause/resume
```

### Keyboard PID Control

```bash
python examples/keyboard_pid_control.py --kp 1.0 --ki 0.0 --kd 0.05
```

This uses the reusable `JointPIDController`.

### State Snapshot

```bash
python examples/print_robot_state.py
```

Use this as the shape for future REST `/state` responses or WebSocket telemetry.

### FK/Jacobian Demo

```bash
python examples/fk_jacobian_demo.py
```

Use this when validating your own FK and Jacobian.

### IK Demo

```bash
python examples/ik_position_demo.py --dx 0.02 --dy 0.00 --dz 0.02
```

This solves a small position target from the current end-effector pose.

### Pose IK Demo

```bash
python examples/ik_pose_demo.py
```

This generates a reachable target pose from a nearby 7-joint configuration, then solves full position plus orientation IK from the home pose.

### Trajectory Demo

```bash
python examples/trajectory_demo.py --method quintic
```

This moves three joints smoothly from home to a nearby target.

## 7. How To Implement Forward Kinematics Later

Start in:

```text
src/franka_mujoco/kinematics/analytical.py
```

Recommended workflow:

1. Write a helper for one homogeneous transform.
2. Encode the 7 Panda link transforms.
3. Multiply them from base to hand.
4. Return `{position, rotation}` in the same style as `Pose`.
5. Run several random or hand-picked `q` values through both implementations.
6. Print position error and rotation error.

Validation idea:

```python
pose_truth = mujoco_kin.forward(q)
pose_mine = analytical_kin.forward(q)
position_error = norm(pose_truth.position - pose_mine.position)
```

When your FK is correct, the position error should be very small for the same end-effector frame.

## 8. How To Implement Jacobian Later

You can implement the geometric Jacobian from frame origins and joint axes:

```text
Jv_i = z_{i-1} x (p_e - p_{i-1})
Jw_i = z_{i-1}
J_i  = [Jv_i; Jw_i]
```

For a 7-DOF manipulator:

```text
J shape = 6 x 7
```

Validation:

```python
J_truth = mujoco_kin.jacobian(q)
J_mine = analytical_kin.jacobian(q)
```

Also compare the Jacobian with finite differences:

```text
small delta q -> small delta end-effector position
```

## 9. Future API And Dashboard Shape

The starter now includes a FastAPI backend and React dashboard.

Run the production build served by FastAPI:

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

For frontend development:

```bash
python -m franka_mujoco.app --host 127.0.0.1 --port 8000
cd frontend
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The dashboard includes:

- 7-joint sliders with live measured joint values.
- Manual, PID, IK, trajectory, and reserved AI modes.
- Cartesian IK target input with optional orientation.
- PID gain editing.
- End-effector pose, rotation matrix, Euler angles, and quaternion.
- Homogeneous transform matrix.
- 6x7 Jacobian heatmap and condition number.
- Live joint/error plots.
- WebSocket telemetry and event log.

Implemented endpoints:

```text
GET  /api/state              current q, qd, ctrl, end-effector pose
POST /api/control/mode       manual, pid, ik, trajectory, ai
POST /api/control/running    pause or resume simulator stepping
POST /api/reset              reset to home
POST /api/target/joint       7 joint targets
POST /api/target/cartesian   position plus optional orientation target
POST /api/pid                PID gains
POST /api/trajectory/start   start a generated trajectory
WS   /ws/telemetry       live state stream
```

Example telemetry payload:

```json
{
  "time_s": 1.234,
  "q": [0.0, 0.1, 0.0, -1.57, 0.0, 1.57, -0.78],
  "qd": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "ctrl": [0.0, 0.1, 0.0, -1.57, 0.0, 1.57, -0.78],
  "ee_position": [0.55, 0.0, 0.62],
  "mode": "pid",
  "jacobian_condition": 12.43
}
```

The dashboard backend lives in [src/franka_mujoco/app.py](../src/franka_mujoco/app.py). The React app lives in [frontend/src/main.jsx](../frontend/src/main.jsx).

## 10. Learning Roadmap

Follow this order:

1. Run `viewer.py` and `print_robot_state.py`.
2. Learn `q`, `qd`, `ctrl`, and joint limits from `arm.py`.
3. Run `fk_jacobian_demo.py`.
4. Implement analytical FK and validate against MuJoCo.
5. Implement the Jacobian and validate against MuJoCo.
6. Experiment with pose IK damping, orientation weight, and joint update limits.
7. Tune PID with small joint targets.
8. Use trajectory targets as desired inputs to PID.
9. Add logging and plots.
10. Extend the dashboard with ML panels after the math/control pieces are trustworthy.

## 11. Troubleshooting

If `mujoco` cannot be imported:

```bash
source .venv/bin/activate
pip install -e .
```

If `scene.xml` is missing, install MuJoCo Menagerie assets:

```bash
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git assets/mujoco_menagerie
git -C assets/mujoco_menagerie sparse-checkout set franka_emika_panda
```

If the viewer does not open over SSH or a remote desktop, use headless checks:

```bash
python viewer.py --headless --duration 1.0
python examples/trajectory_demo.py --headless --duration 1.0
```

If IK fails, try a smaller Cartesian offset first:

```bash
python examples/ik_position_demo.py --dx 0.01 --dy 0.0 --dz 0.01
```

## 12. Tests

Run the test suite:

```bash
pytest
```

The tests currently check:

- PID zero-error and integral-clamping behavior.
- Trajectory endpoint and midpoint behavior.
- MuJoCo FK/Jacobian shape smoke tests.
- Small reachable position-IK and full pose-IK solves.

The MuJoCo tests skip automatically if the Panda assets are not installed.
