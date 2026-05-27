# Franka Panda MuJoCo Robot Arm Specification

**Project:** Franka Panda MuJoCo Simulation and Control Stack  
**Primary robot:** Franka Emika Panda / Franka-compatible 7-DOF manipulator  
**Simulation engine:** MuJoCo  
**Initial model source:** `mujoco_menagerie/franka_emika_panda/scene.xml`  
**Language:** Python  
**Target use cases:** robot-arm simulation, joint control, forward kinematics, inverse kinematics, PID control, pick-and-place, future perception/RL integration

> Current starter note: this file is a long-term design reference. For the implemented project layout, available modules, examples, and learning path, use `README.md` and `docs/MANUAL.md` as the source of truth.

---

## 1. Purpose

This specification defines the first production-minded architecture for a Franka Panda robot-arm project using MuJoCo.

The system shall support:

1. Loading and simulating the Franka Panda robot arm.
2. Reading robot state from MuJoCo.
3. Sending joint and gripper commands.
4. Running joint-space PID control.
5. Computing forward kinematics.
6. Solving inverse kinematics.
7. Executing joint and Cartesian trajectories.
8. Running staged manipulation tasks such as reaching and pick-and-place.
9. Logging state, commands, errors, and metrics for debugging and later learning-based extensions.

The goal is to create a clean simulation foundation before adding advanced features such as reinforcement learning, camera perception, force control, or sim-to-real transfer.

---

## 2. Scope

### 2.1 In Scope

- Franka Panda simulation in MuJoCo.
- Python project structure.
- Model loading.
- Joint state reading.
- Actuator command interface.
- Joint-space PID controller.
- Forward kinematics interface.
- Jacobian-based inverse kinematics.
- Trajectory generation.
- Gripper command abstraction.
- Task state machine.
- Viewer loop.
- Logging and metrics.
- Unit and integration test plan.

### 2.2 Out of Scope for Initial Version

- Real Franka hardware control.
- ROS 2 integration.
- Reinforcement learning training.
- Visual servoing.
- Collision-aware global motion planning.
- Tactile sensing.
- Force/torque sensor modeling beyond MuJoCo-provided contact data.
- Multi-arm coordination.

These may be added after the base simulation/control stack is validated.

---

## 3. System Goals

### 3.1 Functional Goals

| ID | Requirement | Priority |
|---|---|---|
| F-001 | Load Franka Panda model from MuJoCo Menagerie | Must |
| F-002 | Initialize robot to stable neutral pose | Must |
| F-003 | Read joint positions and velocities | Must |
| F-004 | Send joint position commands | Must |
| F-005 | Send gripper open/close commands | Must |
| F-006 | Implement joint-space PID controller | Must |
| F-007 | Implement forward kinematics query for end-effector pose | Must |
| F-008 | Implement Jacobian-based inverse kinematics | Must |
| F-009 | Execute joint-space trajectories | Must |
| F-010 | Execute Cartesian target-reaching task | Should |
| F-011 | Execute pick-and-place state machine | Should |
| F-012 | Log simulation state and controller metrics | Should |
| F-013 | Support headless simulation mode | Should |
| F-014 | Support viewer/debug mode | Must |

### 3.2 Non-Functional Goals

| Category | Requirement |
|---|---|
| Performance | Simulation should run at or near real time in viewer mode on a typical development machine. |
| Determinism | Given the same initial conditions and timestep, scripted tasks should produce repeatable behavior. |
| Maintainability | Controllers, robot wrapper, task logic, and simulation loop should be separated. |
| Testability | FK, IK, PID, trajectory generation, and task logic should be testable without the viewer. |
| Observability | Joint errors, Cartesian errors, control commands, and task states should be loggable. |
| Safety | Joint commands should be clipped to valid ranges before being applied. |
| Extensibility | The architecture should allow adding ROS, RL, perception, and real hardware adapters later. |

---

## 4. Recommended Repository Structure

```text
franka_panda_mujoco/
├── assets/
│   └── mujoco_menagerie/
│       └── franka_emika_panda/
├── configs/
│   ├── robot.yaml
│   ├── controller.yaml
│   └── task.yaml
├── src/
│   └── franka_mujoco/
│       ├── __init__.py
│       ├── app.py
│       ├── config.py
│       ├── simulation.py
│       ├── robot.py
│       ├── kinematics.py
│       ├── controllers/
│       │   ├── __init__.py
│       │   ├── pid.py
│       │   ├── joint_position.py
│       │   ├── ik_controller.py
│       │   └── trajectory.py
│       ├── tasks/
│       │   ├── __init__.py
│       │   ├── reach.py
│       │   └── pick_place.py
│       ├── logging_utils.py
│       └── safety.py
├── tests/
│   ├── test_pid.py
│   ├── test_kinematics.py
│   ├── test_ik.py
│   ├── test_trajectory.py
│   └── test_simulation_smoke.py
├── experiments/
│   ├── joint_sine_test.py
│   ├── reach_target.py
│   └── pick_place_cube.py
├── requirements.txt
└── README.md
```

---

## 5. High-Level Architecture

```text
┌──────────────────────────────┐
│        Experiment Script      │
│ joint_sine / reach / pick     │
└───────────────┬──────────────┘
                │
                v
┌──────────────────────────────┐
│        Simulation App         │
│ viewer loop / headless loop   │
└───────────────┬──────────────┘
                │
                v
┌──────────────────────────────┐
│        MuJoCo Backend         │
│ MjModel + MjData              │
└───────┬───────────────┬──────┘
        │               │
        v               v
┌──────────────┐   ┌────────────────┐
│ Robot Wrapper│   │ Task State     │
│ q, qdot, ee  │   │ Machine        │
└──────┬───────┘   └──────┬─────────┘
       │                  │
       v                  v
┌──────────────────────────────┐
│        Controllers            │
│ PID / IK / trajectory / grip  │
└───────────────┬──────────────┘
                │
                v
┌──────────────────────────────┐
│        Safety Layer           │
│ limits / clipping / checks    │
└──────────────────────────────┘
```

---

## 6. Core Modules

### 6.1 `Simulation`

#### Responsibility

Owns the MuJoCo model, MuJoCo data, timestep loop, viewer mode, and stepping behavior.

#### Interface

```python
class Simulation:
    def __init__(self, model_path: str, realtime: bool = True, viewer: bool = True):
        ...

    def reset(self) -> None:
        ...

    def step(self) -> None:
        ...

    def run(self, callback: Callable[[float], None]) -> None:
        ...

    @property
    def time(self) -> float:
        ...
```

#### Acceptance Criteria

- Can load `scene.xml`.
- Can step simulation without a viewer.
- Can run with passive viewer.
- Can reset robot state.
- Can expose `model`, `data`, and simulation time to other modules.

---

### 6.2 `RobotArm`

#### Responsibility

Provides a robot-centric abstraction over MuJoCo’s raw `model` and `data`.

It should hide MuJoCo indexing details from controllers and tasks.

#### Interface

```python
class RobotArm:
    ARM_DOF = 7

    def __init__(self, model, data):
        ...

    def get_qpos(self) -> np.ndarray:
        ...

    def get_qvel(self) -> np.ndarray:
        ...

    def set_qpos(self, q: np.ndarray) -> None:
        ...

    def command_arm(self, q_desired: np.ndarray) -> None:
        ...

    def command_gripper_width(self, width: float) -> None:
        ...

    def get_end_effector_position(self) -> np.ndarray:
        ...

    def get_end_effector_quaternion(self) -> np.ndarray:
        ...

    def get_end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        ...

    def get_site_jacobian(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        ...
```

#### State Variables

| Symbol | Meaning | Shape |
|---|---|---|
| `q` | Arm joint positions | `(7,)` |
| `qdot` | Arm joint velocities | `(7,)` |
| `x_ee` | End-effector position | `(3,)` |
| `R_ee` | End-effector rotation matrix | `(3,3)` |
| `quat_ee` | End-effector quaternion | `(4,)` |
| `u` | Actuator command | model-dependent |

#### Implementation Notes

- The first implementation may assume that the first seven position-controlled actuators map to the Panda arm joints.
- Later versions should resolve joint and actuator IDs by name instead of assuming index order.
- All command vectors must be validated for shape, finite values, and joint limits.

---

## 7. Kinematics

### 7.1 Forward Kinematics

#### Definition

Forward kinematics computes the end-effector pose from the robot joint configuration:

```text
q -> T_base_ee
```

where:

- `q` is the 7-DOF joint vector.
- `T_base_ee` is the homogeneous transform from base frame to end-effector frame.

#### Required Output

```python
@dataclass
class Pose:
    position: np.ndarray
    quaternion: np.ndarray
    rotation: np.ndarray
```

#### FK Interface

```python
class Kinematics:
    def __init__(self, model, data, ee_site_name: str):
        ...

    def forward_kinematics(self, q: np.ndarray) -> Pose:
        ...

    def current_pose(self) -> Pose:
        ...
```

#### MuJoCo FK Strategy

1. Set `data.qpos[:7] = q`.
2. Call `mujoco.mj_forward(model, data)`.
3. Read end-effector site pose:
   - `data.site_xpos[site_id]`
   - `data.site_xmat[site_id]`

#### Homogeneous Transform Form

```text
T_base_ee =
[ R  p ]
[ 0  1 ]
```

where:

- `R` is the 3x3 rotation matrix.
- `p` is the 3D end-effector position.

#### FK Acceptance Tests

| Test | Expected Result |
|---|---|
| Neutral pose FK | End-effector pose is finite and stable |
| Same `q` repeated | Same pose returned |
| Small joint perturbation | Small pose change |
| Invalid shape | Raises `ValueError` |
| NaN input | Raises `ValueError` |

---

### 7.2 Jacobian

#### Definition

The geometric Jacobian maps joint velocity to end-effector spatial velocity:

```text
v_ee = J(q) qdot
```

For a 7-DOF Panda:

```text
J_pos(q): 3 x 7
J_rot(q): 3 x 7
J_full(q): 6 x 7
```

#### Interface

```python
class Kinematics:
    def jacobian(self, q: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        ...
```

#### MuJoCo Jacobian Strategy

Use:

```python
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
```

where:

- `jacp` is translational Jacobian.
- `jacr` is rotational Jacobian.

Then select the columns corresponding to the 7 Panda arm DOFs.

#### Jacobian Acceptance Tests

| Test | Expected Result |
|---|---|
| Shape check | `J_pos.shape == (3, 7)` |
| Shape check | `J_rot.shape == (3, 7)` |
| Finite values | No NaN or Inf |
| Numerical check | Finite-difference FK roughly matches `J_pos` |

---

### 7.3 Inverse Kinematics

#### Definition

Inverse kinematics solves for joint angles that place the end-effector at a desired pose.

Initial version shall support position-only IK:

```text
Given x_target, find q such that FK(q).position ≈ x_target
```

Later version shall support position + orientation IK.

#### Recommended Initial Solver

Use damped least-squares Jacobian IK.

#### Position-Only Error

```text
e_pos = x_target - x_current
```

#### Damped Least-Squares Update

```text
dq = Jᵀ (J Jᵀ + λ² I)^-1 e
```

where:

- `J` is the translational Jacobian.
- `λ` is the damping factor.
- `e` is Cartesian position error.

#### IK Interface

```python
@dataclass
class IKResult:
    q_solution: np.ndarray
    success: bool
    iterations: int
    final_error_norm: float
    reason: str


class IKSolver:
    def __init__(
        self,
        kinematics: Kinematics,
        joint_limits: tuple[np.ndarray, np.ndarray],
        damping: float = 0.05,
        max_iterations: int = 100,
        tolerance: float = 1e-3,
        step_size: float = 0.5,
    ):
        ...

    def solve_position(
        self,
        q_seed: np.ndarray,
        x_target: np.ndarray,
    ) -> IKResult:
        ...
```

#### IK Algorithm

```text
Input: q_seed, x_target

q ← q_seed

for i in range(max_iterations):
    pose ← FK(q)
    e ← x_target - pose.position

    if norm(e) < tolerance:
        return success(q)

    J_pos ← translational_jacobian(q)

    dq ← J_pos.T @ inv(J_pos @ J_pos.T + damping² I) @ e

    q ← q + step_size * dq
    q ← clip(q, q_min, q_max)

return failure(q)
```

#### Position + Orientation IK Extension

For full pose IK:

```text
e = [e_pos, e_rot]
J = [J_pos, J_rot]
```

Orientation error can be represented using axis-angle error:

```text
R_err = R_target R_currentᵀ
e_rot = axis_angle(R_err)
```

#### IK Acceptance Tests

| Test | Expected Result |
|---|---|
| Reachable target near current pose | Success |
| Target equal to current pose | Zero or near-zero iterations |
| Unreachable target | Graceful failure |
| Joint limits | Solution remains within limits |
| Singular configuration | Damping prevents numerical explosion |
| Invalid target shape | Raises `ValueError` |

---

## 8. PID Control

### 8.1 Joint-Space PID Controller

#### Purpose

The PID controller converts joint position error into a command. In the first MuJoCo Menagerie Panda setup, the actuators may already be position actuators, so PID may be used at the application layer for reference filtering or for torque-control variants later.

#### Control Law

```text
e(t) = q_desired(t) - q_actual(t)
```

```text
u(t) = Kp e(t) + Ki ∫e(t)dt + Kd de(t)/dt
```

For joint-space control:

```text
u ∈ R⁷
e ∈ R⁷
Kp, Ki, Kd ∈ R⁷
```

#### Interface

```python
@dataclass
class PIDGains:
    kp: np.ndarray
    ki: np.ndarray
    kd: np.ndarray


class JointPIDController:
    def __init__(
        self,
        gains: PIDGains,
        output_limits: tuple[np.ndarray, np.ndarray] | None = None,
        integral_limits: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        ...

    def reset(self) -> None:
        ...

    def compute(
        self,
        q_desired: np.ndarray,
        q_actual: np.ndarray,
        qdot_actual: np.ndarray,
        dt: float,
        qdot_desired: np.ndarray | None = None,
    ) -> np.ndarray:
        ...
```

#### Recommended First Gains

For position-actuator command smoothing:

```yaml
pid:
  kp: [1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.6]
  ki: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  kd: [0.05, 0.05, 0.05, 0.05, 0.03, 0.03, 0.02]
```

For direct torque control, gains must be tuned more carefully and validated incrementally.

#### Anti-Windup Requirements

The PID implementation shall include:

1. Integral state reset.
2. Integral clamping.
3. Output clamping.
4. `dt > 0` validation.
5. NaN/Inf rejection.

#### PID Acceptance Tests

| Test | Expected Result |
|---|---|
| Zero error | Zero or near-zero output |
| Positive error | Positive proportional output |
| Integral accumulates | Integral term increases over time |
| Integral clamp | Integral state does not exceed limits |
| Output clamp | Command does not exceed output limits |
| Reset | Integral and previous-error state cleared |
| Invalid `dt` | Raises `ValueError` |

---

### 8.2 Controller Modes

The system shall support at least three command modes.

#### Mode 1: Direct Position Command

```text
q_desired -> data.ctrl[:7]
```

Use for the first MVP.

#### Mode 2: PID-Filtered Position Command

```text
q_target -> PID/reference filter -> q_command -> data.ctrl[:7]
```

Use when smoothing target transitions.

#### Mode 3: Torque Control

```text
q_target, q_actual, qdot_actual -> PID torque -> data.ctrl[:7]
```

Use only if the MuJoCo actuator model is configured for torque/motor control.

---

## 9. Trajectory Generation

### 9.1 Joint-Space Trajectory

#### Purpose

Generate smooth joint references between two configurations.

#### Minimum Required Trajectories

1. Linear interpolation.
2. Cubic polynomial interpolation.
3. Quintic polynomial interpolation.

#### Interface

```python
class JointTrajectory:
    def __init__(self, q_start: np.ndarray, q_goal: np.ndarray, duration: float):
        ...

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...
```

#### Recommended Initial Trajectory

Use cubic interpolation with zero start and end velocity:

```text
s(t) = 3τ² - 2τ³
τ = clamp(t / T, 0, 1)
q(t) = q_start + s(t) (q_goal - q_start)
```

#### Acceptance Tests

| Test | Expected Result |
|---|---|
| `t = 0` | returns `q_start` |
| `t = duration` | returns `q_goal` |
| Midpoint | finite value between start and goal |
| Out-of-range time | clamps to start or goal |
| Invalid duration | raises `ValueError` |

---

### 9.2 Cartesian Trajectory

#### Purpose

Generate Cartesian end-effector targets.

Initial implementation can use linear interpolation in position space:

```text
x(t) = x_start + s(t)(x_goal - x_start)
```

Each Cartesian target is then converted to a joint target using IK.

#### Interface

```python
class CartesianTrajectory:
    def __init__(self, x_start: np.ndarray, x_goal: np.ndarray, duration: float):
        ...

    def sample(self, t: float) -> np.ndarray:
        ...
```

---

## 10. Gripper Control

### Responsibility

Provide a simple command interface for the Panda gripper.

### Interface

```python
class Gripper:
    def open(self) -> None:
        ...

    def close(self) -> None:
        ...

    def command_width(self, width_m: float) -> None:
        ...
```

### Requirements

| ID | Requirement |
|---|---|
| G-001 | Support fully open command |
| G-002 | Support fully closed command |
| G-003 | Clip width to valid range |
| G-004 | Hide actuator indexing from tasks |
| G-005 | Allow future force-control extension |

---

## 11. Task Layer

### 11.1 Reach Task

#### Purpose

Move the end-effector to a target 3D position.

#### State Machine

```text
INIT
  |
  v
MOVE_TO_TARGET
  |
  v
HOLD
  |
  v
DONE
```

#### Interface

```python
class ReachTask:
    def __init__(self, target_position: np.ndarray, tolerance: float = 0.01):
        ...

    def reset(self) -> None:
        ...

    def step(self, t: float, robot: RobotArm) -> np.ndarray:
        ...

    def is_done(self) -> bool:
        ...
```

#### Completion Criteria

```text
norm(x_target - x_ee) < tolerance
```

for at least `N` consecutive control steps.

---

### 11.2 Pick-and-Place Task

#### Purpose

Pick a cube from one table location and place it at another.

#### State Machine

```text
INIT
  |
  v
MOVE_ABOVE_OBJECT
  |
  v
OPEN_GRIPPER
  |
  v
DESCEND_TO_GRASP
  |
  v
CLOSE_GRIPPER
  |
  v
LIFT_OBJECT
  |
  v
MOVE_ABOVE_GOAL
  |
  v
DESCEND_TO_PLACE
  |
  v
OPEN_GRIPPER
  |
  v
RETREAT
  |
  v
DONE
```

#### Requirements

| ID | Requirement |
|---|---|
| PNP-001 | Task shall use Cartesian waypoints |
| PNP-002 | Waypoints shall be converted to joint targets using IK |
| PNP-003 | Gripper state shall be explicit |
| PNP-004 | Each state shall have timeout protection |
| PNP-005 | Failure reason shall be reported if task fails |

---

## 12. Safety Layer

### Purpose

Prevent invalid or unstable commands from entering MuJoCo.

### Requirements

| ID | Requirement |
|---|---|
| S-001 | Validate command shape |
| S-002 | Reject NaN or Inf commands |
| S-003 | Clip joint commands to joint limits |
| S-004 | Clip gripper commands to valid range |
| S-005 | Limit maximum joint step per control cycle |
| S-006 | Log all clipping events |
| S-007 | Stop task if IK repeatedly fails |

### Interface

```python
class SafetyLimiter:
    def __init__(
        self,
        q_min: np.ndarray,
        q_max: np.ndarray,
        max_delta_q: np.ndarray,
    ):
        ...

    def limit_joint_command(
        self,
        q_command: np.ndarray,
        q_current: np.ndarray,
    ) -> np.ndarray:
        ...
```

---

## 13. Configuration

### `configs/robot.yaml`

```yaml
robot:
  name: franka_panda
  model_path: assets/mujoco_menagerie/franka_emika_panda/scene.xml
  ee_site_name: attachment_site
  arm_dof: 7

  neutral_qpos:
    - 0.0
    - -0.785
    - 0.0
    - -2.356
    - 0.0
    - 1.571
    - 0.785

  joint_limits:
    lower:
      - -2.8973
      - -1.7628
      - -2.8973
      - -3.0718
      - -2.8973
      - -0.0175
      - -2.8973
    upper:
      - 2.8973
      - 1.7628
      - 2.8973
      - -0.0698
      - 2.8973
      - 3.7525
      - 2.8973
```

### `configs/controller.yaml`

```yaml
control:
  mode: position
  control_frequency_hz: 500
  simulation_timestep_s: 0.002

pid:
  kp: [1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.6]
  ki: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  kd: [0.05, 0.05, 0.05, 0.05, 0.03, 0.03, 0.02]

ik:
  damping: 0.05
  max_iterations: 100
  tolerance_m: 0.001
  step_size: 0.5

safety:
  max_delta_q_per_step:
    - 0.02
    - 0.02
    - 0.02
    - 0.02
    - 0.02
    - 0.02
    - 0.02
```

---

## 14. Simulation Loop

### Control Loop Pseudocode

```text
load model
create data
create robot wrapper
create controller
create task
reset simulation

while running:
    t = data.time

    q = robot.get_qpos()
    qdot = robot.get_qvel()
    ee_pose = robot.get_end_effector_pose()

    q_target = task.step(t, robot)

    q_safe = safety.limit_joint_command(q_target, q)

    robot.command_arm(q_safe)

    mujoco.mj_step(model, data)

    logger.record(t, q, qdot, q_target, ee_pose, task.state)

    viewer.sync()
```

### Timing Requirements

| Parameter | Initial Value |
|---|---|
| MuJoCo timestep | `0.002 s` |
| Control frequency | `500 Hz` |
| Viewer target | Real-time or slower |
| Logging frequency | Configurable, default `50 Hz` |

---

## 15. Data Logging

### Required Logged Signals

| Signal | Shape | Purpose |
|---|---:|---|
| `time` | scalar | timeline |
| `q_actual` | `(7,)` | joint state |
| `qdot_actual` | `(7,)` | velocity state |
| `q_command` | `(7,)` | commanded joint target |
| `joint_error` | `(7,)` | tracking error |
| `ee_position` | `(3,)` | FK output |
| `ee_target` | `(3,)` | Cartesian goal |
| `ee_error_norm` | scalar | reach quality |
| `gripper_command` | scalar/vector | manipulation debugging |
| `task_state` | string/int | task progress |
| `ik_success` | bool | IK health |
| `ik_iterations` | int | IK performance |

### Preferred Output Formats

- CSV for quick analysis.
- JSONL for event/state logs.
- NumPy `.npz` for dense arrays.
- Optional: TensorBoard or Weights & Biases later.

---

## 16. Testing Strategy

### 16.1 Unit Tests

#### PID Tests

- Zero error output.
- Proportional response.
- Integral accumulation.
- Derivative response.
- Output saturation.
- Integral anti-windup.
- Reset behavior.

#### FK Tests

- Valid output at neutral pose.
- Repeated input gives identical pose.
- Invalid input shape fails.
- NaN input fails.

#### Jacobian Tests

- Shape correctness.
- Finite values.
- Finite-difference consistency check.

#### IK Tests

- Current pose target succeeds.
- Nearby reachable target succeeds.
- Far unreachable target fails cleanly.
- Joint limits are respected.
- Singular-like configurations do not explode.

#### Trajectory Tests

- Start and end boundary correctness.
- Monotonic interpolation scalar.
- Invalid duration rejection.
- Correct shape outputs.

---

### 16.2 Integration Tests

| Test | Acceptance |
|---|---|
| Load model | Model loads without exception |
| Neutral hold | Arm remains stable for 10 seconds |
| Joint sine | Joint 1 tracks sine command |
| Reach target | End-effector reaches target within tolerance |
| Pick-and-place dry run | State machine completes with mocked gripper/contact |
| Headless mode | Runs without viewer |

---

### 16.3 Metrics

| Metric | Target |
|---|---|
| Joint tracking RMS error | Defined after baseline |
| Cartesian reach error | < 1 cm for simple targets |
| IK success rate | > 95% for reachable sampled targets |
| IK average iterations | < 50 for nearby targets |
| Simulation crash rate | 0 crashes in smoke tests |
| NaN command count | 0 |

---

## 17. Milestone Plan

### Milestone 0: Repository Setup

#### Deliverables

- Project structure.
- `requirements.txt`.
- MuJoCo installs successfully.
- Menagerie model path configured.

#### Acceptance

```text
python experiments/joint_sine_test.py
```

opens the MuJoCo viewer.

---

### Milestone 1: Stable Panda Load

#### Deliverables

- Load Panda.
- Set neutral pose.
- Hold position.
- Realtime viewer loop.

#### Acceptance

- Arm remains stable for 60 seconds.
- No exploding joints.
- No actuator indexing errors.

---

### Milestone 2: Joint Control

#### Deliverables

- `RobotArm.command_arm()`.
- Joint sine test.
- Joint trajectory test.

#### Acceptance

- Single joint sinusoid works.
- Multi-joint interpolation works.
- Commands are clipped to limits.

---

### Milestone 3: FK and Jacobian

#### Deliverables

- FK API.
- End-effector pose extraction.
- Jacobian API.
- Unit tests.

#### Acceptance

- FK returns finite pose.
- Jacobian shape is correct.
- Finite-difference check passes approximately.

---

### Milestone 4: IK

#### Deliverables

- Damped least-squares position IK.
- IK result object.
- Joint-limit handling.
- Reach target experiment.

#### Acceptance

- End-effector reaches nearby Cartesian targets within 1 cm.
- IK fails gracefully for invalid targets.

---

### Milestone 5: PID and Trajectory Control

#### Deliverables

- Joint PID controller.
- Cubic trajectory generator.
- PID test suite.
- Logging of tracking error.

#### Acceptance

- Joint trajectory tracking is stable.
- No integral windup.
- Commands remain finite and bounded.

---

### Milestone 6: Pick-and-Place

#### Deliverables

- Table and cube scene.
- Cartesian waypoint state machine.
- Gripper open/close control.
- Pick-and-place experiment.

#### Acceptance

- Robot completes scripted pick-and-place in simulation.
- Failure states are observable and logged.

---

## 18. Implementation Order

Recommended first implementation sequence:

```text
1. Simulation loader
2. RobotArm wrapper
3. Neutral pose hold
4. Joint sine command
5. Safety limiter
6. Joint trajectory generator
7. FK
8. Jacobian
9. Position-only IK
10. Reach task
11. PID controller
12. Logging
13. Pick-and-place state machine
14. Gripper refinement
15. Tests and CI
```

---

## 19. Minimal MVP Acceptance Checklist

The MVP is complete when the following are true:

```text
[ ] Project installs from clean environment.
[ ] Panda model loads from Menagerie.
[ ] Viewer opens.
[ ] Robot initializes in neutral pose.
[ ] Joint commands are accepted.
[ ] One joint can move sinusoidally.
[ ] Joint commands are clipped safely.
[ ] FK returns end-effector pose.
[ ] Jacobian returns 3x7 and 3x7 matrices.
[ ] IK reaches a nearby Cartesian target.
[ ] PID unit tests pass.
[ ] Simulation can run headless.
[ ] Logs are written for at least one experiment.
```

---

## 20. Future Extensions

After the MVP, consider:

1. **Operational-space control**
   - Cartesian impedance.
   - Null-space posture control.

2. **Collision-aware planning**
   - Sampling-based planning.
   - Signed-distance collision checking.
   - Obstacle avoidance.

3. **Vision**
   - MuJoCo camera rendering.
   - RGB-D observation.
   - Object pose estimation.

4. **Learning**
   - Gymnasium environment wrapper.
   - Imitation learning.
   - Reinforcement learning.
   - MJX/JAX acceleration.

5. **ROS 2**
   - Joint state publisher.
   - Command subscriber.
   - TF tree.
   - MoveIt integration.

6. **Real hardware adapter**
   - Same `RobotArm` interface.
   - Hardware backend replacing MuJoCo backend.
   - Safety-critical command validation.

---

## 21. Key Design Decision

The first version shall use **MuJoCo position-controlled actuators** and focus on stable simulation, FK, IK, PID scaffolding, and task execution.

Torque control, operational-space control, perception, and hardware control should be treated as second-stage extensions.

This keeps the first build simple, testable, and useful while preserving a clean path to more advanced robotics work.
