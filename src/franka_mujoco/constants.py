from __future__ import annotations

PANDA_DOF = 7

PANDA_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, PANDA_DOF + 1))
PANDA_ACTUATOR_NAMES = tuple(f"actuator{i}" for i in range(1, PANDA_DOF + 1))
PANDA_FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")

DEFAULT_END_EFFECTOR_BODY = "hand"
