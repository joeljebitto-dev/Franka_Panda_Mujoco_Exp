from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from franka_mujoco.env import FrankaPandaEnv  # noqa: E402


def main() -> None:
    env = FrankaPandaEnv()
    state = env.state_snapshot()
    print(json.dumps(state.as_dict(), indent=2))


if __name__ == "__main__":
    main()
